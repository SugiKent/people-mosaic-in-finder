"""Frozen app entry: interactive GUI, offline installation, and release checks."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def self_test():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import av
    import cv2
    from build_macos import service_update_function
    from PySide6.QtWidgets import QApplication

    from people_mosaic.models import MODELS, model_paths
    from people_mosaic.runtime import frozen, resource_root, tool_path

    application = QApplication.instance() or QApplication(sys.argv[:1])
    detector, recognizer = model_paths()
    cv2.FaceDetectorYN_create(str(detector), "", (320, 320))
    cv2.FaceRecognizerSF_create(str(recognizer), "")
    tools = {}
    for name in ("ffmpeg", "ffprobe"):
        tools[name] = subprocess.run(
            [tool_path(name), "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
    assert application.platformName() == os.environ["QT_QPA_PLATFORM"]
    assert "ffv1" in av.codecs_available
    av.Codec("prores_ks", "w")
    assert (
        "prores_ks"
        in subprocess.run(
            [tool_path("ffmpeg"), "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    print(
        json.dumps(
            {
                "frozen": frozen(),
                "resources": str(resource_root()),
                "python": sys.version.split()[0],
                "models": list(MODELS),
                "tools": tools,
                "finder_service_update_api": service_update_function().__name__,
            }
        )
    )


def smoke_test(directory):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from threading import Event

    import av
    import numpy as np

    from people_mosaic.domain import Analysis, Face
    from people_mosaic.engine import analyze, export
    from people_mosaic.models import digest
    from people_mosaic.overlay import cloud, native_plane, placement
    from people_mosaic.runtime import tool_path

    directory.mkdir(parents=True, exist_ok=True)
    video = directory / "sample.mkv"
    subprocess.run(
        [
            tool_path("ffmpeg"),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=12:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            str(video),
        ],
        check=True,
    )
    scanned = analyze(video, lambda *_: None, Event())
    assert len(scanned.frames) == 12
    with av.open(str(video)) as source:
        times = [float(frame.time) for frame in source.decode(video=0)]
    analysis = Analysis(
        video, digest(video), 320, 240, 12, [[Face((100, 70, 70, 80))] for _ in times], times
    )
    saved = []
    rgba = cloud()
    for extension in ("mov", "mp4", "mp4"):
        path = export(
            analysis,
            directory / f"sample_mosaic.{extension}",
            set(),
            rgba,
            [],
            extension == "mp4",
            lambda *_: None,
            Event(),
            mov=extension == "mov",
        )
        with av.open(str(path)) as result:
            frames = list(result.decode(video=0))
            assert len(frames) == 12
            assert np.allclose([float(frame.time) for frame in frames], times, atol=0.001)
        saved.append(str(path))
    assert Path(saved[-1]).name == "sample_mosaic_1.mp4"
    metadata = json.loads(
        subprocess.run(
            [tool_path("ffprobe"), "-v", "error", "-show_streams", "-of", "json", saved[0]],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["streams"]
    mov_video = next(stream for stream in metadata if stream["codec_type"] == "video")
    assert mov_video["codec_name"] == "prores"
    assert mov_video["profile"] == "HQ"
    assert mov_video["pix_fmt"] == "yuv422p10le"
    assert (
        next(stream for stream in metadata if stream["codec_type"] == "audio")["codec_name"]
        == "pcm_s16le"
    )
    with av.open(str(video)) as original, av.open(saved[0]) as rendered:
        before = native_plane(next(original.decode(video=0)), 0).astype(np.int32) * 4
        after = native_plane(next(rendered.decode(video=0)), 0).astype(np.int32)
        # ProRes HQ is an editing codec, not mathematically lossless. The top
        # rows sit outside the cloud: bound their 10-bit quantization error.
        difference = np.abs(before[:10, :] - after[:10, :])
        assert difference.mean() <= 4
        assert difference.max() <= 64
        assert not np.array_equal(before, after)
        x, y, patch = placement((100, 70, 70, 80), 320, 240, rgba)
        white = (patch[:, :, :3].min(axis=2) == 255) & (patch[:, :, 3] == 255)
        white_luma = None
        if white.any():
            cloud_luma = after[y : y + patch.shape[0], x : x + patch.shape[1]][white]
            white_luma = float(np.percentile(cloud_luma, 5))
            assert white_luma >= 232 * 4
    with av.open(str(video)) as original, av.open(saved[0]) as rendered:
        original_audio = np.concatenate(
            [frame.to_ndarray() for frame in original.decode(audio=0)], axis=1
        )
        rendered_audio = np.concatenate(
            [frame.to_ndarray() for frame in rendered.decode(audio=0)], axis=1
        )
        assert np.array_equal(original_audio, rendered_audio)
    print(
        json.dumps(
            {
                "analysis_frames": len(scanned.frames),
                "outputs": saved,
                "mov_codec": mov_video["codec_name"],
                "mov_profile": mov_video["profile"],
                "mov_pixel_format": mov_video["pix_fmt"],
                "untouched_luma_mean_error_10bit": float(difference.mean()),
                "white_luma_5th_percentile_10bit": white_luma,
            }
        )
    )


def main():
    if "--install" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--install", type=Path, required=True)
        parser.add_argument("--home", type=Path)
        args = parser.parse_args()
        from install_release import finish_install, install

        try:
            app = install(args.install, args.home or Path.home())
            if args.home is None:
                finish_install(app)
            else:
                # Verification uses an isolated HOME and must not refresh the
                # real user's services database or open Finder.
                print(f"インストール完了: {app}")
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            raise SystemExit(f"インストールを完了できませんでした: {error}") from error
    elif sys.argv[1:] == ["--refresh-services"]:
        from build_macos import refresh_services

        refresh_services()
        print("Finder のサービス一覧を更新しました。")
    elif sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
        smoke_test(Path(sys.argv[2]))
    else:
        from people_mosaic.app import main as gui_main

        gui_main()


if __name__ == "__main__":
    main()
