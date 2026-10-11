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
    from people_mosaic.overlay import cloud
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
    for mp4 in (False, True, True):
        path = export(
            analysis,
            directory / ("sample_mosaic.mp4" if mp4 else "sample_mosaic.mkv"),
            set(),
            cloud(),
            [],
            mp4,
            lambda *_: None,
            Event(),
        )
        with av.open(str(path)) as result:
            assert len(list(result.decode(video=0))) == 12
        saved.append(str(path))
    assert Path(saved[-1]).name == "sample_mosaic_1.mp4"
    with av.open(str(video)) as original, av.open(saved[0]) as rendered:
        before = next(original.decode(video=0)).to_ndarray(format="yuv420p")
        after = next(rendered.decode(video=0)).to_ndarray(format="yuv420p")
        assert np.array_equal(before[:10, :], after[:10, :])
        assert not np.array_equal(before, after)
    print(json.dumps({"analysis_frames": len(scanned.frames), "outputs": saved}))


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
    elif sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
        smoke_test(Path(sys.argv[2]))
    else:
        from people_mosaic.app import main as gui_main

        gui_main()


if __name__ == "__main__":
    main()
