import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from threading import Event

import av
import cv2
import numpy as np

from .color import display_bgr
from .domain import CLUSTER_THRESHOLD, Analysis, Face, Person, hidden_faces, identify, iou
from .errors import UserError
from .models import digest, model_paths
from .overlay import SUPPORTED, blend_native, native_plane


class Cancelled(Exception):
    pass


def check_cancel(cancel: Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def probe(path: Path) -> dict:
    if not path.is_file():
        raise UserError("動画ファイルを選んでください。")
    executable = shutil.which("ffprobe")
    if not executable:
        raise UserError("FFmpeg が必要です。brew install ffmpeg を実行してください。")
    result = subprocess.run(
        [executable, "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        timeout=60,
    )
    if result.returncode:
        raise UserError("動画を開けませんでした。対応する動画ファイルを選んでください。")
    videos = [s for s in json.loads(result.stdout)["streams"] if s["codec_type"] == "video"]
    if len(videos) != 1:
        raise UserError("映像が 1 トラックの動画を選んでください。")
    video = videos[0]
    if video.get("pix_fmt") not in SUPPORTED:
        raise UserError("SDR の YUV 420/422/444（8/10 bit）に対応しています。")
    if video.get("color_transfer") == "smpte2084" or video.get("codec_tag_string") in {
        "dvh1",
        "dvhe",
    }:
        raise UserError(
            "PQ HDR / Dolby Vision は未対応です。SDR または HLG HDR の動画を選んでください。"
        )
    if video.get("color_transfer") == "arib-std-b67" and (
        video.get("pix_fmt") != "yuv420p10le"
        or video.get("color_space") != "bt2020nc"
        or video.get("color_primaries") != "bt2020"
    ):
        raise UserError("HLG は BT.2020 の YUV 420 10bit に対応しています。")
    video["audio_codecs"] = [
        s.get("codec_name")
        for s in json.loads(result.stdout)["streams"]
        if s["codec_type"] == "audio"
    ]
    if video.get("field_order", "progressive") not in {"progressive", "unknown"}:
        raise UserError("インターレース動画は未対応です。")
    if video["width"] % 2 or video["height"] % 2:
        raise UserError("縦横が偶数ピクセルの動画を選んでください。")
    if video.get("sample_aspect_ratio", "1:1") not in {"1:1", "N/A", "0:1"}:
        raise UserError("正方形ピクセルの動画を選んでください。")
    angle = next(
        (s.get("rotation", 0) for s in video.get("side_data_list", []) if "rotation" in s), 0
    )
    if angle % 90:
        raise UserError("90 度単位以外の回転情報を持つ動画は未対応です。")
    if angle % 180 and "422" in video["pix_fmt"]:
        raise UserError("回転した YUV 422 動画は未対応です。")
    video["rotation"] = round(angle / 90) % 4
    return video


def oriented(frame: av.VideoFrame, rotation: int) -> av.VideoFrame:
    if not rotation:
        return frame
    width, height = (frame.height, frame.width) if rotation % 2 else (frame.width, frame.height)
    rotated = av.VideoFrame(width, height, frame.format.name)
    for index in range(3):
        native_plane(rotated, index)[:] = np.rot90(native_plane(frame, index), rotation)
    rotated.pts, rotated.time_base = frame.pts, frame.time_base
    rotated.colorspace, rotated.color_range = frame.colorspace, frame.color_range
    return rotated


def follow(previous_gray: np.ndarray, gray: np.ndarray, face: Face) -> tuple:
    x, y, w, h = face.box
    points = np.array(
        [[[x + w * fx, y + h * fy]] for fx in (0.2, 0.5, 0.8) for fy in (0.2, 0.5, 0.8)],
        dtype=np.float32,
    )
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None)
    if moved is None or status is None or int(status.sum()) < 4:
        return face.box
    delta = np.median((moved - points)[status.ravel() == 1], axis=0).ravel()
    if np.linalg.norm(delta) > max(w, h):
        return face.box
    return (x + float(delta[0]), y + float(delta[1]), w, h)


def analyze(path: Path, progress: Callable, cancel: Event) -> Analysis:
    info = probe(path)
    fingerprint = digest(path)
    detector_path, recognizer_path = model_paths()
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.65, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    previous_gray = None
    held: list[tuple[Face, float]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 30)
        duration = float(stream.duration * stream.time_base) if stream.duration else 0
        width, height = info["width"], info["height"]
        if info["rotation"] % 2:
            width, height = height, width
        analysis = Analysis(path, fingerprint, width, height, fps)
        analysis.hlg = info.get("color_transfer") == "arib-std-b67"
        for index, original in enumerate(container.decode(stream)):
            check_cancel(cancel)
            if original.pts is None:
                raise UserError("時刻情報のない動画は処理できません。")
            timestamp = float(original.pts * original.time_base)
            if analysis.times and timestamp <= analysis.times[-1]:
                raise UserError("動画の時刻情報が不正です。")
            frame = oriented(original, info["rotation"])
            bgr = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            detector.setInputSize((width, height))
            _, detections = detector.detect(bgr)
            faces = []
            for detection in detections if detections is not None else []:
                box = tuple(float(value) for value in detection[:4])
                face = Face(box)
                if min(box[2:]) >= 32 and detection[-1] >= 0.85:
                    aligned = recognizer.alignCrop(bgr, detection)
                    feature = recognizer.feature(aligned).ravel().copy()
                    feature /= max(float(np.linalg.norm(feature)), 1e-10)
                    person, confident = identify(feature, analysis.people)
                    scores = [float(feature @ p.feature) for p in analysis.people]
                    if (
                        person is None
                        and (not scores or max(scores) < CLUSTER_THRESHOLD)
                        and len(analysis.people) < 256
                    ):
                        person = len(analysis.people)
                        analysis.people.append(Person(feature, aligned.copy(), index))
                        confident = True
                    elif person is not None:
                        analysis.people[person].count += 1
                    face.person, face.confident, face.feature = person, confident, feature
                faces.append(face)
            # Tracking never grants an exemption. A missed/reacquired face is hidden.
            next_held = [(face, timestamp) for face in faces]
            if previous_gray is not None and np.mean(cv2.absdiff(gray, previous_gray)) < 45:
                for old, last_seen in held:
                    if timestamp - last_seen > 0.5:
                        continue
                    moved = Face(follow(previous_gray, gray, old))
                    if not any(iou(moved.box, face.box) > 0.2 for face in faces):
                        faces.append(moved)
                        next_held.append((moved, last_seen))
            held, previous_gray = next_held, gray
            analysis.frames.append(faces)
            analysis.times.append(timestamp)
            if index % 8 == 0:
                relative = timestamp - analysis.times[0]
                progress(
                    min(95, int(relative / duration * 95)) if duration else 0,
                    f"顔を検出中 · {index + 1:,} フレーム",
                )
    if not analysis.frames:
        raise UserError("映像フレームがありません。")
    # Later clusters can make an earlier classification ambiguous: re-evaluate all.
    for faces in analysis.frames:
        for face in faces:
            if face.feature is not None:
                face.person, face.confident = identify(face.feature, analysis.people)
        ids = [face.person for face in faces if face.person is not None]
        for face in faces:
            if face.person is not None and ids.count(face.person) > 1:
                face.confident = False
            face.feature = None
    if digest(path) != fingerprint:
        raise UserError("解析中に元動画が変更されました。もう一度選んでください。")
    return analysis


def preview_frame(analysis: Analysis, index: int) -> np.ndarray:
    info = probe(analysis.path)
    with av.open(str(analysis.path)) as container:
        stream = container.streams.video[0]
        target = analysis.times[index]
        container.seek(int(target / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if float(frame.pts * frame.time_base) >= target - 1e-6:
                return display_bgr(oriented(frame, info["rotation"]), analysis.hlg)
    raise UserError("プレビューを読み込めませんでした。")


def export(
    analysis: Analysis,
    output: Path,
    revealed: set[int],
    rgba: np.ndarray,
    manual: list[Face],
    mp4: bool,
    progress: Callable,
    cancel: Event,
) -> Path:
    output = output.resolve()
    if output == analysis.path.resolve() or output.exists():
        raise UserError(
            "元動画や既存ファイルは上書きできません。"
            "同名の出力ファイルを移動または名前変更してから再実行してください。"
        )
    if digest(analysis.path) != analysis.fingerprint:
        raise UserError("解析後に元動画が変更されました。もう一度解析してください。")
    info = probe(analysis.path)
    direct_mp4 = mp4 and all(codec in {"aac", "alac", "mp3"} for codec in info["audio_codecs"])
    # Temporary media is private and is deleted on failure/cancellation.
    with tempfile.TemporaryDirectory(prefix=".people-mosaic-", dir=output.parent) as directory:
        master = Path(directory) / ("public.mp4" if direct_mp4 else "master.mkv")
        with (
            av.open(str(analysis.path)) as source,
            av.open(
                str(master),
                "w",
                options=(
                    {"movflags": "+faststart", "avoid_negative_ts": "disabled"}
                    if direct_mp4
                    else {"avoid_negative_ts": "disabled"}
                ),
            ) as dest,
        ):
            original_video = source.streams.video[0]
            codec = "ffv1"
            if direct_mp4:
                codec = "hevc_videotoolbox" if analysis.hlg else "libx264"
            video = dest.add_stream(
                codec, rate=original_video.base_rate or original_video.average_rate or 30
            )
            video.width, video.height = analysis.width, analysis.height
            video.pix_fmt = (
                ("p010le" if analysis.hlg else "yuv420p") if direct_mp4 else info["pix_fmt"]
            )
            video.time_base = original_video.time_base
            video.codec_context.time_base = original_video.time_base
            video.codec_context.color_range = original_video.codec_context.color_range
            video.codec_context.colorspace = original_video.codec_context.colorspace
            video.codec_context.color_primaries = original_video.codec_context.color_primaries
            video.codec_context.color_trc = original_video.codec_context.color_trc
            if direct_mp4 and analysis.hlg:
                video.bit_rate = max(
                    int(info.get("bit_rate", 0)),
                    int(analysis.width * analysis.height * analysis.fps * 0.15),
                )
                video.options = {"profile": "main10"}
                video.codec_context.codec_tag = "hvc1"
            elif direct_mp4:
                video.options = {"crf": "16", "preset": "medium"}
            else:
                video.options = {"level": "3", "slicecrc": "1"}
            audio_map = {
                stream.index: dest.add_stream_from_template(stream)
                for stream in source.streams.audio
            }
            count = 0
            for packet in source.demux():
                check_cancel(cancel)
                if packet.stream.index in audio_map:
                    if packet.dts is not None:
                        packet.stream = audio_map[packet.stream.index]
                        dest.mux(packet)
                elif packet.stream.index == original_video.index:
                    for original in packet.decode():
                        if (
                            count >= len(analysis.frames)
                            or abs(float(original.pts * original.time_base) - analysis.times[count])
                            > 1e-6
                        ):
                            raise UserError(
                                "解析時と動画のフレームが一致しません。再解析してください。"
                            )
                        frame = oriented(original, info["rotation"])
                        masks = hidden_faces(analysis.frames[count], revealed) + manual
                        blend_native(frame, masks, rgba, analysis.hlg)
                        if direct_mp4:
                            frame = frame.reformat(format=video.pix_fmt)
                        for encoded in video.encode(frame):
                            dest.mux(encoded)
                        count += 1
                        if count % 8 == 0:
                            progress(
                                int(
                                    count
                                    / len(analysis.frames)
                                    * (65 if mp4 and not direct_mp4 else 95)
                                ),
                                f"雲を合成中 · {count:,} / {len(analysis.frames):,}",
                            )
            if count != len(analysis.frames):
                raise UserError("動画の終端が解析時と一致しません。")
            for encoded in video.encode():
                dest.mux(encoded)
        check_cancel(cancel)
        final = master
        if mp4 and not direct_mp4:
            progress(70, "公開用 MP4 を作成中")
            final = Path(directory) / "public.mp4"
            command = [
                shutil.which("ffmpeg") or "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(master),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-map_metadata",
                "-1",
                "-c:v",
                "libx264",
                "-crf",
                "16",
                "-preset",
                "slow",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "passthrough",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-movflags",
                "+faststart",
                str(final),
            ]
            if analysis.hlg:
                video_index = command.index("-c:v")
                pixel_index = command.index("-fps_mode")
                command[video_index:pixel_index] = [
                    "-c:v",
                    "hevc_videotoolbox",
                    "-profile:v",
                    "main10",
                    "-pix_fmt",
                    "p010le",
                    "-b:v",
                    str(
                        max(
                            int(info.get("bit_rate", 0)),
                            int(analysis.width * analysis.height * analysis.fps * 0.15),
                        )
                    ),
                    "-tag:v",
                    "hvc1",
                    "-color_primaries",
                    "bt2020",
                    "-color_trc",
                    "arib-std-b67",
                    "-colorspace",
                    "bt2020nc",
                ]
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stderr=errors, stdout=subprocess.DEVNULL)
                try:
                    while True:
                        check_cancel(cancel)
                        try:
                            result = process.wait(timeout=0.2)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                    if result:
                        raise UserError(
                            "MP4 の書き出しに失敗しました。保存用 MKV を選ぶか入力を確認してください。"
                        )
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
        check_cancel(cancel)
        if digest(analysis.path) != analysis.fingerprint:
            raise UserError("書き出し中に元動画が変更されました。出力を破棄しました。")
        # Hard link is atomic and refuses overwrite, including a raced-in destination.
        os.link(final, output)
    progress(100, "書き出しが完了しました")
    return output
