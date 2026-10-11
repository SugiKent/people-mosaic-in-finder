import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from itertools import chain
from pathlib import Path
from threading import Event, local

import av
import cv2
import numpy as np
from av.codec.hwaccel import HWAccel, hwdevices_available

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


# Four two-thread inference lanes target the eight performance cores of M1 Max.
# Each lane owns
# its networks; sharing a FaceDetectorYN/FaceRecognizerSF across threads is unsafe.
ANALYSIS_WORKERS = 4
OPENCV_THREADS = 2
TRACKING_LONG_EDGE = 960


def tracking_gray(bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    scale = min(1.0, TRACKING_LONG_EDGE / max(width, height))
    if scale < 1:
        bgr = cv2.resize(
            bgr,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def follow_all(
    previous_gray: np.ndarray, gray: np.ndarray, faces: list[Face], width: int, height: int
) -> list[tuple]:
    """Build optical-flow pyramids once for all faces, in a small tracking image."""
    if not faces:
        return []
    scale = np.array([gray.shape[1] / width, gray.shape[0] / height], np.float32)
    points = (
        np.array(
            [
                [[x + w * fx, y + h * fy]]
                for face in faces
                for x, y, w, h in [face.box]
                for fx in (0.2, 0.5, 0.8)
                for fy in (0.2, 0.5, 0.8)
            ],
            dtype=np.float32,
        )
        * scale
    )
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None)
    boxes = []
    for index, face in enumerate(faces):
        box = face.box
        if moved is not None and status is not None:
            part = slice(index * 9, (index + 1) * 9)
            valid = status[part].ravel() == 1
            if int(valid.sum()) >= 4:
                delta = np.median((moved[part] - points[part])[valid], axis=0).ravel() / scale
                if np.all(np.isfinite(delta)) and np.linalg.norm(delta) <= max(box[2:]):
                    box = (box[0] + float(delta[0]), box[1] + float(delta[1]), *box[2:])
        boxes.append(box)
    return boxes


class FrameAnalyzer:
    def __init__(
        self,
        detector_path: Path,
        recognizer_path: Path,
        rotation: int,
        cancel: Event,
        max_edge: int | None = None,
    ):
        self.detector_path, self.recognizer_path = detector_path, recognizer_path
        self.rotation, self.cancel = rotation, cancel
        self.max_edge = max_edge
        self.lane = local()

    def __call__(self, original: av.VideoFrame) -> tuple:
        check_cancel(self.cancel)
        if not hasattr(self.lane, "detector"):
            self.lane.detector = cv2.FaceDetectorYN.create(
                str(self.detector_path), "", (320, 320), 0.65, 0.3, 5000
            )
            self.lane.recognizer = cv2.FaceRecognizerSF.create(str(self.recognizer_path), "")
        bgr = original.to_ndarray(format="bgr24")
        if self.rotation:
            bgr = np.ascontiguousarray(np.rot90(bgr, self.rotation))
        height, width = bgr.shape[:2]
        detection_image = bgr
        if self.max_edge and max(width, height) > self.max_edge:
            scale = self.max_edge / max(width, height)
            detection_image = cv2.resize(
                bgr,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        dh, dw = detection_image.shape[:2]
        self.lane.detector.setInputSize((dw, dh))
        _, detections = self.lane.detector.detect(detection_image)
        if detections is not None and (dw, dh) != (width, height):
            # Restore boxes and all five landmarks before identity alignment.
            detections[:, :14:2] *= width / dw
            detections[:, 1:14:2] *= height / dh
        found = []
        for detection in detections if detections is not None else []:
            check_cancel(self.cancel)
            face = Face(tuple(float(value) for value in detection[:4]))
            aligned = None
            if min(face.box[2:]) >= 32 and detection[-1] >= 0.85:
                aligned = self.lane.recognizer.alignCrop(bgr, detection)
                feature = self.lane.recognizer.feature(aligned).ravel().copy()
                feature /= max(float(np.linalg.norm(feature)), 1e-10)
                face.feature = feature
            found.append((face, aligned))
        return float(original.pts * original.time_base), found, tracking_gray(detection_image)


@contextmanager
def analysis_decoder(path: Path, hardware: bool):
    """Use Apple media decoding for analysis, falling back before any results."""
    use_hardware = hardware and sys.platform == "darwin" and "videotoolbox" in hwdevices_available()
    container = None
    try:
        for accelerated in [True, False] if use_hardware else [False]:
            try:
                options = (
                    {"hwaccel": HWAccel("videotoolbox", allow_software_fallback=True)}
                    if accelerated
                    else {}
                )
                container = av.open(str(path), **options)
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                stream.codec_context.thread_count = 2
                decoded = iter(container.decode(stream))
                first = next(decoded, None)
                break
            except av.error.FFmpegError:
                if container is not None:
                    container.close()
                    container = None
                if not accelerated:
                    raise
        yield stream, chain([first], decoded) if first is not None else iter(())
    finally:
        if container is not None:
            container.close()


def analyzed_frames(decoded: Iterator, analyze_frame: Callable, cancel: Event) -> Iterator[tuple]:
    """Bound queued 4K frames and preserve decode order, including VFR timestamps."""
    pending = deque()
    executor = ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS, thread_name_prefix="face")
    try:
        for original in decoded:
            check_cancel(cancel)
            if original.pts is None:
                raise UserError("時刻情報のない動画は処理できません。")
            pending.append(executor.submit(analyze_frame, original))
            if len(pending) >= ANALYSIS_WORKERS * 2:
                yield pending.popleft().result()
        while pending:
            check_cancel(cancel)
            yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def analyze(
    path: Path,
    progress: Callable,
    cancel: Event,
    max_edge: int | None = 1920,
    *,
    hardware: bool = True,
) -> Analysis:
    info = probe(path)
    fingerprint = digest(path)
    detector_path, recognizer_path = model_paths()
    cv2.setNumThreads(OPENCV_THREADS)
    analyze_frame = FrameAnalyzer(
        detector_path, recognizer_path, info["rotation"], cancel, max_edge
    )
    previous_gray = None
    held: list[tuple[Face, float]] = []
    with analysis_decoder(path, hardware) as (stream, decoded):
        fps = float(stream.average_rate or stream.base_rate or 30)
        duration = float(stream.duration * stream.time_base) if stream.duration else 0
        width, height = info["width"], info["height"]
        if info["rotation"] % 2:
            width, height = height, width
        analysis = Analysis(path, fingerprint, width, height, fps)
        analysis.hlg = info.get("color_transfer") == "arib-std-b67"
        with closing(analyzed_frames(decoded, analyze_frame, cancel)) as results:
            for index, (timestamp, found, gray) in enumerate(results):
                check_cancel(cancel)
                if analysis.times and timestamp <= analysis.times[-1]:
                    raise UserError("動画の時刻情報が不正です。")
                faces = []
                for face, aligned in found:
                    feature = face.feature
                    if feature is not None:
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
                    eligible = [
                        (face, last_seen)
                        for face, last_seen in held
                        if timestamp - last_seen <= 0.5
                    ]
                    boxes = follow_all(
                        previous_gray, gray, [face for face, _ in eligible], width, height
                    )
                    for box, (_, last_seen) in zip(boxes, eligible, strict=True):
                        moved = Face(box)
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
