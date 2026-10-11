import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from fractions import Fraction
from heapq import merge
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
from .overlay import SUPPORTED, PreparedOverlay, native_plane
from .runtime import tool_path
from .stabilize import stabilize_faces


class Cancelled(Exception):
    pass


def check_cancel(cancel: Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def probe(path: Path) -> dict:
    if not path.is_file():
        raise UserError("動画ファイルを選んでください。")
    executable = tool_path("ffprobe")
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
    rotated.duration = frame.duration
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
    cut_frames = set()
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
                scene_cut = (
                    previous_gray is not None and np.mean(cv2.absdiff(gray, previous_gray)) >= 45
                )
                if scene_cut:
                    cut_frames.add(index)
                if previous_gray is not None and not scene_cut:
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
    check_cancel(cancel)
    analysis.frames = stabilize_faces(analysis.frames, analysis.times, cut_frames)
    check_cancel(cancel)
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


# Limit the working set to eight native frames; each lane has a private PNG cache.
EXPORT_WORKERS = 4
EXPORT_PENDING = EXPORT_WORKERS * 2


class FrameCompositor:
    def __init__(self, rgba, rotation, hlg, pixel_format, cancel):
        self.rgba, self.rotation, self.hlg = rgba, rotation, hlg
        self.pixel_format, self.cancel = pixel_format, cancel
        self.lane = local()

    def __call__(self, original, masks):
        check_cancel(self.cancel)
        if not hasattr(self.lane, "overlay"):
            self.lane.overlay = PreparedOverlay(self.rgba)
        frame = oriented(original, self.rotation)
        self.lane.overlay.blend_native(frame, masks, self.hlg)
        check_cancel(self.cancel)
        if self.pixel_format:
            frame = frame.reformat(format=self.pixel_format)
        return frame


def composited_packets(source, video, analysis, revealed, manual, composite, cancel):
    """Parallelize composition, keeping all video/audio events in demux order."""
    pending = deque()
    executor = ThreadPoolExecutor(max_workers=EXPORT_WORKERS, thread_name_prefix="composite")
    count = 0
    try:
        for packet in source.demux():
            check_cancel(cancel)
            if packet.stream.index == video.index:
                for original in packet.decode():
                    if (
                        original.pts is None
                        or count >= len(analysis.frames)
                        or abs(float(original.pts * original.time_base) - analysis.times[count])
                        > 1e-6
                    ):
                        raise UserError(
                            "解析時と動画のフレームが一致しません。再解析してください。"
                        )
                    # Decoder reference buffers must stay immutable. FFmpeg copies
                    # shared buffers here, before any worker can change the pixels.
                    original.make_writable()
                    masks = hidden_faces(analysis.frames[count], revealed) + manual
                    pending.append((True, executor.submit(composite, original, masks)))
                    count += 1
                    if len(pending) >= EXPORT_PENDING:
                        is_video, value = pending.popleft()
                        yield is_video, value.result() if is_video else value
            elif packet.stream.type == "audio" and packet.dts is not None:
                pending.append((False, packet))
                if len(pending) >= EXPORT_PENDING:
                    is_video, value = pending.popleft()
                    yield is_video, value.result() if is_video else value
        if count != len(analysis.frames):
            raise UserError("動画の終端が解析時と一致しません。")
        while pending:
            check_cancel(cancel)
            is_video, value = pending.popleft()
            yield is_video, value.result() if is_video else value
    finally:
        for is_video, value in pending:
            if is_video:
                value.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def public_bitrate(analysis, info):
    # Hardware bitrate control differs from x264 CRF. Keep a generous floor and
    # never request less than the source bitrate; MP4 remains a lossy delivery format.
    return max(
        int(info.get("bit_rate", 0)),
        int(analysis.width * analysis.height * analysis.fps * (0.15 if analysis.hlg else 0.3)),
        4_000_000,
    )


def mp4_codec(analysis, info):
    if analysis.hlg:
        return "hevc_videotoolbox"
    if sys.platform == "darwin" and "h264_videotoolbox" in av.codecs_available:
        # Fail over before producing any frames or opening the output container.
        context = av.CodecContext.create("h264_videotoolbox", "w")
        context.width, context.height = analysis.width, analysis.height
        context.pix_fmt = "yuv420p"
        context.time_base = Fraction(1, max(1, round(analysis.fps)))
        context.bit_rate = public_bitrate(analysis, info)
        context.options = {"profile": "high", "allow_sw": "0"}
        try:
            context.open()
            return "h264_videotoolbox"
        except av.error.FFmpegError:
            pass
        finally:
            # PyAV has no CodecContext.close(); releasing its last reference
            # destroys the temporary hardware session before actual encoding.
            del context
    if "libx264" in av.codecs_available:
        return "libx264"
    if "libopenh264" in av.codecs_available:
        return "libopenh264"
    raise UserError("MP4 用のエンコーダーを確認できません。アプリを再インストールしてください。")


def remux_public_mp4(video_path: Path, audio_path: Path, output: Path, cancel: Event) -> None:
    """Copy encoded video/audio packets without CLI duration inference.

    In particular, FFmpeg 6's CLI can recalculate the final VFR video duration
    during stream copy. PyAV muxes the explicit packet durations from our master.
    Only one packet per input is prefetched; encoded video is never decoded again.
    """
    with (
        av.open(str(video_path)) as video,
        av.open(str(audio_path)) as audio,
        av.open(
            str(output),
            "w",
            options={"movflags": "+faststart", "avoid_negative_ts": "disabled"},
        ) as dest,
    ):
        streams = {}
        for source in (video, audio):
            for stream in source.streams:
                target = dest.add_stream_from_template(stream, opaque=True)
                target.codec_context.codec_tag = stream.codec_context.codec_tag
                target.metadata.clear()
                streams[(id(source), stream.index)] = target

        def packets(source):
            for packet in source.demux():
                check_cancel(cancel)
                if packet.dts is not None:
                    yield source, packet

        for source, packet in merge(
            packets(video), packets(audio), key=lambda event: event[1].dts * event[1].time_base
        ):
            check_cancel(cancel)
            packet.stream = streams[(id(source), packet.stream.index)]
            dest.mux(packet)


def publish_output(final: Path, output: Path, cancel: Event) -> Path:
    """Publish without overwriting, numbering collisions atomically."""
    index = 0
    while True:
        check_cancel(cancel)
        candidate = (
            output if index == 0 else output.with_name(f"{output.stem}_{index}{output.suffix}")
        )
        try:
            os.link(final, candidate)
            return candidate
        except FileExistsError:
            index += 1


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
    # Resolve the folder only: an existing output symlink is a name collision,
    # not permission to follow its target into another folder or the input file.
    output = output.parent.resolve() / output.name
    if output == analysis.path.resolve():
        raise UserError("元動画は上書きできません。")
    if digest(analysis.path) != analysis.fingerprint:
        raise UserError("解析後に元動画が変更されました。もう一度解析してください。")
    check_cancel(cancel)
    info = probe(analysis.path)
    copy_audio = all(codec in {"aac", "alac", "mp3"} for codec in info["audio_codecs"])
    transcode_audio = mp4 and not copy_audio
    codec = mp4_codec(analysis, info) if mp4 else "ffv1"
    # Temporary media is private and is deleted on failure/cancellation.
    with tempfile.TemporaryDirectory(prefix=".people-mosaic-", dir=output.parent) as directory:
        master = Path(directory) / ("video.mp4" if mp4 else "master.mkv")
        with (
            av.open(str(analysis.path)) as source,
            av.open(
                str(master),
                "w",
                options=(
                    {"movflags": "+faststart", "avoid_negative_ts": "disabled"}
                    if mp4 and not transcode_audio
                    else {"avoid_negative_ts": "disabled"}
                ),
            ) as dest,
        ):
            original_video = source.streams.video[0]
            # Software decoding retains exact native samples for lossless export.
            original_video.thread_type = "AUTO"
            original_video.codec_context.thread_count = 2
            video = dest.add_stream(
                codec, rate=original_video.base_rate or original_video.average_rate or 30
            )
            video.width, video.height = analysis.width, analysis.height
            video.pix_fmt = ("p010le" if analysis.hlg else "yuv420p") if mp4 else info["pix_fmt"]
            video.time_base = original_video.time_base
            video.codec_context.time_base = original_video.time_base
            video.codec_context.color_range = original_video.codec_context.color_range
            video.codec_context.colorspace = original_video.codec_context.colorspace
            video.codec_context.color_primaries = original_video.codec_context.color_primaries
            video.codec_context.color_trc = original_video.codec_context.color_trc
            if mp4 and "videotoolbox" in codec:
                video.bit_rate = public_bitrate(analysis, info)
                video.options = {"profile": "main10" if analysis.hlg else "high", "allow_sw": "0"}
                if analysis.hlg:
                    video.codec_context.codec_tag = "hvc1"
            elif codec == "libopenh264":
                video.bit_rate = public_bitrate(analysis, info)
                video.codec_context.thread_count = 4
            elif mp4:
                video.options = {"crf": "16", "preset": "medium"}
                video.codec_context.thread_count = 4
            else:
                video.options = {"level": "3", "slicecrc": "1", "slices": "16"}
                video.codec_context.thread_count = 8
            audio_map = (
                {}
                if transcode_audio
                else {
                    stream.index: dest.add_stream_from_template(stream)
                    for stream in source.streams.audio
                }
            )
            composite = FrameCompositor(
                rgba, info["rotation"], analysis.hlg, video.pix_fmt if mp4 else None, cancel
            )
            count = 0
            with closing(
                composited_packets(
                    source, original_video, analysis, revealed, manual, composite, cancel
                )
            ) as events:
                for is_video, value in events:
                    check_cancel(cancel)
                    if is_video:
                        for encoded in video.encode(value):
                            dest.mux(encoded)
                        count += 1
                        if count % 8 == 0:
                            progress(
                                int(count / len(analysis.frames) * (85 if transcode_audio else 95)),
                                f"雲を合成中 · {count:,} / {len(analysis.frames):,}",
                            )
                    elif value.stream.index in audio_map:
                        value.stream = audio_map[value.stream.index]
                        dest.mux(value)
            for encoded in video.encode():
                dest.mux(encoded)
        check_cancel(cancel)
        final = master
        if transcode_audio:
            progress(90, "MP4 の音声を変換中")
            final = Path(directory) / "public.mp4"
            # Convert audio only. A separate PyAV packet mux preserves the
            # masked video timeline even with older system FFmpeg versions.
            audio_only = Path(directory) / "audio.m4a"
            command = [
                tool_path("ffmpeg"),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-copyts",
                "-i",
                str(analysis.path),
                "-map",
                "0:a",
                "-map_metadata",
                "-1",
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-avoid_negative_ts",
                "disabled",
                "-movflags",
                "+faststart",
                str(audio_only),
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
            remux_public_mp4(master, audio_only, final, cancel)
        check_cancel(cancel)
        if digest(analysis.path) != analysis.fingerprint:
            raise UserError("書き出し中に元動画が変更されました。出力を破棄しました。")
        # Retry collisions at publication, including files created while encoding.
        output = publish_output(final, output, cancel)
    progress(100, "書き出しが完了しました")
    return output
