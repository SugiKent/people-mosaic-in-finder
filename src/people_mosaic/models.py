"""Only public model weights are downloaded; user media never leaves this machine."""

import hashlib
import ssl
import urllib.request
from pathlib import Path

import certifi

from .errors import UserError

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "face_detection_yunet",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "face_recognition_sface",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def model_paths() -> tuple[Path, Path]:
    paths = tuple(MODEL_DIR / name for name in MODELS)
    for path in paths:
        if not path.is_file() or digest(path) != MODELS[path.name][1]:
            raise UserError(
                "顔検出モデルが未準備です。先に scripts/setup.command を実行してください。"
            )
    return paths


def download() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    for name, (folder, expected) in MODELS.items():
        target = MODEL_DIR / name
        if target.is_file() and digest(target) == expected:
            continue
        url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{folder}/{name}"
        temporary = target.with_suffix(".download")
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            with (
                urllib.request.urlopen(url, timeout=120, context=context) as response,
                temporary.open("wb") as file,
            ):
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
                    if file.tell() > 50 * 1024 * 1024:
                        raise ValueError("モデルのサイズが想定を超えました。")
            if digest(temporary) != expected:
                raise ValueError("モデルの検証に失敗しました。")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Ready: {name}")


if __name__ == "__main__":
    download()
