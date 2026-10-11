"""Build pinned, LGPL-only FFmpeg shared libraries and tools for Apple Silicon.

Only Apple system libraries are used. The same prefix can be used to build PyAV
from source, avoiding GPL encoders bundled by its prebuilt macOS wheel.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
VERSION = "8.1.3"
SOURCE_URL = f"https://ffmpeg.org/releases/ffmpeg-{VERSION}.tar.xz"
SOURCE_SHA256 = "7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3"
OPENH264_VERSION = "2.6.0"
OPENH264_URL = f"https://github.com/cisco/openh264/archive/refs/tags/v{OPENH264_VERSION}.tar.gz"
OPENH264_SHA256 = "558544ad358283a7ab2930d69a9ceddf913f4a51ee9bf1bfb9e377322af81a69"
CONFIGURE_FLAGS = [
    "--arch=arm64",
    "--target-os=darwin",
    "--cc=clang",
    "--enable-shared",
    "--disable-static",
    "--disable-gpl",
    "--disable-version3",
    "--disable-nonfree",
    "--disable-autodetect",
    "--disable-network",
    "--disable-ffplay",
    "--disable-doc",
    "--disable-debug",
    "--enable-libopenh264",
    "--enable-videotoolbox",
    "--enable-audiotoolbox",
    "--enable-neon",
    "--enable-pthreads",
    "--extra-cflags=-mmacosx-version-min=15.0",
    "--extra-ldflags=-mmacosx-version-min=15.0",
]


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(target: Path, url: str = SOURCE_URL, expected: str = SOURCE_SHA256) -> None:
    if target.is_file() and sha256(target) == expected:
        return
    temporary = target.with_suffix(".download")
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        if sha256(temporary) != expected:
            raise ValueError(f"Official source checksum mismatch: {url}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def build_openh264(prefix: Path, work: Path, jobs: int) -> None:
    archive = work / f"openh264-{OPENH264_VERSION}.tar.gz"
    download(archive, OPENH264_URL, OPENH264_SHA256)
    source = work / f"openh264-{OPENH264_VERSION}"
    if not source.is_dir():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(work, filter="data")
    env = os.environ.copy()
    env["MACOSX_DEPLOYMENT_TARGET"] = "15.0"
    env["CFLAGS"] = "-mmacosx-version-min=15.0"
    env["CXXFLAGS"] = "-mmacosx-version-min=15.0"
    env["LDFLAGS"] = "-mmacosx-version-min=15.0"
    options = [f"PREFIX={prefix}", "ARCH=arm64", "OS=darwin"]
    subprocess.run(["make", f"-j{jobs}", "libraries", *options], cwd=source, env=env, check=True)
    subprocess.run(["make", "install-shared", *options], cwd=source, env=env, check=True)
    # The upstream dylib defaults to @rpath with no search path in FFmpeg tools.
    dylib = prefix / "lib/libopenh264.dylib"
    subprocess.run(
        ["install_name_tool", "-id", str(dylib.resolve()), str(dylib.resolve())], check=True
    )


def build(prefix: Path, work: Path, jobs: int) -> Path:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError("FFmpeg build requires native Apple Silicon macOS")
    pkg_config = shutil.which("pkg-config")
    if pkg_config is None:
        raise ValueError("Build dependency pkg-config is required (brew install pkgconf)")
    prefix = prefix.resolve()
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    build_openh264(prefix, work, jobs)
    archive = work / f"ffmpeg-{VERSION}.tar.xz"
    download(archive)
    source = work / f"ffmpeg-{VERSION}"
    if not source.is_dir():
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(work, filter="data")
    command = [
        str(source / "configure"),
        f"--prefix={prefix}",
        f"--pkg-config={pkg_config}",
        *CONFIGURE_FLAGS,
    ]
    env = os.environ.copy()
    env["MACOSX_DEPLOYMENT_TARGET"] = "15.0"
    env["PKG_CONFIG_PATH"] = str(prefix / "lib/pkgconfig")
    subprocess.run(command, cwd=source, env=env, check=True)
    subprocess.run(["make", f"-j{jobs}"], cwd=source, env=env, check=True)
    subprocess.run(["make", "install"], cwd=source, env=env, check=True)
    provenance = {
        "format": 1,
        "version": VERSION,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "configure": command[1:],
        "license": "LGPL-2.1-or-later",
        "openh264": {
            "version": OPENH264_VERSION,
            "source_url": OPENH264_URL,
            "source_sha256": OPENH264_SHA256,
            "license": "BSD-2-Clause",
        },
    }
    (prefix / "BUILD.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    license_dir = prefix / "licenses"
    license_dir.mkdir(exist_ok=True)
    shutil.copy2(source / "COPYING.LGPLv2.1", license_dir)
    shutil.copy2(source / "LICENSE.md", license_dir)
    shutil.copy2(work / f"openh264-{OPENH264_VERSION}/LICENSE", license_dir / "OpenH264-LICENSE")
    print(f"FFmpeg: {prefix / 'bin/ffmpeg'}")
    print(f"Source: {archive}")
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=ROOT / "build/ffmpeg")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build/ffmpeg-source")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 12))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(args.prefix, args.work_dir, args.jobs)


if __name__ == "__main__":
    main()
