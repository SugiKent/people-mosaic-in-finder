"""Collect bundled notices and publish actual corresponding upstream sources.

The Sources ZIP is a separate release asset. A list of upstream links alone is
not substituted for the corresponding sources when shipping LGPL libraries.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

import certifi

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DISTRIBUTIONS = (
    "av",
    "certifi",
    "numpy",
    "opencv-python-headless",
    "Pillow",
    "PySide6",
    "PySide6_Essentials",
    "PySide6_Addons",
    "shiboken6",
    "pyinstaller",
)
NOTICE_PATTERN = re.compile(r"(?:licen[cs]e|copying|copyright|notice|authors)", re.I)
ZOO_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
MODEL_LICENSE_HASHES = {
    "face_detection_yunet": "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958",
    "face_recognition_sface": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request(url):
    return urllib.request.urlopen(
        url, timeout=120, context=ssl.create_default_context(cafile=certifi.where())
    )


def download(url, directory, *, filename=None, expected=None):
    directory.mkdir(parents=True, exist_ok=True)
    name = filename or url.rsplit("/", 1)[-1]
    if not name or Path(name).name != name:
        raise ValueError(f"Unsafe source archive filename: {name}")
    path = directory / name
    remembered = path.with_name(path.name + ".sha256")
    if expected is None and remembered.is_file():
        expected = remembered.read_text().strip()
    if path.exists() and expected and digest(path) == expected:
        return path
    temporary = path.with_name(path.name + ".download")
    try:
        with request(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if expected and digest(temporary) != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
        temporary.replace(path)
        remembered.write_text(digest(path) + "\n")
    finally:
        temporary.unlink(missing_ok=True)
    return path


def pypi_source(name, version):
    with request(f"https://pypi.org/pypi/{name}/{version}/json") as response:
        metadata = json.load(response)
    sources = [entry for entry in metadata["urls"] if entry["packagetype"] == "sdist"]
    if len(sources) != 1:
        raise ValueError(f"No unique exact-version source distribution for {name} {version}")
    source = sources[0]
    return source["url"], source["filename"], source["digests"]["sha256"]


def copy_source_notices(archive, output):
    """Read notices without extracting executable files or following tar symlinks."""
    copied = 0
    with tarfile.open(archive, "r:*") as source:
        for entry in source:
            parts = PurePosixPath(entry.name).parts
            if (
                not entry.isfile()
                or not parts
                or entry.name.startswith("/")
                or any(part in {".", ".."} for part in parts)
                or not NOTICE_PATTERN.search(parts[-1])
                or entry.size > 5 * 1024 * 1024
            ):
                continue
            stream = source.extractfile(entry)
            if stream is None:
                continue
            target = output / archive.name / Path(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream, target.open("wb") as destination:
                shutil.copyfileobj(stream, destination)
            copied += 1
    return copied


def installed_notices(output, distributions=RUNTIME_DISTRIBUTIONS):
    packages = []
    for name in distributions:
        distribution = importlib.metadata.distribution(name)
        destination = output / "installed" / name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "METADATA.txt").write_text(distribution.read_text("METADATA") or "")
        for file in distribution.files or ():
            if not NOTICE_PATTERN.search(Path(str(file)).name):
                continue
            original = Path(distribution.locate_file(file))
            if not original.is_file() or original.stat().st_size > 5 * 1024 * 1024:
                continue
            parts = PurePosixPath(str(file)).parts
            if any(part in {".", ".."} for part in parts) or str(file).startswith("/"):
                continue
            target = destination / Path(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        packages.append({"name": name, "version": distribution.version})
    return packages


def interpreter_notices(output):
    """Keep portable CPython's bundled dependency notices, including libffi/SSL."""
    root = Path(sys.base_prefix)
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or not NOTICE_PATTERN.search(path.name)
            or path.stat().st_size > 5 * 1024 * 1024
            or "site-packages" in path.parts
        ):
            continue
        destination = output / "python-runtime" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def model_notices(output, cache, root=ROOT):
    """Verify distributed weights and their commit-pinned redistribution notices."""
    tree = ast.parse((root / "src/people_mosaic/models.py").read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MODELS" for target in node.targets)
    )
    models = ast.literal_eval(assignment.value)
    records, sources = [], []
    for name, (model, expected) in models.items():
        weight = root / "models" / name
        if not weight.is_file() or digest(weight) != expected:
            raise ValueError(f"Bundled model checksum mismatch: {name}")
        url = f"https://raw.githubusercontent.com/opencv/opencv_zoo/{ZOO_COMMIT}/models/{model}/LICENSE"
        license_path = download(
            url,
            cache,
            filename=f"OpenCV-Zoo-{ZOO_COMMIT[:12]}-{model}-LICENSE.txt",
            expected=MODEL_LICENSE_HASHES[model],
        )
        destination = output / "models" / license_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(license_path, destination)
        records.append(
            {
                "name": name,
                "sha256": expected,
                "license_url": url,
                "license_sha256": digest(license_path),
                "zoo_commit": ZOO_COMMIT,
            }
        )
        sources.append(
            {
                "name": f"OpenCV Zoo {model} notice",
                "version": ZOO_COMMIT,
                "url": url,
                "sha256": digest(license_path),
                "archive": license_path.name,
                "path": license_path,
            }
        )
    return records, sources


def collect(output, sources_output, cache, ffmpeg_manifest):
    output.mkdir(parents=True, exist_ok=True)
    packages = installed_notices(output)
    interpreter_notices(output)
    sources = []

    def source(name, version, url, filename=None, expected=None, cached=None):
        archive = Path(cached) if cached else None
        if not archive or not archive.is_file() or (expected and digest(archive) != expected):
            archive = download(url, cache, filename=filename, expected=expected)
        notices = copy_source_notices(archive, output / "upstream")
        # SQLite is public domain and its autoconf archive may carry the notice
        # in source headers; PBS's LICENSE.sqlite.txt is included separately.
        if not notices and name != "portable Python sqlite":
            raise ValueError(f"Source archive contains no notices: {archive}")
        sources.append(
            {
                "name": name,
                "version": version,
                "url": url,
                "sha256": digest(archive),
                "archive": archive.name,
                "path": archive,
            }
        )
        print(f"Source ready: {name} {version}", flush=True)
        return archive

    from PySide6.QtCore import qVersion

    pyside = importlib.metadata.version("PySide6")
    qt_version = qVersion()
    series = ".".join(qt_version.split(".")[:2])
    source(
        "PySide6 and Shiboken6",
        pyside,
        f"https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-{pyside}-src/"
        f"pyside-setup-everywhere-src-{pyside}.tar.xz",
    )
    # Qt base and image plugins are the Qt modules used by the frozen Widgets UI.
    for module in ("qtbase", "qtsvg", "qtimageformats"):
        source(
            module,
            qt_version,
            f"https://download.qt.io/official_releases/qt/{series}/{qt_version}/submodules/"
            f"{module}-everywhere-src-{qt_version}.tar.xz",
        )
    python = ".".join(map(str, sys.version_info[:3]))
    source("CPython", python, f"https://www.python.org/ftp/python/{python}/Python-{python}.tar.xz")
    # Portable CPython's SSL modules dynamically link the bundled OpenSSL runtime.
    openssl = ssl.OPENSSL_VERSION.split()[1]
    source(
        "OpenSSL",
        openssl,
        f"https://github.com/openssl/openssl/releases/download/openssl-{openssl}/"
        f"openssl-{openssl}.tar.gz",
    )
    for name in ("av", "certifi", "numpy", "opencv-python-headless", "Pillow", "pyinstaller"):
        version = importlib.metadata.version(name)
        url, filename, expected = pypi_source(name, version)
        source(name, version, url, filename, expected)
    # Model licenses are separate from OpenCV; unrelated Zoo models are omitted.
    models, model_sources = model_notices(output, cache)
    sources.extend(model_sources)

    # uv's portable Python applies build patches and statically includes these
    # libraries. Keep its actual build recipes and hash-verified input sources.
    build_file = Path(sys.base_prefix) / "BUILD"
    if not build_file.is_file() or not re.fullmatch(r"[0-9]{8}", build_file.read_text().strip()):
        raise ValueError("Collect sources using the managed build/freeze-venv interpreter")
    python_build = build_file.read_text().strip()
    pbs = source(
        "python-build-standalone build recipes",
        python_build,
        f"https://github.com/astral-sh/python-build-standalone/archive/refs/tags/{python_build}.tar.gz",
        filename=f"python-build-standalone-{python_build}.tar.gz",
    )
    with tarfile.open(pbs, "r:*") as archive:
        member = next(
            entry for entry in archive if entry.name.endswith("/pythonbuild/downloads.py")
        )
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Portable Python source recipes are missing")
        with stream:
            tree = ast.parse(stream.read().decode())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DOWNLOADS" for target in node.targets
        )
    )
    recipes = ast.literal_eval(assignment.value)
    for name, recipe_name in (
        ("CPython", f"cpython-{sys.version_info.major}.{sys.version_info.minor}"),
        ("OpenSSL", f"openssl-{'.'.join(openssl.split('.')[:2])}"),
    ):
        recipe = recipes[recipe_name]
        actual = next(entry for entry in sources if entry["name"] == name)
        if actual["version"] != recipe["version"] or actual["sha256"] != recipe["sha256"]:
            raise ValueError(f"Portable Python build source mismatch: {name}")
    for dependency in ("bzip2", "expat", "libffi", "mpdecimal", "sqlite", "xz", "zlib"):
        entry = recipes[dependency]
        source(
            f"portable Python {dependency}",
            entry["version"],
            entry["url"],
            expected=entry["sha256"],
        )

    # These source URLs and checksums are the exact inputs of our LGPL-only build.
    ffmpeg = json.loads(ffmpeg_manifest.read_text())
    for name, entry, filename in (
        ("FFmpeg", ffmpeg, f"ffmpeg-{ffmpeg['version']}.tar.xz"),
        ("OpenH264", ffmpeg["openh264"], f"openh264-{ffmpeg['openh264']['version']}.tar.gz"),
    ):
        source(
            name,
            entry["version"],
            entry["source_url"],
            filename,
            expected=entry["source_sha256"],
            cached=ffmpeg_manifest.parent.parent / "ffmpeg-source" / filename,
        )
    manifest = {
        "packages": packages,
        "models": models,
        "sources": [
            {key: value for key, value in entry.items() if key != "path"} for entry in sources
        ],
    }
    (output / "SOURCES.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "FFMPEG-BUILD.json").write_bytes(ffmpeg_manifest.read_bytes())
    (output / "README.txt").write_text(
        "Third-party license and copyright notices\n\n"
        "Qt/PySide/Shiboken and FFmpeg are dynamically linked under the LGPL.\n"
        "The separate Sources ZIP on the same GitHub Release contains their exact\n"
        "upstream archives and our build scripts. Libraries may be replaced and\n"
        "modified for debugging under their license terms; no reverse engineering\n"
        "restriction is imposed. Re-sign a modified .app locally with codesign.\n"
        "Build the source with scripts/build_ffmpeg.py and scripts/build_bundle.py.\n"
        "https://github.com/SugiKent/people-mosaic-in-finder/releases/latest\n"
    )
    sources_output.mkdir(parents=True, exist_ok=True)
    # The source asset is identified by immutable commit, independently of app version.
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    archive_path = sources_output / f"People-Mosaic-Sources-{commit[:12]}.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in sources:
            archive.write(entry["path"], f"upstream/{entry['archive']}")
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, "notices/" + path.relative_to(output).as_posix())
        application = subprocess.run(
            ["git", "archive", "HEAD"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        archive.writestr("application-source.tar", application)
    checksum = archive_path.with_suffix(".zip.sha256")
    checksum.write_text(f"{digest(archive_path)}  {archive_path.name}\n")
    return archive_path, checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/bundle-licenses")
    parser.add_argument("--sources-output-dir", type=Path, default=ROOT / "build/releases")
    parser.add_argument("--source-cache", type=Path, default=ROOT / "build/third-party-sources")
    parser.add_argument("--ffmpeg-manifest", type=Path, default=ROOT / "build/ffmpeg/BUILD.json")
    args = parser.parse_args()
    for output in collect(
        args.output_dir, args.sources_output_dir, args.source_cache, args.ffmpeg_manifest
    ):
        print(output)


if __name__ == "__main__":
    main()
