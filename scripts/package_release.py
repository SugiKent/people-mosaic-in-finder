"""Build a reproducible source installer ZIP from a committed Git revision.

Only the installer and runtime source allowlist is published. Local environments,
models, videos, generated images, and uncommitted changes are never included.
"""

import argparse
import hashlib
import io
import json
import re
import stat
import subprocess
import tarfile
import tomllib
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = frozenset(
    {
        "Install.command",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "scripts/setup.command",
        "scripts/launch.command",
        "scripts/install_finder.command",
        "scripts/build_macos.py",
        "scripts/install_release.py",
        "src/people_mosaic/__init__.py",
        "src/people_mosaic/app.py",
        "src/people_mosaic/models.py",
    }
)
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?")


def is_allowed(path: str) -> bool:
    """Restrict publishing to explicit installer files and Python runtime source."""
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or any(part in {".", ".."} for part in parts):
        return False
    if "\\" in path or PurePosixPath(path).as_posix() != path:
        return False
    return path in REQUIRED_FILES or (
        len(parts) >= 3
        and parts[:2] == ("src", "people_mosaic")
        and path.endswith(".py")
        and "__pycache__" not in parts
    )


def resolve_revision(root: Path, ref: str) -> str:
    """Resolve a release revision to its full immutable commit identifier."""
    return subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_files(root: Path, ref: str) -> dict[str, tuple[bytes, int]]:
    """Read tracked source from Git; never inspect working-directory assets."""
    revision = resolve_revision(root, ref)
    # Pathspecs also keep unrelated committed assets out of the intermediate tar.
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision, *sorted(REQUIRED_FILES), "src/people_mosaic"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    entries = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        for item in source:
            if not is_allowed(item.name):
                continue
            if not item.isfile():
                if item.isdir():
                    continue
                raise ValueError(f"Release source must be a regular file: {item.name}")
            contents = source.extractfile(item)
            if contents is None:
                raise ValueError(f"Cannot read release source: {item.name}")
            entries[item.name] = (contents.read(), item.mode)
    return entries


def package_files(
    entries: dict[str, tuple[bytes, int]], output_dir: Path, ref: str = "HEAD", *, commit: str
) -> tuple[Path, Path]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Release commit must be a full Git SHA-1 identifier")
    files = {path: entry for path, entry in entries.items() if is_allowed(path)}
    missing = REQUIRED_FILES - files.keys()
    if missing:
        raise ValueError(f"Missing release files: {', '.join(sorted(missing))}")
    version = tomllib.loads(files["pyproject.toml"][0].decode("utf-8"))["project"]["version"]
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Project version must be a safe semantic version")
    if ref.startswith("v") and ref != f"v{version}":
        raise ValueError(f"Release tag {ref} does not match project version {version}")
    for path, (_, mode) in files.items():
        if path.endswith(".command") and not mode & stat.S_IXUSR:
            raise ValueError(f"Installer command must be executable in Git: {path}")
    manifest = {"format": 1, "version": version, "commit": commit}
    files["RELEASE.json"] = (
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        0o644,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"People-Mosaic-{version}-macOS-arm64.zip"
    prefix = f"People-Mosaic-{version}/"
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path, (contents, mode) in sorted(files.items()):
            item = ZipInfo(prefix + path, date_time=(1980, 1, 1, 0, 0, 0))
            item.create_system = 3
            permissions = 0o755 if mode & stat.S_IXUSR else 0o644
            item.external_attr = (stat.S_IFREG | permissions) << 16
            item.compress_type = ZIP_DEFLATED
            archive.writestr(item, contents, compresslevel=9)
    checksum = output.with_suffix(".zip.sha256")
    checksum.write_text(f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n")
    return output, checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD", help="Committed Git revision (e.g. v0.1.0)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/releases")
    args = parser.parse_args()
    try:
        revision = resolve_revision(ROOT, args.ref)
        outputs = package_files(
            git_files(ROOT, revision), args.output_dir, args.ref, commit=revision
        )
    except (ValueError, KeyError, tomllib.TOMLDecodeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release packaging failed: {error}\n")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
