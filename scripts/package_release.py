"""Package an Apple Silicon app bundle and its offline installer into a ZIP.

Runtime files come only from the explicitly supplied app bundle. Installer text
comes from the requested committed Git revision, never local working files.
"""

import argparse
import hashlib
import io
import json
import plistlib
import re
import stat
import subprocess
import tarfile
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = frozenset({"Install.command", "README.md", "pyproject.toml"})
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?")


def resolve_revision(root: Path, ref: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_files(root: Path, ref: str) -> dict[str, tuple[bytes, int]]:
    revision = resolve_revision(root, ref)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision, *sorted(REQUIRED_FILES)],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    entries = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        for item in source:
            if item.name not in REQUIRED_FILES:
                continue
            if not item.isfile():
                raise ValueError(f"Release source must be a regular file: {item.name}")
            contents = source.extractfile(item)
            if contents is None:
                raise ValueError(f"Cannot read release source: {item.name}")
            mode = 0o755 if item.mode & stat.S_IXUSR else 0o644
            entries[item.name] = (contents.read(), mode)
    return entries


def bundle_files(bundle: Path, version: str, commit: str) -> list[tuple[Path, int]]:
    if bundle.is_symlink() or not bundle.is_dir() or bundle.name != "People Mosaic.app":
        raise ValueError("Bundle must be a regular People Mosaic.app directory")
    info_path = bundle / "Contents/Info.plist"
    executable = bundle / "Contents/MacOS/PeopleMosaic"
    try:
        info = plistlib.loads(info_path.read_bytes())
        manifest = json.loads((bundle / "Contents/Resources/RELEASE.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Bundle metadata is missing or invalid") from error
    if (
        info.get("CFBundleIdentifier") != "local.people-mosaic.app"
        or info.get("CFBundleExecutable") != "PeopleMosaic"
        or info.get("CFBundleShortVersionString") != version
    ):
        raise ValueError("Bundle identity or version does not match release source")
    if manifest != {
        "format": 2,
        "kind": "app-bundle",
        "version": version,
        "commit": commit,
        "dirty": False,
    }:
        raise ValueError("Bundle manifest does not match release commit")
    if not executable.is_file() or not executable.stat().st_mode & stat.S_IXUSR:
        raise ValueError("Bundle executable is missing or not executable")
    root = bundle.resolve()
    result = [(bundle, stat.S_IFDIR | 0o755)]
    for path in sorted(bundle.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            target = path.readlink()
            if target.is_absolute():
                raise ValueError(f"Bundle symlink must be relative: {path}")
            try:
                path.resolve(strict=True).relative_to(root)
            except (OSError, ValueError, RuntimeError) as error:
                raise ValueError(
                    f"Bundle symlink leaves the bundle or is broken: {path}"
                ) from error
            result.append((path, stat.S_IFLNK | 0o777))
        elif stat.S_ISREG(mode):
            result.append((path, stat.S_IFREG | (0o755 if mode & stat.S_IXUSR else 0o644)))
        elif stat.S_ISDIR(mode):
            result.append((path, stat.S_IFDIR | 0o755))
        else:
            raise ValueError(f"Bundle contains a special filesystem entry: {path}")
    return result


def zip_info(name: str, mode: int) -> ZipInfo:
    item = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    item.create_system = 3
    item.external_attr = mode << 16
    item.compress_type = ZIP_DEFLATED
    return item


def package_files(
    entries: dict[str, tuple[bytes, int]],
    output_dir: Path,
    ref: str = "HEAD",
    *,
    commit: str,
    bundle: Path,
) -> tuple[Path, Path]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Release commit must be a full Git SHA-1 identifier")
    missing = REQUIRED_FILES - entries.keys()
    if missing:
        raise ValueError(f"Missing release files: {', '.join(sorted(missing))}")
    version = tomllib.loads(entries["pyproject.toml"][0].decode())["project"]["version"]
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Project version must be a safe semantic version")
    if ref.startswith("v") and ref != f"v{version}":
        raise ValueError(f"Release tag {ref} does not match project version {version}")
    if not entries["Install.command"][1] & stat.S_IXUSR:
        raise ValueError("Installer command must be executable in Git")
    runtime = bundle_files(bundle, version, commit)
    manifest = {
        "format": 2,
        "kind": "app-bundle",
        "version": version,
        "commit": commit,
        "dirty": False,
    }
    text_files = {
        "Install.command": entries["Install.command"],
        "README.md": entries["README.md"],
        "RELEASE.json": ((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(), 0o644),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"People-Mosaic-{version}-macOS-arm64.zip"
    prefix = f"People-Mosaic-{version}/"
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, (contents, mode) in sorted(text_files.items()):
            archive.writestr(zip_info(prefix + name, stat.S_IFREG | mode), contents)
        for path, mode in runtime:
            name = prefix + "People Mosaic.app"
            if path != bundle:
                name += "/" + path.relative_to(bundle).as_posix()
            if stat.S_ISDIR(mode):
                archive.writestr(zip_info(name + "/", mode), b"")
            elif stat.S_ISLNK(mode):
                archive.writestr(zip_info(name, mode), str(path.readlink()).encode())
            else:
                with path.open("rb") as source, archive.open(zip_info(name, mode), "w") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
    checksum = output.with_suffix(".zip.sha256")
    with output.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    checksum.write_text(f"{digest}  {output.name}\n")
    return output, checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD", help="Committed Git revision (e.g. v0.1.0)")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "build/bundle/People Mosaic.app")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/releases")
    args = parser.parse_args()
    try:
        revision = resolve_revision(ROOT, args.ref)
        outputs = package_files(
            git_files(ROOT, revision),
            args.output_dir,
            args.ref,
            commit=revision,
            bundle=args.bundle_dir,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        tomllib.TOMLDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"Release packaging failed: {error}\n")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
