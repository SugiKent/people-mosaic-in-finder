"""Build an offline Apple Silicon application with Python, models, and FFmpeg.

Run on arm64 macOS using the locked developer environment. The selected FFmpeg
and ffprobe are bundled together with their complete non-system library closure.
Production releases should select the project's minimal LGPL FFmpeg build.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PREFIXES = ("/usr/lib/", "/System/Library/")
ALLOWED_QT_FRAMEWORKS = {
    "QtCore.framework",
    "QtGui.framework",
    "QtWidgets.framework",
    "QtNetwork.framework",
    "QtDBus.framework",
    "QtOpenGL.framework",
    "QtOpenGLWidgets.framework",
    "QtPrintSupport.framework",
    "QtSvg.framework",
    "QtSvgWidgets.framework",
}
MACHO_MAGICS = {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}


def dependencies(output: str) -> list[str]:
    """Parse otool -L without interpreting paths containing spaces as arguments."""
    return [
        line.strip().split(" (compatibility version", 1)[0]
        for line in output.splitlines()[1:]
        if " (compatibility version" in line
    ]


def dylib_dependencies(path: Path) -> list[str]:
    output = subprocess.run(
        ["/usr/bin/otool", "-L", str(path)], check=True, capture_output=True, text=True
    ).stdout
    return dependencies(output)


@lru_cache(maxsize=None)
def runtime_paths(path: Path) -> list[str]:
    output = subprocess.run(
        ["/usr/bin/otool", "-l", str(path)], check=True, capture_output=True, text=True
    ).stdout
    return re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.*?) \(offset \d+\)", output)


def resolve_dependency(dependency: str, owner: Path, executable: Path) -> Path | None:
    owner = owner.resolve()
    executable = executable.resolve()
    if dependency.startswith(SYSTEM_PREFIXES):
        return None
    candidates = []
    if dependency.startswith("@loader_path/"):
        candidates.append(owner.parent / dependency.removeprefix("@loader_path/"))
    elif dependency.startswith("@executable_path/"):
        candidates.append(executable.parent / dependency.removeprefix("@executable_path/"))
    elif dependency.startswith("@rpath/"):
        suffix = dependency.removeprefix("@rpath/")
        for origin in dict.fromkeys((owner, executable)):
            for search in runtime_paths(origin):
                search = search.replace("@loader_path", str(origin.parent)).replace(
                    "@executable_path", str(executable.parent)
                )
                if search.startswith("/"):
                    candidates.append(Path(search) / suffix)
    elif dependency.startswith("/"):
        candidates.append(Path(dependency))
    else:
        raise ValueError(f"Unsupported dynamic library reference: {owner}: {dependency}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"Missing dynamic library dependency: {owner}: {dependency}")


def binary_closure(executables: list[Path]) -> list[tuple[str, str]]:
    """Collect full FFmpeg dependency closure before PyInstaller rewrites imports."""
    result = []
    libraries = {}
    pending = [(path.resolve(), path.resolve()) for path in executables]
    visited = set()
    for executable in executables:
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"FFmpeg tool must be executable: {executable}")
        result.append((str(executable.resolve()), "tools"))
    while pending:
        path, executable = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for dependency in dylib_dependencies(path):
            library = resolve_dependency(dependency, path, executable)
            if library is None or library == path:
                continue
            previous = libraries.get(library.name)
            if previous is not None and previous != library:
                raise ValueError(f"Conflicting library names: {previous} and {library}")
            if previous is None:
                libraries[library.name] = library
                result.append((str(library), "tools/lib"))
            pending.append((library, executable))
    return sorted(set(result))


def source_manifest(root: Path, allow_dirty: bool = False) -> dict:
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError("Project version must be a safe semantic version")
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "scripts",
            "pyproject.toml",
            "uv.lock",
            "licenses",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty and not allow_dirty:
        raise ValueError("Commit runtime/build/lock/license changes before building a release")
    return {
        "format": 2,
        "kind": "app-bundle",
        "version": version,
        "commit": None if dirty else revision,
        "dirty": bool(dirty),
    }


def model_data(root: Path) -> list[tuple[str, str]]:
    """Only two public model files with pinned digests may enter the app."""
    sys.path.insert(0, str(root / "src"))
    from people_mosaic.models import MODELS

    files = []
    for name, (_, expected) in MODELS.items():
        model = root / "models" / name
        if not model.is_file():
            raise ValueError(f"Required public model is missing: {name}")
        with model.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Public model hash mismatch: {name}")
        files.append((str(model), "models"))
    return files


def spec_text(
    root: Path, binaries: list[tuple[str, str]], datas: list[tuple[str, str]], version: str
) -> str:
    return f"""# Generated by scripts/build_bundle.py; no local media or directories are globbed.
a = Analysis([{str(root / "scripts/bundle_entry.py")!r}],
    pathex=[{str(root / "src")!r}, {str(root / "scripts")!r}], binaries={binaries!r}, datas={datas!r},
    hiddenimports=[], hookspath=[], runtime_hooks=[],
    excludes=['pytest', 'ruff', 'pip_audit', 'PyInstaller', 'tkinter',
        'PySide6.QtPdf', 'PySide6.QtPdfWidgets', 'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets', 'PySide6.QtMultimedia', 'PySide6.QtQml', 'PySide6.QtQuick',
        'PySide6.QtVirtualKeyboard'],
    noarchive=False)
# QtGui's imageformat hook also collects the PDF plugin and QtPdf. Neither is
# needed for this app; exclude PDF/WebEngine and unrelated optional modules.
def allowed_qt_entry(entry):
    name = entry[0].lower()
    if '/imageformats/' in name:
        allowed = ('libqgif.dylib', 'libqicns.dylib', 'libqico.dylib', 'libqjpeg.dylib',
            'libqsvg.dylib', 'libqtga.dylib', 'libqtiff.dylib', 'libqwbmp.dylib', 'libqwebp.dylib')
        if name.rsplit('/', 1)[-1] not in allowed:
            return False
    return not any(token in name for token in
        ('qtpdf', 'libqpdf', 'qtwebengine', 'qtmultimedia', 'qtqml', 'qtquick', 'qtvirtualkeyboard'))
a.binaries = [entry for entry in a.binaries if allowed_qt_entry(entry)]
a.datas = [entry for entry in a.datas if allowed_qt_entry(entry)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
    name='PeopleMosaic', debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False, argv_emulation=False,
    target_arch='arm64', codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='PeopleMosaic')
app = BUNDLE(coll, name='People Mosaic.app', icon=None,
    bundle_identifier='local.people-mosaic.app',
    info_plist={{'CFBundleShortVersionString': {version!r},
        'CFBundleVersion': {version!r}, 'LSMinimumSystemVersion': '15.0',
        'NSHighResolutionCapable': True}})
"""


def minimum_macos_versions(output: str) -> list[tuple[int, ...]]:
    """Read both modern and legacy Mach-O deployment-version load commands."""
    values = re.findall(r"\bminos ([0-9]+(?:\.[0-9]+)*)", output)
    values += re.findall(
        r"cmd LC_VERSION_MIN_MACOSX\n\s+cmdsize \d+\n\s+version ([0-9]+(?:\.[0-9]+)*)", output
    )
    return [tuple(int(part) for part in value.split(".")) for value in values]


def external_library_references(app: Path) -> list[str]:
    """Audit every bundled Mach-O file for surviving local-machine dependencies."""
    app = app.resolve()
    problems = []
    executable = app / "Contents/MacOS/PeopleMosaic"
    for path in app.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if path.name.startswith(("libx264", "libx265")):
            problems.append(f"{path.relative_to(app)}: GPL encoder is forbidden in the LGPL bundle")
        with path.open("rb") as source:
            if source.read(4) not in MACHO_MAGICS:
                continue
        architectures = subprocess.run(
            ["/usr/bin/lipo", "-archs", str(path)], check=True, capture_output=True, text=True
        ).stdout.split()
        if "arm64" not in architectures:
            problems.append(f"{path.relative_to(app)}: arm64 architecture is missing")
        commands = subprocess.run(
            ["/usr/bin/otool", "-l", str(path)], check=True, capture_output=True, text=True
        ).stdout
        install_ids = re.findall(
            r"cmd LC_ID_DYLIB\n\s+cmdsize \d+\n\s+name (.*?) \(offset \d+\)", commands
        )
        for dependency in dylib_dependencies(path):
            if dependency in install_ids:
                continue
            if dependency.startswith("/") and not dependency.startswith(SYSTEM_PREFIXES):
                problems.append(f"{path.relative_to(app)}: {dependency}")
            elif dependency.startswith("@"):
                try:
                    resolved = resolve_dependency(dependency, path, executable)
                    if resolved is not None and not resolved.is_relative_to(app.resolve()):
                        problems.append(
                            f"{path.relative_to(app)}: dependency escapes app: {dependency}"
                        )
                except ValueError as error:
                    problems.append(str(error))
        for rpath in re.findall(
            r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.*?) \(offset \d+\)", commands
        ):
            if rpath.startswith("/") and not rpath.startswith(SYSTEM_PREFIXES):
                problems.append(f"{path.relative_to(app)}: external library search path {rpath}")
        for component in path.parts:
            if (
                component.startswith("Qt")
                and component.endswith(".framework")
                and component not in ALLOWED_QT_FRAMEWORKS
            ):
                problems.append(
                    f"{path.relative_to(app)}: optional Qt module is outside the licensed bundle scope"
                )
        if any(
            token in path.name.lower()
            for token in ("qtpdf", "libqpdf", "qtwebengine", "qtvirtualkeyboard")
        ):
            problems.append(f"{path.relative_to(app)}: unused PDF/WebEngine module is forbidden")
        for version in minimum_macos_versions(commands):
            padded = (version + (0, 0))[:3]
            if padded > (15, 0, 0):
                problems.append(
                    f"{path.relative_to(app)}: requires macOS {'.'.join(map(str, version))}, target is 15.0"
                )
    return problems


def build(args: argparse.Namespace) -> Path:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError("Build requires native Apple Silicon macOS")
    manifest = source_manifest(ROOT, args.allow_dirty)
    work = ROOT / "build/pyinstaller"
    work.mkdir(parents=True, exist_ok=True)
    manifest_file = work / "RELEASE.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    data = model_data(ROOT) + [(str(manifest_file), ".")]
    license_root = args.licenses_dir
    if not license_root.is_dir() and not args.allow_dirty:
        raise ValueError("Collect third-party licenses before building a release")
    if license_root.is_dir():
        for path in sorted(license_root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                data.append(
                    (str(path), str(Path("licenses") / path.relative_to(license_root).parent))
                )
    binaries = binary_closure([args.ffmpeg, args.ffprobe])
    spec = work / "PeopleMosaic.spec"
    spec.write_text(spec_text(ROOT, binaries, data, manifest["version"]))
    env = os.environ.copy()
    env["MACOSX_DEPLOYMENT_TARGET"] = "15.0"
    env["PYINSTALLER_CONFIG_DIR"] = str(work / "cache")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(args.output_dir),
            "--workpath",
            str(work / "work"),
            str(spec),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    app = args.output_dir / "People Mosaic.app"
    problems = external_library_references(app)
    if problems:
        raise ValueError("Bundled library portability audit failed:\n" + "\n".join(problems))
    print(app)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/bundle")
    parser.add_argument("--licenses-dir", type=Path, default=ROOT / "build/bundle-licenses")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Debug build, cannot be published"
    )
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Bundle build failed: {error}\n")


if __name__ == "__main__":
    main()
