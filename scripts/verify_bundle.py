"""Verify the released ZIP installs and processes video without external tools.

All installation paths are under a temporary home. The downloaded package is
removed before the installed application is tested; the actual user's Finder
services and Applications directory are never touched.
"""

import argparse
import json
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from package_release import bundle_files

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


def safe_zip_root(path: Path) -> str:
    """Reject path traversal and multiple roots before passing ZIP to ditto."""
    roots = set()
    names = set()
    with ZipFile(path) as archive:
        for item in archive.infolist():
            name = item.filename.rstrip("/")
            parts = PurePosixPath(name).parts
            if (
                not parts
                or name.startswith("/")
                or "\\" in name
                or "\x00" in name
                or any(part in {".", ".."} for part in parts)
                or PurePosixPath(name).as_posix() != name
                or name in names
            ):
                raise ValueError(f"Unsafe ZIP entry: {item.filename}")
            roots.add(parts[0])
            names.add(name)
            mode = item.external_attr >> 16
            if stat.S_IFMT(mode) not in {stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK}:
                raise ValueError("Release ZIP contains an unsupported file type")
        if len(roots) != 1:
            raise ValueError("Release ZIP must contain exactly one package folder")
        root = next(iter(roots))
        if not re.fullmatch(r"People-Mosaic-[0-9]+\.[0-9]+\.[0-9]+", root):
            raise ValueError("Unexpected release package folder")
        required = {f"{root}/{name}" for name in ("Install.command", "README.md", "RELEASE.json")}
        if not required <= names:
            raise ValueError("Release ZIP is missing installer metadata")
        # Symlinks must stay within the app after lexical resolution. Framework
        # links are supported, including directory links preserved by ditto.
        for item in archive.infolist():
            if (item.external_attr >> 16) & 0o170000 != 0o120000:
                continue
            target = archive.read(item).decode("utf-8")
            if target.startswith("/") or "\\" in target:
                raise ValueError("Release ZIP contains an unsafe symlink")
            segments = list(PurePosixPath(item.filename).parent.parts)
            for segment in target.split("/"):
                if segment in {"", "."}:
                    continue
                if segment == "..":
                    if not segments:
                        raise ValueError("Release ZIP symlink escapes package")
                    segments.pop()
                else:
                    segments.append(segment)
            if segments[:2] != [root, "People Mosaic.app"]:
                raise ValueError("Release ZIP symlink leaves app bundle")
    return root


def offline_environment(home: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "DYLD_", "QT_", "QML_"))
        and key not in {"VIRTUAL_ENV", "CONDA_PREFIX", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    }
    env.update({"PATH": SYSTEM_PATH, "HOME": str(home), "QT_QPA_PLATFORM": "offscreen"})
    return env


def run_offline(command: list[str], env: dict[str, str], *, timeout: int = 180) -> str:
    sandbox = Path("/usr/bin/sandbox-exec")
    if sandbox.is_file():
        command = [str(sandbox), "-p", "(version 1)(allow default)(deny network*)", *command]
    result = subprocess.run(
        command, check=True, env=env, capture_output=True, text=True, timeout=timeout
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    return result.stdout


def expected_manifest() -> dict:
    commit = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    project = subprocess.run(
        ["/usr/bin/git", "show", f"{commit}:pyproject.toml"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    version = tomllib.loads(project)["project"]["version"]
    return {"format": 2, "kind": "app-bundle", "version": version, "commit": commit, "dirty": False}


def verify(zip_path: Path) -> None:
    if sys.platform != "darwin":
        raise ValueError("Bundle verification requires Apple Silicon macOS")
    manifest = expected_manifest()
    folder = safe_zip_root(zip_path)
    with tempfile.TemporaryDirectory(prefix="people-mosaic-offline-") as directory:
        root = Path(directory).resolve()
        extracted = root / "Downloaded package with spaces 日本語"
        home = root / "Temporary home with spaces 日本語"
        home.mkdir()
        subprocess.run(
            ["/usr/bin/ditto", "-x", "-k", str(zip_path.resolve()), str(extracted)], check=True
        )
        package = extracted / folder
        if json.loads((package / "RELEASE.json").read_text()) != manifest:
            raise ValueError("Release ZIP does not match the current committed version")
        bundled = package / "People Mosaic.app"
        bundle_files(bundled, manifest["version"], manifest["commit"])
        executable = bundled / "Contents/MacOS/PeopleMosaic"
        env = offline_environment(home)
        run_offline([str(executable), "--install", str(package), "--home", str(home)], env)
        installed = home / "Applications/People Mosaic.app"
        installed_root = installed.resolve()
        installed_root.relative_to(home)
        if not installed.is_dir():
            raise ValueError("Installer did not create an application")
        subprocess.run(
            ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(installed)],
            check=True,
            env=env,
        )
        app_manifest = json.loads((installed / "Contents/Resources/RELEASE.json").read_text())
        if app_manifest != manifest:
            raise ValueError("Installed app provenance does not match ZIP")
        workflow = home / "Library/Services/People Mosaic.workflow/Contents/document.wflow"
        document = plistlib.loads(workflow.read_bytes())
        metadata = document["workflowMetaData"]
        if (
            metadata.get("presentationMode") != 15
            or metadata.get("inputTypeIdentifier") != "com.apple.Automator.fileSystemObject.movie"
            or metadata.get("applicationBundleID") != "com.apple.finder"
        ):
            raise ValueError("Installed workflow is missing Finder Quick Action metadata")
        action = document["actions"][0]["action"]
        command = action["ActionParameters"]["COMMAND_STRING"]
        if str(installed_root / "Contents/MacOS/PeopleMosaic") not in command:
            # Shell quoting inserts literal quote escapes for certain paths; the
            # deliberately chosen space/Japanese paths need ordinary quotes only.
            raise ValueError("Finder workflow does not point to the installed app")
        if (
            str(extracted) in command
            or "python" in command.lower()
            or "homebrew" in command.lower()
        ):
            raise ValueError("Finder workflow depends on downloaded files or developer tools")
        shutil.rmtree(extracted)
        entry = str(installed / "Contents/MacOS/PeopleMosaic")
        self_test = run_offline([entry, "--self-test"], env)
        report = json.loads(self_test.strip().splitlines()[-1])
        if report.get("frozen") is not True:
            raise ValueError("Installed executable did not use its bundled Python")
        if report.get("finder_service_update_api") != "NSUpdateDynamicServices":
            raise ValueError("Installed app cannot load the native Finder service refresh API")
        Path(report["resources"]).resolve().relative_to(installed_root)
        smoke_test = run_offline(
            [entry, "--smoke-test", str(root / "Processed outputs")], env, timeout=300
        )
        smoke = json.loads(smoke_test.strip().splitlines()[-1])
        if smoke.get("analysis_frames") != 12 or len(smoke.get("outputs", [])) != 3:
            raise ValueError("Installed app did not pass video analysis/export smoke checks")
        for output in smoke["outputs"]:
            path = Path(output)
            path.resolve().relative_to(root / "Processed outputs")
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Smoke test output is missing")
        print(
            json.dumps(
                {
                    "verified": True,
                    "network_denied": Path("/usr/bin/sandbox-exec").is_file(),
                    "commit": manifest["commit"],
                    "version": manifest["version"],
                }
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.zip)
    except (
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        if isinstance(error, subprocess.CalledProcessError):
            if error.stdout:
                print(error.stdout, file=sys.stderr)
            if error.stderr:
                print(error.stderr, file=sys.stderr)
        parser.exit(1, f"Offline bundle verification failed: {error}\n")


if __name__ == "__main__":
    main()
