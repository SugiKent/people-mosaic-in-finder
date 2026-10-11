"""Install an extracted release at a permanent path, keeping previous versions."""

import fcntl
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

OWNER = "local.people-mosaic.release"
MARKER = ".people-mosaic-install.json"


def owned(path: Path) -> bool:
    try:
        return json.loads((path / MARKER).read_text())["owner"] == OWNER
    except (OSError, ValueError, KeyError, TypeError):
        return False


def bundle_id(path: Path) -> str | None:
    try:
        info = plistlib.loads((path / "Contents/Info.plist").read_bytes())
        return info.get("CFBundleIdentifier") if isinstance(info, dict) else None
    except (OSError, ValueError, plistlib.InvalidFileException):
        return None


def owned_workflow(path: Path) -> bool:
    if path.is_symlink() or bundle_id(path) != "local.people-mosaic.finder":
        return False
    try:
        workflow = plistlib.loads((path / "Contents/document.wflow").read_bytes())
        info = plistlib.loads((path / "Contents/Info.plist").read_bytes())
        actions = workflow["actions"]
        if len(actions) != 1:
            return False
        action = actions[0]["action"]
        command = action["ActionParameters"]["COMMAND_STRING"]
        return action["BundleIdentifier"] == "com.apple.RunShellScript" and (
            "-m people_mosaic.app" in command
            or (info.get("MosaicOwner") == OWNER and "/Contents/MacOS/PeopleMosaic" in command)
        )
    except (OSError, ValueError, KeyError, TypeError, plistlib.InvalidFileException):
        return False


def exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def matches_release(app: Path, release: dict) -> bool:
    try:
        manifest = json.loads((app / "Contents/Resources/RELEASE.json").read_text())
        info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
        return (
            isinstance(manifest, dict)
            and manifest.get("dirty") is not True
            and all(
                manifest.get(key) == release[key] for key in ("format", "kind", "version", "commit")
            )
            and info.get("CFBundleShortVersionString") == release["version"]
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def write_marker(path: Path, release: dict):
    (path / MARKER).write_text(json.dumps({"owner": OWNER, **release}) + "\n")


def prepare(path: Path):
    # Validate the copied runtime entirely offline, before changing any launcher.
    executable = path / "build/People Mosaic.app/Contents/MacOS/PeopleMosaic"
    subprocess.run(
        [str(executable), "--self-test"],
        cwd=path,
        check=True,
    )
    from build_macos import build

    build(path, bundled=True)


def install(package: Path, home: Path, prepare_install=prepare) -> Path:
    release = json.loads((package / "RELEASE.json").read_text())
    version = release.get("version") if isinstance(release, dict) else None
    if (
        not isinstance(release, dict)
        or release.get("format") != 2
        or release.get("kind") != "app-bundle"
        or not isinstance(release.get("commit"), str)
        or not re.fullmatch(r"[a-f0-9]{40}", release.get("commit", ""))
        or not isinstance(version, str)
        or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version)
    ):
        raise ValueError("Release 情報を確認できません。ZIP を再ダウンロードしてください。")
    packaged_app = package / "People Mosaic.app"
    if (
        packaged_app.is_symlink()
        or bundle_id(packaged_app) != "local.people-mosaic.app"
        or not matches_release(packaged_app, release)
    ):
        raise ValueError("同梱のアプリを確認できません。ZIP を再ダウンロードしてください。")
    base = home / "Library/Application Support/People Mosaic"
    versions = base / "versions"
    current = base / "current"
    target = versions / f"{version}-{release['commit'][:12]}"
    workflow = home / "Library/Services/People Mosaic.workflow"
    app = home / "Applications/People Mosaic.app"
    # Validate all destinations before preparing or changing an installed application.
    if exists(base) and (base.is_symlink() or not owned(base)):
        raise ValueError(f"別のフォルダを上書きしません: {base}")
    if exists(versions) and (versions.is_symlink() or not versions.is_dir()):
        raise ValueError(f"保存先を確認してください: {versions}")
    if exists(target) and (target.is_symlink() or not owned(target)):
        raise ValueError(f"別のフォルダを上書きしません: {target}")
    if exists(current) and (
        not current.is_symlink()
        or current.resolve().parent != versions.resolve()
        or not owned(current.resolve())
    ):
        raise ValueError(f"既存の起動先を確認してください: {current}")
    if exists(workflow) and not owned_workflow(workflow):
        raise ValueError(f"別の Finder アクションを上書きしません: {workflow}")
    if exists(app) and bundle_id(app) != "local.people-mosaic.app":
        raise ValueError(f"別のアプリを上書きしません: {app}")
    if not base.exists():
        base.mkdir(parents=True, mode=0o700)
        write_marker(base, {"format": 1})
    # A second double-click must not race runtime validation or publication.
    lock_fd = os.open(base / ".install.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "別のインストールが実行中です。完了してから再実行してください。"
            ) from error
        return install_at_paths(
            package, release, base, versions, current, target, workflow, app, prepare_install
        )


def install_at_paths(
    package, release, base, versions, current, target, workflow, app, prepare_install
):
    versions.mkdir(exist_ok=True)
    created = not target.exists()
    previous = current.resolve() if current.is_symlink() else None
    try:
        if created:
            target.mkdir(mode=0o700)
            write_marker(target, release)
            (target / "build").mkdir()
            shutil.copytree(
                package / "People Mosaic.app",
                target / "build/People Mosaic.app",
                symlinks=True,
            )
            for name in ("README.md", "RELEASE.json"):
                shutil.copy2(package / name, target / name)
        if not matches_release(target / "build/People Mosaic.app", release):
            raise ValueError(
                "アプリと Release 情報が一致しません。ZIP を再ダウンロードしてください。"
            )
        prepare_install(target)
        new_workflow = target / "build/People Mosaic.workflow"
        new_app = target / "build/People Mosaic.app"
        if (
            not owned_workflow(new_workflow)
            or bundle_id(new_app) != "local.people-mosaic.app"
            or not matches_release(new_app, release)
        ):
            raise ValueError("生成したアプリまたは Finder アクションが不正です。")
    except BaseException:
        if created:
            shutil.rmtree(target)
        raise

    workflow.parent.mkdir(parents=True, exist_ok=True)
    app.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    staged = workflow.with_name(f".People Mosaic.{token}.workflow")
    staged_app = app.with_name(f".People Mosaic.{token}.app")
    staged_current = base / f".current-{token}"
    backup = base / "backups" / token
    backup.mkdir(parents=True)
    moved_workflow = moved_app = switched = published_workflow = published_app = False
    try:
        shutil.copytree(new_workflow, staged)
        staged_current.symlink_to(target, target_is_directory=True)
        staged_app.symlink_to(current / "build/People Mosaic.app", target_is_directory=True)
        if exists(workflow):
            os.replace(workflow, backup / workflow.name)
            moved_workflow = True
        if exists(app):
            os.replace(app, backup / app.name)
            moved_app = True
        os.replace(staged_current, current)
        switched = True
        os.replace(staged, workflow)
        published_workflow = True
        os.replace(staged_app, app)
        published_app = True
    except BaseException:
        if published_app:
            app.unlink()
        if published_workflow:
            shutil.rmtree(workflow)
        if switched:
            current.unlink()
            if previous:
                current.symlink_to(previous, target_is_directory=True)
        if moved_app:
            os.replace(backup / app.name, app)
        if moved_workflow:
            os.replace(backup / workflow.name, workflow)
        raise
    finally:
        if staged.exists():
            shutil.rmtree(staged)
        staged_app.unlink(missing_ok=True)
        staged_current.unlink(missing_ok=True)
    return app


def main():
    if sys.platform != "darwin":
        raise SystemExit("Apple Silicon の macOS 向けのインストーラです。")
    try:
        package = Path(__file__).resolve().parents[1]
        app = install(package, Path.home())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"インストールを完了できませんでした: {error}") from error
    subprocess.run(["/System/Library/CoreServices/pbs", "-flush"], check=False)
    print(f"インストール完了: {app}")
    print("展開した ZIP フォルダは削除できます。更新も新しい ZIP の Install.command から行えます。")
    subprocess.run(["/usr/bin/open", "-R", str(app)], check=False)


if __name__ == "__main__":
    main()
