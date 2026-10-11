"""Release installation prepares at final paths and preserves existing launchers."""

import importlib.util
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load_script("install_release")
builder = load_script("build_macos")


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "download a space & quote'"
    root.mkdir()
    app = root / "People Mosaic.app"
    (app / "Contents/MacOS").mkdir(parents=True)
    (app / "Contents/Resources/models").mkdir(parents=True)
    (app / "Contents/Resources/models/verified.onnx").write_bytes(b"bundled verified model")
    (app / "Contents/Resources/model-alias").symlink_to("models", target_is_directory=True)
    executable = app / "Contents/MacOS/PeopleMosaic"
    executable.write_bytes(b"fixture standalone runtime")
    executable.chmod(0o755)
    (app / "Contents/Info.plist").write_bytes(
        plistlib.dumps(
            {"CFBundleIdentifier": "local.people-mosaic.app", "CFBundleShortVersionString": "0.1.0"}
        )
    )
    (root / "README.md").write_text("installation instructions\n")
    set_commit(root, "a")
    return root


def set_commit(package, character):
    release = {
        "format": 2,
        "kind": "app-bundle",
        "version": "0.1.0",
        "commit": character * 40,
    }
    (package / "RELEASE.json").write_text(json.dumps(release))
    (package / "People Mosaic.app/Contents/Resources/RELEASE.json").write_text(json.dumps(release))


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home with spaces"


@pytest.fixture
def prepare():
    calls = []

    def run(path):
        calls.append(path)
        # Test the actual app/workflow generator against the immutable version root.
        assert (path / "build/People Mosaic.app/Contents/MacOS/PeopleMosaic").is_file()
        builder.build(path, bundled=True)

    run.calls = calls
    return run


def paths(home):
    base = home / "Library/Application Support/People Mosaic"
    return (
        base,
        base / "current",
        home / "Applications/People Mosaic.app",
        home / "Library/Services/People Mosaic.workflow",
    )


def workflow_command(path):
    data = plistlib.loads((path / "Contents/document.wflow").read_bytes())
    return data["actions"][0]["action"]["ActionParameters"]["COMMAND_STRING"]


def test_first_install_builds_at_permanent_version_path(package, home, prepare):
    app = installer.install(package, home, prepare)
    base, current, expected_app, workflow = paths(home)
    target = base / "versions/0.1.0-aaaaaaaaaaaa"
    assert app == expected_app
    assert prepare.calls == [target]
    assert current.is_symlink() and current.resolve() == target
    assert app.is_symlink() and app.resolve() == target / "build/People Mosaic.app"
    assert installer.owned(base) and installer.owned(target)
    installed_app = target / "build/People Mosaic.app"
    assert (installed_app / "Contents/MacOS/PeopleMosaic").read_bytes() == (
        package / "People Mosaic.app/Contents/MacOS/PeopleMosaic"
    ).read_bytes()
    assert not (target / ".venv").exists()
    assert not (target / "src").exists()
    assert (installed_app / "Contents/Resources/model-alias").is_symlink()
    command = workflow_command(workflow)
    assert str(package) not in command
    # Quoting must produce executable shell syntax for a path containing spaces.
    subprocess.run(["zsh", "-n"], input=command, text=True, check=True)
    assert installer.owned_workflow(workflow)


def test_update_copies_bundled_models_and_reinstall_keeps_same_version_root(package, home, prepare):
    installer.install(package, home, prepare)
    base, current, app, workflow = paths(home)
    previous = current.resolve()
    model = "build/People Mosaic.app/Contents/Resources/models/verified.onnx"
    (previous / model).write_bytes(b"old model")
    set_commit(package, "b")
    installer.install(package, home, prepare)
    updated = current.resolve()
    assert updated != previous and previous.exists()
    assert (updated / model).read_bytes() == b"bundled verified model"
    assert prepare.calls == [previous, updated]
    assert str(previous) not in workflow_command(workflow)
    assert installer.owned_workflow(workflow)
    installer.install(package, home, prepare)
    assert current.resolve() == updated
    assert app.resolve() == updated / "build/People Mosaic.app"
    assert set((base / "versions").iterdir()) == {previous, updated}
    assert prepare.calls == [previous, updated, updated]


@pytest.mark.parametrize("existing", [False, True])
def test_prepare_failure_does_not_publish_or_break_previous_install(
    package, home, prepare, existing
):
    base, current, app, workflow = paths(home)
    if existing:
        installer.install(package, home, prepare)
        previous = current.resolve()
        command = workflow_command(workflow)
        set_commit(package, "b")

    def fail(path):
        raise subprocess.CalledProcessError(1, ["mock uv"])

    with pytest.raises(subprocess.CalledProcessError):
        installer.install(package, home, fail)
    if existing:
        assert current.resolve() == previous
        assert app.resolve() == previous / "build/People Mosaic.app"
        assert workflow_command(workflow) == command
        assert set((base / "versions").iterdir()) == {previous}
    else:
        assert not installer.exists(current)
        assert not installer.exists(app)
        assert not installer.exists(workflow)
        assert not list((base / "versions").iterdir())


def test_publication_failure_restores_previous_app_workflow_and_current(
    package, home, prepare, monkeypatch
):
    installer.install(package, home, prepare)
    _, current, app, workflow = paths(home)
    previous = current.resolve()
    old_command = workflow_command(workflow)
    set_commit(package, "b")
    real_replace = os.replace

    def fail_app_publish(source, destination):
        if Path(destination) == app and Path(source).name.startswith(".People Mosaic."):
            raise OSError("mock publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(installer.os, "replace", fail_app_publish)
    with pytest.raises(OSError, match="mock publication failure"):
        installer.install(package, home, prepare)
    assert current.resolve() == previous
    assert app.resolve() == previous / "build/People Mosaic.app"
    assert workflow_command(workflow) == old_command
    assert not list(workflow.parent.glob(".People Mosaic.*"))
    assert not list(app.parent.glob(".People Mosaic.*"))


@pytest.mark.parametrize("foreign", ["base", "workflow", "app", "version", "base_symlink"])
def test_foreign_destinations_are_preserved_before_preparation(package, home, prepare, foreign):
    base, _, app, workflow = paths(home)
    if foreign == "base_symlink":
        other = home / "unrelated"
        other.mkdir(parents=True)
        installer.write_marker(other, {"format": 1})
        base.parent.mkdir(parents=True)
        base.symlink_to(other, target_is_directory=True)
        occupied = other
    elif foreign == "base":
        occupied = base
    elif foreign == "version":
        base.mkdir(parents=True)
        installer.write_marker(base, {"format": 1})
        occupied = base / "versions/0.1.0-aaaaaaaaaaaa"
    else:
        occupied = app if foreign == "app" else workflow
    occupied.mkdir(parents=True, exist_ok=True)
    sentinel = occupied / "do not remove.txt"
    sentinel.write_text("unrelated data")
    with pytest.raises(ValueError):
        installer.install(package, home, prepare)
    assert prepare.calls == []
    assert sentinel.read_text() == "unrelated data"
    if foreign in {"workflow", "app"}:
        assert not base.exists()


def test_migrates_existing_owned_repository_workflow_and_app_with_backup(package, home, prepare):
    old_root = home / "old repo ' & spaces"
    old_root.mkdir(parents=True)
    builder.build(old_root)
    base, current, app, workflow = paths(home)
    app.parent.mkdir(parents=True)
    workflow.parent.mkdir(parents=True)
    shutil.copytree(old_root / "build/People Mosaic.app", app)
    shutil.copytree(old_root / "build/People Mosaic.workflow", workflow)
    old_app = (app / "Contents/MacOS/PeopleMosaic").read_text()
    old_command = workflow_command(workflow)
    installer.install(package, home, prepare)
    backup = next((base / "backups").iterdir())
    assert workflow_command(backup / workflow.name) == old_command
    assert (backup / app.name / "Contents/MacOS/PeopleMosaic").read_text() == old_app
    assert installer.owned_workflow(workflow)
    assert str(old_root) not in workflow_command(workflow)
    assert app.resolve() == current.resolve() / "build/People Mosaic.app"
    assert old_root.exists()


def test_second_installer_is_refused_before_touching_runtime(package, home, prepare):
    attempts = []

    def competing_prepare(path):
        with pytest.raises(ValueError, match="別のインストールが実行中"):
            installer.install(package, home, lambda _: attempts.append("raced"))
        prepare(path)

    installer.install(package, home, competing_prepare)
    assert attempts == []
    assert len(prepare.calls) == 1
    # Lock lifetime ends even on completion, so subsequent installs work.
    installer.install(package, home, prepare)
    assert len(prepare.calls) == 2


def test_malformed_foreign_bundle_is_not_recognized_as_owned(tmp_path):
    bundle = tmp_path / "foreign.app"
    (bundle / "Contents").mkdir(parents=True)
    (bundle / "Contents/Info.plist").write_bytes(plistlib.dumps(["not a dictionary"]))
    assert installer.bundle_id(bundle) is None
    assert not installer.owned_workflow(bundle)


def test_prepare_only_runs_copied_runtime_self_test_then_builds_workflow(
    package, home, monkeypatch
):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))

    monkeypatch.setattr(installer.subprocess, "run", run)
    monkeypatch.setitem(sys.modules, "build_macos", builder)
    installer.install(package, home)
    _, current, _, workflow = paths(home)
    assert calls == [
        (
            [
                str(current.resolve() / "build/People Mosaic.app/Contents/MacOS/PeopleMosaic"),
                "--self-test",
            ],
            {"cwd": current.resolve(), "check": True},
        )
    ]
    assert installer.owned_workflow(workflow)
    assert ".venv" not in workflow_command(workflow)
    assert "people_mosaic.app" not in workflow_command(workflow)


def test_updates_format_one_install_without_copying_its_environment_or_models(
    package, home, prepare
):
    base, current, app, workflow = paths(home)
    old = base / "versions/0.1.0-cccccccccccc"
    old.mkdir(parents=True)
    installer.write_marker(base, {"format": 1})
    installer.write_marker(old, {"format": 1, "version": "0.1.0", "commit": "c" * 40})
    (old / ".venv").mkdir()
    (old / "models").mkdir()
    (old / "models/old.onnx").write_bytes(b"old downloaded model")
    builder.build(old)
    current.symlink_to(old, target_is_directory=True)
    app.parent.mkdir(parents=True)
    app.symlink_to(current / "build/People Mosaic.app", target_is_directory=True)
    workflow.parent.mkdir(parents=True)
    shutil.copytree(old / "build/People Mosaic.workflow", workflow)
    installer.install(package, home, prepare)
    assert current.resolve() != old and old.exists()
    assert not (current / ".venv").exists()
    assert not (current / "models").exists()
    assert installer.owned_workflow(workflow)


@pytest.mark.parametrize("bad", ["legacy", "kind", "foreign_app", "provenance", "dirty_bundle"])
def test_rejects_non_bundled_or_invalid_package_before_any_install_mutation(
    package, home, prepare, bad
):
    if bad == "foreign_app":
        plist = package / "People Mosaic.app/Contents/Info.plist"
        plist.write_bytes(plistlib.dumps({"CFBundleIdentifier": "unrelated.app"}))
    elif bad in {"provenance", "dirty_bundle"}:
        manifest = package / "People Mosaic.app/Contents/Resources/RELEASE.json"
        release = json.loads(manifest.read_text())
        release["commit" if bad == "provenance" else "dirty"] = (
            "b" * 40 if bad == "provenance" else True
        )
        manifest.write_text(json.dumps(release))
    else:
        release = json.loads((package / "RELEASE.json").read_text())
        release["format" if bad == "legacy" else "kind"] = 1 if bad == "legacy" else "source"
        (package / "RELEASE.json").write_text(json.dumps(release))
    with pytest.raises(ValueError):
        installer.install(package, home, prepare)
    assert prepare.calls == []
    assert not home.exists()
