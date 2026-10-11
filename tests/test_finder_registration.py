"""CI-only native discovery checks never modify local Services or preferences."""

import importlib.util
import plistlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("verify_bundle_ci", SCRIPTS / "verify_bundle.py")
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
sys.path.remove(str(SCRIPTS))


@pytest.fixture
def runner(tmp_path, monkeypatch):
    home = tmp_path / "runner"
    source = tmp_path / "installed/People Mosaic.workflow"
    (source / "Contents").mkdir(parents=True)
    (source / "Contents/Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": "local.people-mosaic.finder",
                "NSServices": [{"NSMenuItem": {"default": "People Mosaic"}}],
            }
        )
    )
    token = "123456789abcdef"
    workflow = home / f"Library/Services/People Mosaic CI {token}.workflow"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(verifier.sys, "platform", "darwin")
    monkeypatch.setattr(verifier.Path, "home", lambda: home)
    monkeypatch.setattr(verifier, "uuid4", lambda: SimpleNamespace(hex=token))
    return home, source, workflow, token


def service_dump(path, menu):
    return f'({{ NSBundlePath = "{path}"; NSMenuItem = {{ default = "{menu}"; }}; }})'


@pytest.mark.parametrize("ci,platform", [("false", "darwin"), ("true", "linux")])
def test_ci_guard_prevents_all_filesystem_or_native_changes(tmp_path, monkeypatch, ci, platform):
    monkeypatch.setenv("GITHUB_ACTIONS", ci)
    monkeypatch.setattr(verifier.sys, "platform", platform)
    monkeypatch.setattr(verifier.Path, "home", lambda: pytest.fail("real HOME accessed"))
    with pytest.raises(ValueError, match="restricted"):
        verifier.finder_registration(tmp_path / "executable", tmp_path / "workflow")


@pytest.mark.parametrize("broken_symlink", [False, True])
def test_existing_workflow_is_never_replaced(runner, monkeypatch, broken_symlink):
    _, source, workflow, _ = runner
    workflow.parent.mkdir(parents=True)
    if broken_symlink:
        workflow.symlink_to(workflow.parent / "missing")
    else:
        workflow.mkdir()
        (workflow / "unrelated").write_text("preserve")
    monkeypatch.setattr(verifier, "run_offline", lambda *a, **k: pytest.fail("native call"))
    with pytest.raises(ValueError, match="Refusing to replace"):
        verifier.finder_registration(Path("exe"), source)
    assert (
        workflow.is_symlink()
        if broken_symlink
        else (workflow / "unrelated").read_text() == "preserve"
    )


def test_registers_unique_workflow_with_runner_home_and_cleans_only_it(runner, monkeypatch, capsys):
    home, source, workflow, token = runner
    unrelated = workflow.parent / "Other.workflow"
    unrelated.mkdir(parents=True)
    calls = []

    def native(command, env, **kwargs):
        assert env["HOME"] == str(home)
        assert command == ["exe", "--refresh-services"]
        calls.append("register" if workflow.exists() else "cleanup")

    def dump(command, **kwargs):
        assert command == ["/System/Library/CoreServices/pbs", "-dump"]
        info = plistlib.loads((workflow / "Contents/Info.plist").read_bytes())
        assert info["CFBundleIdentifier"] == f"local.people-mosaic.ci.{token}"
        return SimpleNamespace(stdout=service_dump(workflow, info["CFBundleName"]), stderr="")

    monkeypatch.setattr(verifier, "run_offline", native)
    monkeypatch.setattr(verifier.subprocess, "run", dump)
    verifier.finder_registration(Path("exe"), source)
    assert calls == ["register", "cleanup"]
    assert not workflow.exists() and unrelated.is_dir()
    assert '"finder_service_registered": true' in capsys.readouterr().out
    assert (
        plistlib.loads((source / "Contents/Info.plist").read_bytes())["CFBundleIdentifier"]
        == "local.people-mosaic.finder"
    )


def test_registration_timeout_cleans_workflow_and_never_reports_success(
    runner, monkeypatch, capsys
):
    _, source, workflow, _ = runner
    clock = [0.0]
    calls = []
    monkeypatch.setattr(verifier.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        verifier.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(verifier, "run_offline", lambda *a, **k: calls.append("refresh"))
    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="XPC message error: Connection invalid", stderr=""),
    )
    with pytest.raises(ValueError, match="did not register"):
        verifier.finder_registration(Path("exe"), source)
    assert clock[0] == 15 and calls == ["refresh", "refresh"]
    assert not workflow.exists()
    assert '"finder_service_registered": true' not in capsys.readouterr().out


def test_native_failure_still_removes_owned_workflow_and_refreshes_cleanup(runner, monkeypatch):
    _, source, workflow, _ = runner
    calls = []

    def native(*args, **kwargs):
        calls.append(workflow.exists())
        if len(calls) == 1:
            raise OSError("native discovery failure")

    monkeypatch.setattr(verifier, "run_offline", native)
    with pytest.raises(OSError, match="native discovery failure"):
        verifier.finder_registration(Path("exe"), source)
    assert calls == [True, False] and not workflow.exists()


def test_cleanup_api_failure_never_reports_a_pass(runner, monkeypatch, capsys):
    _, source, workflow, token = runner
    calls = []

    def native(*args, **kwargs):
        calls.append(workflow.exists())
        if len(calls) == 2:
            raise OSError("cleanup refresh failure")

    monkeypatch.setattr(verifier, "run_offline", native)
    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=service_dump(workflow, f"People Mosaic CI {token}"), stderr=""
        ),
    )
    with pytest.raises(OSError, match="cleanup refresh failure"):
        verifier.finder_registration(Path("exe"), source)
    assert calls == [True, False] and not workflow.exists()
    assert '"finder_service_registered": true' not in capsys.readouterr().out


def test_cleanup_refuses_foreign_replacement(runner, monkeypatch):
    _, source, workflow, _ = runner
    calls = []

    def native(*args, **kwargs):
        calls.append("refresh")
        (workflow / ".people-mosaic-ci-owner").write_text("foreign replacement")
        raise OSError("failed while ownership changed")

    monkeypatch.setattr(verifier, "run_offline", native)
    with pytest.raises(ValueError, match="Refusing to remove an unowned"):
        verifier.finder_registration(Path("exe"), source)
    assert workflow.is_dir() and calls == ["refresh"]
    assert (workflow / ".people-mosaic-ci-owner").read_text() == "foreign replacement"


def test_service_matching_requires_same_path_and_menu():
    wanted = Path("/Users/runner/Library/Services/Wanted.workflow")
    wrong = Path("/Users/runner/Library/Services/Other.workflow")
    dump = service_dump(wanted, "Other menu") + service_dump(wrong, "Wanted menu")
    assert not verifier.service_registered(dump, wanted, "Wanted menu")
    assert verifier.service_registered(service_dump(wanted, "Wanted menu"), wanted, "Wanted menu")
