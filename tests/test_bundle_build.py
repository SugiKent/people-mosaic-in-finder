"""Bundled tools must carry every external dependency and public model only."""

import ast
import hashlib
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_bundle", Path(__file__).resolve().parents[1] / "scripts/build_bundle.py"
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def executable(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"executable fixture")
    path.chmod(0o755)
    return path


def test_otool_parser_preserves_spaces_and_excludes_header():
    output = """/source tools/ffmpeg:
\t/source libraries/libfirst.dylib (compatibility version 1.0.0, current version 1.1.0)
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1.0.0)
"""
    assert builder.dependencies(output) == [
        "/source libraries/libfirst.dylib",
        "/usr/lib/libSystem.B.dylib",
    ]


def test_binary_closure_collects_transitive_libraries_only_once(tmp_path, monkeypatch):
    ffmpeg = executable(tmp_path / "bin/ffmpeg")
    ffprobe = executable(tmp_path / "bin/ffprobe")
    first = executable(tmp_path / "lib/libfirst.dylib")
    second = executable(tmp_path / "lib/libsecond.dylib")
    links = {
        ffmpeg: [str(first), "/usr/lib/libSystem.B.dylib"],
        ffprobe: [str(first)],
        first: [str(first), str(second)],
        second: [str(second)],
    }
    monkeypatch.setattr(builder, "dylib_dependencies", lambda path: links[path])
    assert builder.binary_closure([ffmpeg, ffprobe]) == sorted(
        [
            (str(ffmpeg), "tools"),
            (str(ffprobe), "tools"),
            (str(first), "tools/lib"),
            (str(second), "tools/lib"),
        ]
    )


def test_missing_library_is_a_build_failure(tmp_path):
    with pytest.raises(ValueError, match="Missing dynamic library dependency"):
        builder.resolve_dependency(
            str(tmp_path / "missing.dylib"), tmp_path / "tool", tmp_path / "tool"
        )


def test_rpath_and_loader_path_resolution(tmp_path, monkeypatch):
    tool = executable(tmp_path / "bin/ffmpeg")
    library = executable(tmp_path / "lib/libcodec.dylib")
    monkeypatch.setattr(builder, "runtime_paths", lambda _: ["@loader_path/../lib"])
    assert builder.resolve_dependency("@rpath/libcodec.dylib", tool, tool) == library
    assert builder.resolve_dependency("@loader_path/../lib/libcodec.dylib", tool, tool) == library
    assert builder.resolve_dependency("/System/Library/Frameworks/CoreMedia", tool, tool) is None


def test_conflicting_library_names_fail(tmp_path, monkeypatch):
    tool = executable(tmp_path / "bin/ffmpeg")
    first = executable(tmp_path / "one/libcodec.dylib")
    second = executable(tmp_path / "two/libcodec.dylib")
    monkeypatch.setattr(
        builder,
        "dylib_dependencies",
        lambda path: [str(first), str(second)] if path == tool else [],
    )
    with pytest.raises(ValueError, match="Conflicting library names"):
        builder.binary_closure([tool])


def test_modified_runtime_cannot_claim_commit_provenance(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.1.0"\n')

    def run(command, **kwargs):
        output = "a" * 40 if "rev-parse" in command else " M src/people_mosaic/app.py\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(builder.subprocess, "run", run)
    with pytest.raises(ValueError, match="Commit runtime/build/lock/license changes"):
        builder.source_manifest(tmp_path)
    manifest = builder.source_manifest(tmp_path, allow_dirty=True)
    assert manifest == {
        "format": 2,
        "kind": "app-bundle",
        "version": "0.1.0",
        "commit": None,
        "dirty": True,
    }


def test_model_data_is_pinned_and_never_globs_local_media(tmp_path, monkeypatch):
    from people_mosaic import models

    model = tmp_path / "models/public.onnx"
    model.parent.mkdir()
    model.write_bytes(b"verified public model")
    (model.parent / "private-video.mp4").write_bytes(b"private user data")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    monkeypatch.setattr(models, "MODELS", {"public.onnx": ("public", expected)})
    assert builder.model_data(tmp_path) == [(str(model), "models")]
    model.write_bytes(b"tampered model")
    with pytest.raises(ValueError, match="Public model hash mismatch"):
        builder.model_data(tmp_path)


def test_macos_deployment_parser_checks_modern_and_legacy_load_commands():
    output = """
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 26.0
      sdk 26.4
Load command 5
      cmd LC_VERSION_MIN_MACOSX
  cmdsize 16
  version 11.0
      sdk 12.3
"""
    assert builder.minimum_macos_versions(output) == [(26, 0), (11, 0)]


def test_bundle_audit_rejects_newer_macos_and_external_machine_libraries(tmp_path, monkeypatch):
    binary = tmp_path / "Contents/MacOS/PeopleMosaic"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\xcf\xfa\xed\xfe" + b"fixture")
    monkeypatch.setattr(
        builder, "dylib_dependencies", lambda _: ["/opt/homebrew/lib/private.dylib"]
    )
    monkeypatch.setattr(
        builder.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, "arm64" if "-archs" in cmd else "cmd LC_BUILD_VERSION\nminos 26.0\n", ""
        ),
    )
    problems = builder.external_library_references(tmp_path)
    assert len(problems) == 2
    assert any("/opt/homebrew/lib/private.dylib" in item for item in problems)
    assert any("requires macOS 26.0" in item for item in problems)


def test_bundle_audit_rejects_missing_arm64_and_unresolved_rpath(tmp_path, monkeypatch):
    binary = tmp_path / "Contents/MacOS/PeopleMosaic"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\xcf\xfa\xed\xfe" + b"fixture")
    monkeypatch.setattr(builder, "dylib_dependencies", lambda _: ["@rpath/libmissing.dylib"])
    monkeypatch.setattr(builder, "runtime_paths", lambda _: ["@loader_path"])
    monkeypatch.setattr(
        builder.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, "x86_64" if "-archs" in cmd else "", ""
        ),
    )
    problems = builder.external_library_references(tmp_path)
    assert any("arm64 architecture is missing" in item for item in problems)
    assert any("Missing dynamic library dependency" in item for item in problems)


def test_transitive_rpath_uses_executable_search_paths(tmp_path, monkeypatch):
    tool = executable(tmp_path / "bin/ffmpeg")
    owner = executable(tmp_path / "lib/libowner.dylib")
    dependency = executable(tmp_path / "lib/libcodec.dylib")
    monkeypatch.setattr(
        builder, "runtime_paths", lambda path: ["@executable_path/../lib"] if path == tool else []
    )
    assert builder.resolve_dependency("@rpath/libcodec.dylib", owner, tool) == dependency


def test_relative_build_paths_resolve_rpath_identically(tmp_path, monkeypatch):
    tool = executable(tmp_path / "bin/ffmpeg")
    library = executable(tmp_path / "lib/libcodec.dylib")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builder, "runtime_paths", lambda _: ["@loader_path/../lib"])
    assert (
        builder.resolve_dependency("@rpath/libcodec.dylib", Path("bin/ffmpeg"), Path("bin/ffmpeg"))
        == library
    )
    assert builder.resolve_dependency("@rpath/libcodec.dylib", tool, tool) == library


def test_relative_license_cli_path_is_readable_from_generated_spec(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    notices = checkout / "build/bundle-licenses"
    notices.mkdir(parents=True)
    (notices / "FFMPEG-BUILD.json").write_bytes(b"exact build provenance")
    asset = checkout / "src/people_mosaic/assets/default-fluffy.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"default fluffy illustration")
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(builder, "ROOT", checkout)
    monkeypatch.setattr(builder.sys, "platform", "darwin")
    monkeypatch.setattr(builder.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(builder, "source_manifest", lambda *a: {"version": "0.1.0"})
    monkeypatch.setattr(builder, "model_data", lambda _: [])
    monkeypatch.setattr(builder, "binary_closure", lambda _: [])
    monkeypatch.setattr(builder, "external_library_references", lambda _: [])
    collected = []

    def pyinstaller(command, **kwargs):
        spec = Path(command[-1])
        assignment = ast.parse(spec.read_text()).body[0]
        sources = ast.literal_eval(
            next(keyword.value for keyword in assignment.value.keywords if keyword.arg == "datas")
        )
        for source, destination in sources:
            # The data consumer resolves relative paths from the .spec directory,
            # which is a different directory from the CLI's checkout root.
            path = Path(source)
            content = (path if path.is_absolute() else spec.parent / path).read_bytes()
            collected.append((destination, content))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(builder.subprocess, "run", pyinstaller)
    builder.build(
        SimpleNamespace(
            allow_dirty=False,
            licenses_dir=Path("build/bundle-licenses"),
            ffmpeg=Path("tools/ffmpeg"),
            ffprobe=Path("tools/ffprobe"),
            output_dir=checkout / "dist",
        )
    )
    assert ("licenses", b"exact build provenance") in collected
    assert ("people_mosaic/assets", b"default fluffy illustration") in collected
