"""Create a portable CPython runtime with LGPL-only video libraries for freezing."""

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PYTHON = "3.12.13"


def run(command, env):
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("Install uv on the build machine first.")
    prefix = ROOT / "build/ffmpeg"
    if not (prefix / "lib/pkgconfig/libavcodec.pc").is_file():
        raise SystemExit("Run scripts/build_ffmpeg.py first.")
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / "build/freeze-venv")
    env["MACOSX_DEPLOYMENT_TARGET"] = "15.0"
    # Managed portable Python has an older deployment target than a Homebrew
    # interpreter compiled on a newer macOS. Never inherit a local brew Python.
    run([uv, "python", "install", BUNDLE_PYTHON], env)
    python = subprocess.run(
        [uv, "python", "find", "--managed-python", BUNDLE_PYTHON],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run([uv, "sync", "--frozen", "--dev", "--group", "bundle", "--python", python], env)
    frozen_python = ROOT / "build/freeze-venv/bin/python"
    env["PKG_CONFIG_PATH"] = str(prefix / "lib/pkgconfig")
    env["PATH"] = os.pathsep.join(
        (str(frozen_python.parent), str(prefix / "bin"), env.get("PATH", ""))
    )
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(min(os.cpu_count() or 1, 8)))
    env["CPPFLAGS"] = "-I" + str(prefix / "include")
    env["LDFLAGS"] = "-L" + str(prefix / "lib")
    env["DYLD_LIBRARY_PATH"] = str(prefix / "lib")
    env["CMAKE_ARGS"] = " ".join(
        [
            "-DWITH_FFMPEG=OFF",
            "-DBUILD_opencv_videoio=OFF",
            "-DWITH_GSTREAMER=OFF",
            "-DWITH_QT=OFF",
            "-DBUILD_LIST=core,imgproc,dnn,objdetect,video,imgcodecs,python3",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PERF_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_JAVA=OFF",
            "-DWITH_IPP=OFF",
            "-DWITH_ITT=OFF",
            "-DWITH_OPENEXR=OFF",
        ]
    )
    versions = json.loads(
        subprocess.run(
            [
                str(frozen_python),
                "-c",
                "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) "
                "for n in ('av','opencv-python-headless')}))",
            ],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    for name in ("av", "opencv-python-headless"):
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(frozen_python),
                "--no-build-isolation",
                "--no-binary",
                name,
                "--reinstall",
                "--no-deps",
                f"{name}=={versions[name]}",
            ],
            env,
        )
    # Do not let a package manager silently substitute the GPL-containing wheels.
    run(
        [
            str(frozen_python),
            "-c",
            "import av,cv2; assert 'libx264' not in av.codecs_available; "
            "assert 'libopenh264' in av.codecs_available; "
            "import re; assert not re.search(r'FFMPEG:\\s+YES', cv2.getBuildInformation()); "
            "assert hasattr(cv2,'FaceDetectorYN_create'); "
            "assert hasattr(cv2,'FaceRecognizerSF_create'); print(av.library_versions)",
        ],
        env,
    )


if __name__ == "__main__":
    main()
