"""Generate a local app launcher and an Automator Finder Quick Action."""

import argparse
import ctypes
import plistlib
import shlex
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
APPKIT = "/System/Library/Frameworks/AppKit.framework/AppKit"


def service_update_function():
    """Bind the native API without changing the user's services database."""
    appkit = ctypes.CDLL(APPKIT)
    update = appkit.NSUpdateDynamicServices
    update.argtypes = []
    update.restype = None
    return update


def refresh_services() -> None:
    """Notify macOS immediately after installing a service on a fresh machine.

    NSUpdateDynamicServices is the public AppKit API for service discovery.
    Flushing pbs' cache alone does not request immediate rediscovery.
    """
    service_update_function()()


def build(root: Path = ROOT, *, bundled: bool = False) -> None:
    directory = root / "build"
    directory.mkdir(exist_ok=True)
    python = shlex.quote(str(root / ".venv/bin/python"))
    launch = f'export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"\ncd {shlex.quote(str(root))}\n'
    executable_path = directory / "People Mosaic.app/Contents/MacOS/PeopleMosaic"
    runner = shlex.quote(str(executable_path)) if bundled else f"{python} -m people_mosaic.app"
    command = (
        "" if bundled else launch
    ) + f'for selected_video in "$@"; do\n  {runner} "$selected_video"\ndone\n'
    action_id = str(uuid4()).upper()
    action = {
        "AMAccepts": {"Container": "List", "Optional": True, "Types": ["com.apple.cocoa.string"]},
        "AMProvides": {"Container": "List", "Types": ["com.apple.cocoa.string"]},
        "AMActionVersion": "2.0.3",
        "ActionBundlePath": "/System/Library/Automator/Run Shell Script.action",
        "ActionName": "Run Shell Script",
        "ActionParameters": {
            "CheckedForUserDefaultShell": True,
            "COMMAND_STRING": command,
            "inputMethod": 1,
            "shell": "/bin/zsh",
            "source": "",
        },
        "BundleIdentifier": "com.apple.RunShellScript",
        "CFBundleVersion": "2.0.3",
        "CanShowSelectedItemsWhenRun": False,
        "CanShowWhenRun": True,
        "Category": ["AMCategoryUtilities"],
        "Class Name": "RunShellScriptAction",
        "InputUUID": str(uuid4()).upper(),
        "OutputUUID": str(uuid4()).upper(),
        "UUID": action_id,
        "ShowWhenRun": False,
        "isViewVisible": False,
        "UnlocalizedApplications": ["Automator"],
    }
    workflow = {
        "AMApplicationBuild": "521",
        "AMApplicationVersion": "2.10",
        "AMDocumentVersion": "2",
        "actions": [{"action": action, "isViewVisible": False}],
        "connectors": {},
        "workflowMetaData": {
            "applicationBundleID": "com.apple.finder",
            "applicationBundleIDsByPath": {
                "/System/Library/CoreServices/Finder.app": "com.apple.finder"
            },
            "applicationPath": "/System/Library/CoreServices/Finder.app",
            "applicationPaths": ["/System/Library/CoreServices/Finder.app"],
            "inputTypeIdentifier": "com.apple.Automator.fileSystemObject.movie",
            "outputTypeIdentifier": "com.apple.Automator.nothing",
            "presentationMode": 15,
            "processesInput": False,
            "serviceApplicationBundleID": "com.apple.finder",
            "serviceApplicationPath": "/System/Library/CoreServices/Finder.app",
            "serviceInputTypeIdentifier": "com.apple.Automator.fileSystemObject.movie",
            "serviceOutputTypeIdentifier": "com.apple.Automator.nothing",
            "serviceProcessesInput": False,
            "systemImageName": "NSActionTemplate",
            "useAutomaticInputType": False,
            "workflowTypeIdentifier": "com.apple.Automator.servicesMenu",
        },
    }
    contents = directory / "People Mosaic.workflow/Contents"
    contents.mkdir(parents=True, exist_ok=True)
    (contents / "document.wflow").write_bytes(plistlib.dumps(workflow))
    info = {
        "CFBundleIdentifier": "local.people-mosaic.finder",
        "CFBundleName": "People Mosaic",
        "MosaicOwner": "local.people-mosaic.release",
        "NSServices": [
            {
                "NSBackgroundColorName": "background",
                "NSIconName": "NSActionTemplate",
                "NSMenuItem": {"default": "People Mosaic"},
                "NSMessage": "runWorkflowAsService",
                "NSSendFileTypes": ["public.movie"],
                "NSRequiredContext": {"NSApplicationIdentifier": "com.apple.finder"},
            }
        ],
    }
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    if bundled:
        print(f"Quick Action: {directory / 'People Mosaic.workflow'}")
        return
    app_contents = directory / "People Mosaic.app/Contents"
    macos = app_contents / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    executable = macos / "PeopleMosaic"
    executable.write_text(
        "#!/bin/zsh\nset -eu\n" + launch + f"exec {python} -m people_mosaic.app\n"
    )
    executable.chmod(0o755)
    (app_contents / "Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleName": "People Mosaic",
                "CFBundleIdentifier": "local.people-mosaic.app",
                "CFBundleExecutable": "PeopleMosaic",
                "CFBundlePackageType": "APPL",
                "CFBundleVersion": "1",
                "CFBundleShortVersionString": "0.1.0",
                "NSHighResolutionCapable": True,
            }
        )
    )
    print(f"App: {BUILD / 'People Mosaic.app'}")
    print(f"Quick Action: {BUILD / 'People Mosaic.workflow'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-services", action="store_true")
    if parser.parse_args().refresh_services:
        refresh_services()
    else:
        build()
