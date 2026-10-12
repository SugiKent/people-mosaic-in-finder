import gc
import os

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from people_mosaic.app import STYLE


@pytest.fixture(scope="session")
def qapp():
    # Qt allows only one application per process. Keep it alive across test modules;
    # destroying and recreating it can crash the macOS platform/style integration.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setPalette(app.style().standardPalette())
    app.setStyleSheet(STYLE)
    return app


@pytest.fixture(autouse=True)
def dispose_test_widgets():
    yield
    app = QApplication.instance()
    if app is not None:
        # Destroy UI objects on the main thread before later video tests start
        # worker threads that could otherwise trigger collection of Qt cycles.
        for widget in app.topLevelWidgets():
            widget.close()
            widget.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()
