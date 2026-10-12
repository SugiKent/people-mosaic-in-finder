import os

import pytest
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
