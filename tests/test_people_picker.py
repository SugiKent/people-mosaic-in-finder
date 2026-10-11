import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QListView

from people_mosaic.app import STYLE, Window
from people_mosaic.domain import Analysis, Person
from people_mosaic.people_picker import PERSON_ROLE, PeoplePicker


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setPalette(app.style().standardPalette())
    app.setStyleSheet(STYLE)
    return app


def candidates(count=256):
    photo = QPixmap(112, 112)
    photo.fill(Qt.GlobalColor.lightGray)
    return [(i, photo, i * 4, 256 - i) for i in range(count)]


def test_grid_clicks_and_keyboard_reveal_only_explicit_candidate(qapp):
    picker = PeoplePicker()
    picker.resize(1000, 600)
    picker.set_candidates(candidates())
    picker.show()
    qapp.processEvents()
    assert picker.proxy.rowCount() == 256
    assert picker.revealed() == set()
    first = picker.proxy.index(0, 0)
    second = picker.proxy.index(1, 0)
    assert picker.view.visualRect(first).top() == picker.view.visualRect(second).top()
    rect = picker.view.visualRect(second)
    inspected = []
    picker.candidate_clicked.connect(inspected.append)
    QTest.mouseClick(picker.view.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert picker.revealed() == set()  # Inspecting a face must not authorize revealing it.
    QTest.mouseClick(
        picker.view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=picker.view.itemDelegate().checkbox_rect(rect).center(),
    )
    assert picker.revealed() == {1}
    assert inspected == [1]
    picker.filter_changed(True)
    qapp.processEvents()
    only_rect = picker.view.visualRect(picker.proxy.index(0, 0))
    QTest.mouseClick(
        picker.view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=picker.view.itemDelegate().checkbox_rect(only_rect).center(),
    )
    assert picker.revealed() == set()
    assert picker.proxy.rowCount() == 0
    assert inspected == [1]
    picker.filter_changed(False)
    picker.view.setCurrentIndex(picker.proxy.index(1, 0))
    QTest.keyClick(picker.view, Qt.Key.Key_Space)
    assert picker.revealed() == {1}
    picker.close()


def test_filter_sort_and_reset_preserve_immutable_person_ids(qapp):
    picker = PeoplePicker()
    values = candidates(4)
    # Times crossing 10 minutes must sort numerically; frequency reverses the IDs.
    values = [
        (i, photo, seconds, count)
        for (i, photo, _, _), seconds, count in zip(
            values, [601, 120, 60, 0], [1, 8, 6, 2], strict=True
        )
    ]
    picker.set_candidates(values)
    assert picker.proxy.index(0, 0).data(PERSON_ROLE) == 3
    picker.sort_changed(1)
    assert picker.proxy.index(0, 0).data(PERSON_ROLE) == 1
    picker.proxy.setData(
        picker.proxy.index(0, 0), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole
    )
    picker.filter_changed(True)
    assert picker.proxy.rowCount() == 1
    assert picker.revealed() == {1}
    picker.model.item(2).setCheckState(Qt.CheckState.Checked)
    assert picker.proxy.rowCount() == 2
    picker.proxy.setData(
        picker.proxy.index(0, 0), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole
    )
    assert picker.revealed() == {2}
    assert picker.proxy.rowCount() == 1
    picker.filter_changed(False)
    assert picker.revealed() == {2}
    picker.hide_everyone()
    assert picker.revealed() == set()
    picker.filter_changed(True)
    assert picker.proxy.rowCount() == 0
    assert "ありません" in picker.empty_labels[0].text()
    picker.set_candidates(values)
    assert not picker.proxy.selected_only
    assert picker.revealed() == set()
    assert picker.proxy.rowCount() == 4


def test_expanded_window_uses_same_selection_and_survives_reopening(qapp):
    picker = PeoplePicker()
    picker.set_candidates(candidates())
    observations = []

    def select_and_close():
        dialog = picker.findChild(QDialog)
        view = dialog.findChild(QListView)
        assert view.model() is picker.proxy
        picker.sort_changed(1)
        view.model().setData(
            view.model().index(255, 0), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole
        )
        picker.filter_changed(True)
        observations.append((picker.revealed(), view.model().rowCount()))
        dialog.reject()  # Closing the live selection window is not an undo operation.

    for _ in range(2):
        picker.filter_changed(False)
        QTimer.singleShot(0, select_and_close)
        picker.show_overview()
        assert len(picker.controls) == len(picker.count_labels) == len(picker.empty_labels) == 1
        qapp.sendPostedEvents(None, 0)
        qapp.processEvents()
    assert observations == [({255}, 1), ({255}, 1)]
    assert picker.revealed() == {255}


def test_export_available_after_analysis_without_confirmation(qapp, monkeypatch):
    monkeypatch.setattr(
        "people_mosaic.app.engine.preview_frame", lambda *args: np.zeros((120, 160, 3), np.uint8)
    )
    window = Window()
    assert not window.export_button.isEnabled()
    thumbnail = np.zeros((112, 112, 3), np.uint8)
    analysis = Analysis(
        Path("/private/tmp/ui-test.mp4"),
        "unused",
        160,
        120,
        10,
        [[], []],
        [0, 0.1],
        [Person(np.ones(128), thumbnail, 1)],
    )
    window.analyzed(analysis)
    window.set_busy(False)
    assert not hasattr(window, "review")
    assert window.export_button.isEnabled()
    window.people.model.item(0).setCheckState(Qt.CheckState.Checked)
    assert window.revealed() == {0}
    assert window.export_button.isEnabled()
    window.set_busy(True)
    assert not window.people.isEnabled()
    assert not window.overview_button.isEnabled()

    # set_busy is normally called after the worker starts; simulate its running state.
    class Running:
        def isRunning(self):
            return True

    window.worker = Running()
    window.refresh_export()
    assert not window.export_button.isEnabled()
    window.worker = None
    window.close()
