from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton

from people_mosaic.app import Window
from people_mosaic.domain import Analysis, Face, Person


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(
        "people_mosaic.app.engine.preview_frame", lambda *args: np.zeros((180, 320, 3), np.uint8)
    )
    window = Window()
    assert window.detection_threshold.value() == 80
    assert window.threshold_label.text() == "0.80 · 標準"
    thumbnail = np.zeros((112, 112, 3), np.uint8)
    faces = [Face((70, 50, 40, 40), 0, True, track_id=1), Face((230, 50, 40, 40), track_id=2)]
    people = [Person(np.ones(128), thumbnail, 0) for _ in range(256)]
    window.analyzed(
        Analysis(
            Path("/private/tmp/" + "非常に長い動画名" * 40 + ".mp4"),
            "unused",
            320,
            180,
            10,
            [faces, faces],
            [0, 0.1],
            people,
        )
    )
    window.set_busy(False)
    window.show()
    qapp.processEvents()
    yield window
    window.worker = None
    window.close()


def visible_rect(widget, window):
    """Intersect every ancestor, including scroll viewports, to detect clipped controls."""
    result = QRect(widget.mapTo(window, QPoint()), widget.size())
    ancestor = widget.parentWidget()
    while ancestor:
        result = result.intersected(QRect(ancestor.mapTo(window, QPoint()), ancestor.size()))
        if ancestor is window:
            break
        ancestor = ancestor.parentWidget()
    return result


@pytest.mark.parametrize("size", [(640, 480), (800, 600), (1100, 820)])
def test_saving_and_adjusting_remain_reachable_with_video_visible(window, qapp, monkeypatch, size):
    window.resize(*size)
    window.png_label.setText("雲画像の長いファイル名" * 40 + ".png")
    window.status.setText("状態のお知らせです。" * 40)
    exports = []
    monkeypatch.setattr(window, "start_work", lambda operation, success: exports.append(operation))
    for tab in range(2):
        window.tabs.setCurrentIndex(tab)
        qapp.processEvents()
        assert (window.width(), window.height()) == size  # Do not silently grow beyond screen.
        assert window.canvas.isVisible()
        assert visible_rect(window.canvas, window).height() > 0
        assert visible_rect(window.export_button, window).size() == window.export_button.size()
        assert window.export_button.isEnabled()
        QTest.mouseClick(window.export_button, Qt.MouseButton.LeftButton)
    assert len(exports) == 2  # No confirmation checkbox or intermediate dialog.
    for control in (
        window.detection_threshold,
        window.png,
        window.undo,
        window.format,
        window.analysis_mode,
    ):
        window.settings_scroll.ensureWidgetVisible(control, 0, 0)
        qapp.processEvents()
        assert visible_rect(control, window).size() == control.size()
        assert window.canvas.isVisible()
    window.detection_threshold.setFocus()
    before = window.detection_threshold.value()
    QTest.keyClick(window.detection_threshold, Qt.Key.Key_Right)
    assert window.detection_threshold.value() == before + 1
    assert window.canvas.isVisible()


@pytest.mark.parametrize("size", [(640, 480), (800, 600)])
def test_busy_cancel_stays_visible_and_blocks_changes(window, qapp, size):
    class Running:
        def isRunning(self):
            return True

    window.worker = Running()
    window.resize(*size)
    window.tabs.setCurrentIndex(1)
    window.set_busy(True)
    qapp.processEvents()
    assert (window.width(), window.height()) == size
    assert window.progress_row.isVisible()
    assert visible_rect(window.cancel_button, window).size() == window.cancel_button.size()
    assert not window.export_button.isEnabled()
    assert not window.detection_threshold.isEnabled()
    assert not window.analysis_mode.isEnabled()
    window.worker = None
    window.set_busy(False)
    assert not window.progress_row.isVisible()
    assert window.export_button.isEnabled()


def test_light_palette_overrides_dark_system_palette(qapp):
    dark = QPalette()
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        dark.setColor(group, QPalette.ColorRole.Window, QColor("black"))
        dark.setColor(group, QPalette.ColorRole.Base, QColor("black"))
        dark.setColor(group, QPalette.ColorRole.Text, QColor("lime"))
    qapp.setPalette(dark)
    window = Window()
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        for role in (
            QPalette.ColorRole.Window,
            QPalette.ColorRole.Base,
            QPalette.ColorRole.Button,
            QPalette.ColorRole.ToolTipBase,
        ):
            assert qapp.palette().color(group, role).lightnessF() > 0.9
        assert qapp.palette().color(group, QPalette.ColorRole.Text).lightnessF() < 0.6
    assert window.format.palette().color(QPalette.ColorRole.Base).name() == "#ffffff"
    window.close()


def test_click_undo_includes_people_and_unassigned_detections(window, qapp):
    window.reveal_cloud(window.analysis.frames[0][0])
    window.reveal_cloud(window.analysis.frames[0][1])
    assert window.revealed() == {0}
    assert window.excluded_tracks == [2]
    window.undo_exclusion()
    assert window.excluded_tracks == []
    assert window.revealed() == {0}
    window.undo_exclusion()
    assert window.revealed() == set()
    assert not window.undo_exclusion_button.isEnabled()
    window.tabs.setCurrentIndex(1)
    window.region_button.setChecked(True)
    assert window.canvas.isVisible() and window.canvas.drawing
    QTest.mouseClick(window.play, Qt.MouseButton.LeftButton)
    assert not window.canvas.drawing
    window.tabs.setCurrentIndex(0)
    window.jump_to_person(0)
    assert window.tabs.currentIndex() == 0 and window.canvas.isVisible()


def test_expanded_reset_matches_main_reset(window, qapp):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialog

    window.reveal_cloud(window.analysis.frames[0][1])
    window.people.reveal(0)
    window.detection_threshold.setValue(85)

    def reset_dialog():
        dialog = window.people.findChild(QDialog)
        button = next(b for b in dialog.findChildren(QPushButton) if b.text() == "すべての顔を隠す")
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        dialog.reject()

    QTimer.singleShot(0, reset_dialog)
    window.people.show_overview()
    assert window.excluded_tracks == [] and window.cloud_undo == []
    assert window.revealed() == set() and window.detection_threshold.value() == 80


def test_small_screen_initial_person_checkbox_is_visible(window, qapp):
    window.resize(800, 600)
    window.tabs.setCurrentIndex(0)
    qapp.processEvents()
    view = window.people.view
    first = view.model().index(0, 0)
    check = view.itemDelegate().checkbox_rect(view.visualRect(first))
    mapped = QRect(view.viewport().mapTo(window, check.topLeft()), check.size())
    assert visible_rect(view.viewport(), window).contains(mapped)
    assert window.people_scroll.verticalScrollBar().value() == 0
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=check.center())
    assert window.revealed() == {0}
    assert window.canvas.isVisible()


def test_export_choices_use_mov_for_high_quality_editing(window, monkeypatch):
    assert window.format.count() == 2
    assert window.format.itemText(0).startswith("MP4")
    assert window.format.itemText(1).startswith("MOV")
    calls = []
    monkeypatch.setattr(
        "people_mosaic.app.engine.export", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(window, "start_work", lambda operation, success: operation(None, None))
    window.format.setCurrentIndex(1)
    assert "MOV" in window.save_note.text()
    assert "大きく" in window.quality_note.text()
    window.write_video()
    args, kwargs = calls[-1]
    assert args[1] == window.analysis.path.with_name(window.analysis.path.stem + "_mosaic.mov")
    assert args[5] is False and kwargs["mov"] is True
    window.format.setCurrentIndex(0)
    window.write_video()
    args, kwargs = calls[-1]
    assert args[1].suffix == ".mp4"
    assert args[5] is True and kwargs["mov"] is False
