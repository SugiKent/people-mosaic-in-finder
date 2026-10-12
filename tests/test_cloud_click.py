import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from people_mosaic.app import Canvas, Window
from people_mosaic.domain import Analysis, Face, Person, hidden_faces


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def display_point(canvas, x, y):
    rect = canvas.image_rect()
    return QPointF(
        rect.x() + (x + 0.5) * rect.width() / canvas.image.width(),
        rect.y() + (y + 0.5) * rect.height() / canvas.image.height(),
    )


def click(canvas, x, y, button=Qt.MouseButton.LeftButton):
    QTest.mouseClick(canvas, button, pos=display_point(canvas, x, y).toPoint())


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(
        "people_mosaic.app.engine.preview_frame", lambda *args: np.zeros((180, 320, 3), np.uint8)
    )
    faces = [Face((70, 50, 40, 40), 0, True), Face((230, 50, 40, 40), 1, True)]
    people = [Person(np.ones(128), np.zeros((112, 112, 3), np.uint8), i) for i in range(2)]
    window = Window()
    window.analyzed(
        Analysis(
            Path("/private/tmp/click.mp4"), "unused", 320, 180, 10, [faces, faces], [0, 0.1], people
        )
    )
    window.set_busy(False)
    yield window
    window.worker = None
    window.close()


def test_click_reveals_candidate_in_all_frames_and_export(window, monkeypatch):
    window.people.sort_changed(1)
    window.people.filter_changed(True)  # The clicked candidate is not in the visible list.
    click(window.canvas, 90, 70)
    assert window.revealed() == {0}
    assert window.people.proxy.rowCount() == 1
    assert window.people.model.item(0).checkState() == Qt.CheckState.Checked
    assert "候補 1" in window.status.text()
    assert [face.person for face in hidden_faces(window.analysis.frames[1], window.revealed())] == [
        1
    ]
    # With its cloud removed, clicking the same face does not silently re-cover it.
    click(window.canvas, 90, 70)
    assert window.revealed() == {0}
    saved = []
    monkeypatch.setattr("people_mosaic.app.engine.export", lambda *args: saved.append(args[2]))
    monkeypatch.setattr(window, "start_work", lambda operation, success: operation(None, None))
    window.write_video()
    assert saved == [{0}]
    window.people.model.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert window.revealed() == set()
    assert window.canvas.cloud_at(display_point(window.canvas, 90, 70)).person == 0
    click(window.canvas, 250, 70)
    assert window.revealed() == {1}
    window.hide_everyone()
    assert window.revealed() == set()
    assert window.canvas.cloud_at(display_point(window.canvas, 250, 70)).person == 1


def test_click_pauses_on_displayed_frame_even_with_pending_scrub(window):
    window.toggle_play()
    window.slider.setValue(1)
    assert window.debounce.isActive()
    assert window.preview_index == 0
    click(window.canvas, 90, 70)
    assert not window.playback.isActive()
    assert not window.debounce.isActive()
    assert window.slider.value() == window.preview_index == 0
    assert window.revealed() == {0}


@pytest.mark.parametrize("kind", ["uncertain", "unassigned", "manual"])
def test_unselectable_top_cloud_blocks_underlying_person(window, kind):
    top = Face((70, 50, 40, 40), 1 if kind == "uncertain" else None, False)
    if kind == "manual":
        window.manual.append(top)
    else:
        window.analysis.frames[0].append(top)
    window.repaint_preview()
    click(window.canvas, 90, 70)
    assert window.revealed() == set()
    assert (
        "追加範囲" in window.status.text()
        if kind == "manual"
        else "識別できない" in window.status.text()
    )


def test_overlapping_cloud_reveals_only_topmost_candidate(window):
    window.analysis.frames[0][1].box = (70, 50, 40, 40)
    window.repaint_preview()
    click(window.canvas, 90, 70)
    assert window.revealed() == {1}
    assert window.canvas.cloud_at(display_point(window.canvas, 90, 70)).person == 0


@pytest.mark.parametrize("size", [(481, 501), (1001, 301)])
def test_letterbox_resize_alpha_and_clipping_match_drawn_pixels(qapp, size):
    canvas = Canvas()
    canvas.resize(*size)
    rgba = np.zeros((32, 64, 4), np.uint8)  # A wide custom PNG with transparent corners.
    rgba[:, :, :3] = 255
    rgba[8:24, 16:48, 3] = 255
    face = Face((-5, 5, 20, 20), 7, True)
    canvas.set_preview(np.zeros((180, 320, 3), np.uint8), [face], rgba)
    assert canvas.cloud_at(QPointF(0, 0)) is None
    _, left, top, alpha = canvas.overlays[0]
    assert left == top == 0
    assert canvas.cloud_at(display_point(canvas, 0, 0)) is None
    ys, xs = np.where(alpha > 0)
    for i in (0, len(xs) // 2, len(xs) - 1):
        assert canvas.cloud_at(display_point(canvas, xs[i], ys[i])) is face
    # A cloud pixel beyond the raw face rectangle is still a valid click.
    beyond = np.where((alpha > 0) & (np.indices(alpha.shape)[1] >= 15))
    assert len(beyond[0])
    assert canvas.cloud_at(display_point(canvas, beyond[1][0], beyond[0][0])) is face
    canvas.close()


def test_transparent_top_cloud_allows_hit_on_visible_cloud_below(qapp):
    canvas = Canvas()
    below = Face((70, 50, 40, 40), 0, True)
    # The second cloud overlaps only through a completely transparent corner.
    above = Face((100, 80, 40, 40), 1, True)
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[8:24, 8:24, :] = 255
    canvas.set_preview(np.zeros((180, 320, 3), np.uint8), [below, above], rgba)
    assert canvas.cloud_at(display_point(canvas, 90, 70)) is below
    canvas.close()


def test_right_click_drag_and_disabled_canvas_do_not_reveal(window):
    click(window.canvas, 90, 70, Qt.MouseButton.RightButton)
    start = display_point(window.canvas, 90, 70).toPoint()
    QTest.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=start + QPoint(30, 0))
    assert window.revealed() == set()
    window.set_busy(True)
    click(window.canvas, 90, 70)
    assert window.revealed() == set()


def test_changed_frame_between_press_and_release_does_not_reveal(window):
    start = display_point(window.canvas, 90, 70).toPoint()
    QTest.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=start)
    window.repaint_preview()
    QTest.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=start)
    assert window.revealed() == set()


def test_drawing_mode_adds_region_instead_of_revealing(window):
    window.region_button.setChecked(True)
    start = display_point(window.canvas, 85, 65).toPoint()
    end = display_point(window.canvas, 100, 80).toPoint()
    QTest.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=end)
    assert window.revealed() == set()
    assert len(window.manual) == 1
    window.region_button.setChecked(False)
    click(window.canvas, 90, 70)
    assert window.revealed() == set()
    window.undo_region()
    click(window.canvas, 90, 70)
    assert window.revealed() == {0}


def test_invalid_candidate_id_cannot_reveal_another_person(window):
    window.analysis.frames[0][0].person = 500
    window.repaint_preview()
    click(window.canvas, 90, 70)
    assert window.revealed() == set()
