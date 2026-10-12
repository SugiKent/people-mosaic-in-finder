from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from test_cloud_click import click, display_point

from people_mosaic.app import Window
from people_mosaic.domain import Analysis, Face, hidden_faces


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(
        "people_mosaic.app.engine.preview_frame", lambda *args: np.zeros((180, 320, 3), np.uint8)
    )
    faces = [
        Face((70, 50, 40, 40), score=0.82, track_id=1),
        Face((230, 50, 40, 40), score=0.95, track_id=2),
    ]
    window = Window()
    window.analyzed(
        Analysis(
            Path("/private/tmp/threshold.mp4"), "unused", 320, 180, 10, [faces, faces], [0, 0.1]
        )
    )
    window.set_busy(False)
    yield window
    window.worker = None
    window.close()


def test_slider_applies_instantly_without_analysis_and_keeps_manual_masks(window, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Changing the threshold must not reanalyze or decode the video")

    monkeypatch.setattr("people_mosaic.app.engine.analyze", forbidden)
    monkeypatch.setattr("people_mosaic.app.engine.preview_frame", forbidden)
    window.add_region((140, 50, 20, 20))
    assert window.canvas.cloud_at(display_point(window.canvas, 90, 70)) is not None
    window.detection_threshold.setValue(85)
    assert window.canvas.cloud_at(display_point(window.canvas, 90, 70)) is None
    assert window.canvas.cloud_at(display_point(window.canvas, 250, 70)) is not None
    assert window.canvas.cloud_at(display_point(window.canvas, 150, 60)) is window.manual[0]
    assert window.threshold_label.text() == "0.85"
    window.detection_threshold.setValue(30)
    assert window.canvas.cloud_at(display_point(window.canvas, 90, 70)) is not None


def test_unassigned_track_click_applies_across_frames_and_can_be_undone(window):
    click(window.canvas, 90, 70)
    assert window.revealed() == set()
    assert window.excluded_tracks == [1]
    assert window.undo_exclusion_button.isEnabled()
    masks = hidden_faces(
        window.analysis.frames[1], set(), excluded_tracks=set(window.excluded_tracks)
    )
    assert [face.track_id for face in masks] == [2]
    window.slider.setValue(1)
    window.load_preview()
    assert window.canvas.cloud_at(display_point(window.canvas, 90, 70)) is None
    window.undo_exclusion()
    assert window.excluded_tracks == []
    assert not window.undo_exclusion_button.isEnabled()
    assert window.canvas.cloud_at(display_point(window.canvas, 90, 70)).track_id == 1
    click(window.canvas, 90, 70)
    click(window.canvas, 250, 70)
    assert window.excluded_tracks == [1, 2]
    window.undo_exclusion()
    assert window.excluded_tracks == [1]
    window.detection_threshold.setValue(95)
    window.hide_everyone()
    assert window.excluded_tracks == []
    assert window.detection_threshold.value() == 80
    assert window.canvas.cloud_at(display_point(window.canvas, 90, 70)).track_id == 1


def test_export_captures_threshold_and_track_exclusions(window, monkeypatch):
    click(window.canvas, 90, 70)
    window.detection_threshold.setValue(85)
    operations, exported = [], []
    monkeypatch.setattr(
        window, "start_work", lambda operation, success: operations.append(operation)
    )
    monkeypatch.setattr(
        "people_mosaic.app.engine.export", lambda *args, **kwargs: exported.append(kwargs)
    )
    window.write_video()
    window.detection_threshold.setValue(30)
    window.excluded_tracks.clear()
    operations[0](None, None)
    assert exported == [{"mov": False, "score_threshold": 0.85, "excluded_tracks": {1}}]


def test_busy_disables_slider_and_exclusion_undo(window):
    click(window.canvas, 90, 70)
    window.set_busy(True)
    assert not window.detection_threshold.isEnabled()
    assert not window.undo_exclusion_button.isEnabled()
    click(window.canvas, 250, 70)
    assert window.excluded_tracks == [1]
    window.set_busy(False)
    assert window.detection_threshold.isEnabled()
    assert window.undo_exclusion_button.isEnabled()


def test_new_video_clears_threshold_and_track_exclusions(window, monkeypatch):
    click(window.canvas, 90, 70)
    window.detection_threshold.setValue(85)
    monkeypatch.setattr(window, "start_work", lambda *args: None)
    window.load_video(Path("/private/tmp/next.mp4"))
    assert window.excluded_tracks == []
    assert window.detection_threshold.value() == 80
    assert window.analysis is None
    assert window.canvas.overlays == []
    assert not window.undo_exclusion_button.isEnabled()


def test_manual_region_still_blocks_track_click(window):
    window.manual.append(Face((70, 50, 40, 40)))
    window.repaint_preview()
    click(window.canvas, 90, 70)
    assert window.excluded_tracks == []
    assert "追加範囲" in window.status.text()
    window.region_button.setChecked(True)
    assert window.canvas.cursor().shape() == Qt.CursorShape.CrossCursor
