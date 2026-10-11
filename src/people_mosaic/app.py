import argparse
import bisect
import os
import sys
import time
from pathlib import Path
from threading import Event

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import engine
from .domain import Face, hidden_faces
from .errors import UserError
from .overlay import blend_bgr, cloud, load_png
from .people_picker import PeoplePicker


def pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return QPixmap.fromImage(
        QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
    )


class Worker(QThread):
    progress = Signal(int, str)
    result = Signal(object)
    failure = Signal(str)

    def __init__(self, operation):
        super().__init__()
        self.cancel = Event()
        self.operation = operation

    def run(self):
        try:
            self.result.emit(self.operation(self.progress.emit, self.cancel))
        except engine.Cancelled:
            self.failure.emit("処理をキャンセルしました。")
        except UserError as error:
            self.failure.emit(str(error))
        except Exception:
            self.failure.emit(
                "処理に失敗しました。動画の形式と保存先の空き容量を確認してください。"
            )


class Canvas(QWidget):
    region_added = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.image = None
        self.drawing = False
        self.start = self.end = None
        self.setAccessibleName(
            "加工後の動画プレビュー。追加範囲モードでドラッグすると隠す範囲を追加できます"
        )

    def image_rect(self):
        if self.image is None:
            return QRectF()
        size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF(
            (self.width() - size.width()) / 2,
            (self.height() - size.height()) / 2,
            size.width(),
            size.height(),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#152026"))
        if self.image is not None:
            painter.drawPixmap(self.image_rect().toRect(), self.image)
        else:
            painter.setPen(QColor("#b7c7cd"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "動画をここへドロップ\nまたは「動画を選ぶ」から開始",
            )
        if self.start is not None and self.end is not None:
            painter.setPen(QPen(QColor("#76e3bc"), 2))
            painter.drawRect(QRectF(self.start, self.end).normalized())

    def mousePressEvent(self, event):
        if self.drawing and self.image_rect().contains(event.position()):
            self.start = self.end = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start is None or self.image is None:
            return
        rect = QRectF(self.start, event.position()).normalized().intersected(self.image_rect())
        display = self.image_rect()
        sx, sy = self.image.width() / display.width(), self.image.height() / display.height()
        if rect.width() > 4 and rect.height() > 4:
            self.region_added.emit(
                (
                    (rect.x() - display.x()) * sx,
                    (rect.y() - display.y()) * sy,
                    rect.width() * sx,
                    rect.height() * sy,
                )
            )
        self.start = self.end = None
        self.update()


class Window(QMainWindow):
    def __init__(self, initial: Path | None = None):
        super().__init__()
        self.analysis = None
        self.input_path = None
        self.worker = None
        self.rgba = cloud()
        self.manual = []
        self.preview_raw = None
        self.preview_index = -1
        self.play_start = None
        self.setWindowTitle("People Mosaic · 雲で顔を隠す")
        self.resize(1280, 880)
        self.setAcceptDrops(True)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(9)
        heading = QLabel("People Mosaic")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        layout.addWidget(QLabel("見せたい人だけ、選ぶ。あとは雲にまかせる。"))
        top = QHBoxLayout()
        self.open_button = QPushButton("動画を選ぶ")
        self.open_button.clicked.connect(self.choose_video)
        top.addWidget(self.open_button)
        self.filename = QLabel("動画・顔の画像はこの Mac の中だけで処理します")
        top.addWidget(self.filename, 1)
        layout.addLayout(top)
        analysis_row = QHBoxLayout()
        analysis_row.addWidget(QLabel("顔の解析"))
        self.analysis_mode = QComboBox()
        self.analysis_mode.addItems(["高速 · 長辺 1920px", "小さい顔重視 · 元の解像度"])
        self.analysis_mode.setAccessibleName("顔の解析方法")
        self.analysis_mode.currentIndexChanged.connect(self.reanalyze)
        analysis_row.addWidget(self.analysis_mode)
        analysis_note = QLabel("出力は元の解像度。小さい顔が漏れる場合は元の解像度で再解析。")
        analysis_note.setWordWrap(True)
        analysis_row.addWidget(analysis_note, 1)
        layout.addLayout(analysis_row)
        split = QSplitter()
        self.canvas = Canvas()
        self.canvas.region_added.connect(self.add_region)
        split.addWidget(self.canvas)
        people_panel = QWidget()
        people_layout = QVBoxLayout(people_panel)
        people_top = QHBoxLayout()
        people_top.addWidget(QLabel("雲を付けない人物候補"), 1)
        self.overview_button = QPushButton("大きな一覧で選ぶ")
        self.overview_button.clicked.connect(self.show_people_overview)
        people_top.addWidget(self.overview_button)
        people_layout.addLayout(people_top)
        note = QLabel("チェックで雲を外します。顔写真をクリックすると登場位置へ移動します。")
        note.setWordWrap(True)
        people_layout.addWidget(note)
        self.people = PeoplePicker()
        self.people.setMinimumWidth(330)
        self.people.selection_changed.connect(self.repaint_preview)
        self.people.candidate_clicked.connect(self.jump_to_person)
        people_layout.addWidget(self.people, 1)
        self.summary = QLabel(
            "同じ人が複数候補に分かれる場合があります。\n似ているか曖昧なフレームは隠します。"
        )
        self.summary.setWordWrap(True)
        people_layout.addWidget(self.summary)
        reset = QPushButton("全員を隠す")
        reset.clicked.connect(self.hide_everyone)
        people_layout.addWidget(reset)
        self.reset_button = reset
        split.addWidget(people_panel)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([700, 490])
        layout.addWidget(split, 1)
        scrub = QHBoxLayout()
        self.play = QPushButton("再生")
        self.play.clicked.connect(self.toggle_play)
        scrub.addWidget(self.play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setAccessibleName("動画の確認位置")
        self.slider.valueChanged.connect(self.schedule_preview)
        scrub.addWidget(self.slider, 1)
        self.time_label = QLabel("0:00 / 0:00")
        scrub.addWidget(self.time_label)
        layout.addLayout(scrub)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(90)
        self.debounce.timeout.connect(self.load_preview)
        self.playback = QTimer(self)
        self.playback.setInterval(100)
        self.playback.timeout.connect(self.tick)
        overlay_row = QHBoxLayout()
        self.png = QPushButton("透過 PNG を選ぶ")
        self.png.clicked.connect(self.choose_png)
        overlay_row.addWidget(self.png)
        self.default_png = QPushButton("白い雲に戻す")
        self.default_png.clicked.connect(self.reset_cloud)
        overlay_row.addWidget(self.default_png)
        self.png_label = QLabel("白い雲 · 中央は不透明、周囲はふんわり")
        overlay_row.addWidget(self.png_label, 1)
        layout.addLayout(overlay_row)
        region_row = QHBoxLayout()
        self.region_button = QPushButton("隠す範囲を追加")
        self.region_button.setCheckable(True)
        self.region_button.toggled.connect(self.set_draw)
        region_row.addWidget(self.region_button)
        self.undo = QPushButton("追加範囲を 1 つ戻す")
        self.undo.clicked.connect(self.undo_region)
        region_row.addWidget(self.undo)
        self.region_note = QLabel("ドラッグで追加 · 動画全体の同じ位置を隠します")
        region_row.addWidget(self.region_note, 1)
        layout.addLayout(region_row)
        output_row = QHBoxLayout()
        self.format = QComboBox()
        self.format.addItems(["公開用 MP4 · 高画質", "保存用 MKV · 可逆圧縮"])
        self.format.setAccessibleName("出力形式")
        self.format.currentIndexChanged.connect(self.format_changed)
        output_row.addWidget(self.format)
        self.quality_note = QLabel()
        self.quality_note.setWordWrap(True)
        output_row.addWidget(self.quality_note, 1)
        self.export_button = QPushButton("確定して書き出す")
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(self.write_video)
        output_row.addWidget(self.export_button)
        layout.addLayout(output_row)
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        progress_row.addWidget(self.progress, 1)
        self.cancel_button = QPushButton("キャンセル")
        self.cancel_button.clicked.connect(self.cancel_work)
        progress_row.addWidget(self.cancel_button)
        layout.addLayout(progress_row)
        self.status = QLabel(
            "最初は全員を隠します。候補を選び、プレビューで確認してから書き出します。"
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.format_changed()
        self.set_busy(False)
        if initial:
            QTimer.singleShot(0, lambda: self.load_video(initial))

    def busy(self):
        return self.worker is not None and self.worker.isRunning()

    def set_busy(self, busy):
        ready = self.analysis is not None and not busy
        self.canvas.setEnabled(ready)
        self.open_button.setEnabled(not busy)
        self.analysis_mode.setEnabled(not busy)
        for widget in (
            self.people,
            self.overview_button,
            self.reset_button,
            self.slider,
            self.png,
            self.default_png,
            self.region_button,
            self.undo,
            self.play,
            self.format,
        ):
            widget.setEnabled(ready)
        self.cancel_button.setEnabled(busy)
        self.refresh_export()

    def refresh_export(self):
        self.export_button.setEnabled(self.analysis is not None and not self.busy())

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "動画を選ぶ",
            "",
            "動画 (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;すべてのファイル (*)",
        )
        if path:
            self.load_video(Path(path))

    def reanalyze(self):
        if self.input_path and not self.busy():
            self.load_video(self.input_path)

    def load_video(self, path):
        if self.busy():
            return
        self.input_path = path.resolve()
        max_edge = 1920 if self.analysis_mode.currentIndex() == 0 else None
        self.stop_play()
        self.analysis = None
        self.format_changed()
        self.manual.clear()
        self.people.set_candidates([])
        self.canvas.image = None
        self.canvas.update()
        self.preview_raw = None
        self.preview_index = -1
        self.filename.setText(path.name)
        self.setWindowTitle(f"People Mosaic · {path.name}")
        self.region_note.setText("ドラッグで追加 · 動画全体の同じ位置を隠します")
        self.start_work(
            lambda progress, cancel: engine.analyze(path.resolve(), progress, cancel, max_edge),
            self.analyzed,
        )

    def start_work(self, operation, success):
        self.stop_play()
        self.debounce.stop()
        self.progress.setValue(0)
        worker = Worker(operation)
        self.worker = worker
        worker.progress.connect(self.update_progress)
        worker.result.connect(success)
        worker.failure.connect(self.failed)
        worker.finished.connect(self.work_finished)
        worker.start()
        self.set_busy(True)

    def update_progress(self, value, message):
        self.progress.setValue(value)
        self.status.setText(message)

    def work_finished(self):
        self.set_busy(False)

    def analyzed(self, result):
        self.analysis = result
        self.people.set_candidates(
            (
                index,
                pixmap(person.thumbnail),
                result.times[person.first_frame] - result.times[0],
                person.count,
            )
            for index, person in enumerate(result.people)
        )
        self.slider.setRange(0, len(result.frames) - 1)
        self.slider.setValue(0)
        self.status.setText(
            "解析が完了しました。検出漏れは「隠す範囲を追加」で補ってください。"
            if result.people
            else "人物候補がありません。検出漏れの可能性があるため、動画を確認し必要な範囲を追加してください。"
        )
        self.progress.setValue(100)
        self.format_changed()
        self.load_preview()

    @staticmethod
    def clock(seconds):
        return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"

    def revealed(self):
        return self.people.revealed()

    def schedule_preview(self):
        if self.analysis:
            self.debounce.start()

    def load_preview(self):
        if self.analysis is None:
            return
        index = self.slider.value()
        try:
            if index != self.preview_index:
                self.preview_raw = engine.preview_frame(self.analysis, index)
                self.preview_index = index
            self.repaint_preview()
            relative = self.analysis.times[index] - self.analysis.times[0]
            total = self.analysis.times[-1] - self.analysis.times[0]
            self.time_label.setText(f"{self.clock(relative)} / {self.clock(total)}")
        except Exception:
            self.stop_play()
            self.status.setText("プレビューを読み込めませんでした。動画を再選択してください。")

    def repaint_preview(self, *args):
        if self.analysis is None or self.preview_raw is None:
            return
        faces = (
            hidden_faces(self.analysis.frames[self.preview_index], self.revealed()) + self.manual
        )
        self.canvas.image = pixmap(blend_bgr(self.preview_raw, faces, self.rgba))
        self.canvas.update()

    def jump_to_person(self, person):
        if self.analysis:
            self.slider.setValue(self.analysis.people[person].first_frame)

    def show_people_overview(self):
        if self.analysis and not self.busy():
            self.stop_play()
            self.people.show_overview()

    def hide_everyone(self):
        self.people.hide_everyone()

    def choose_png(self):
        path, _ = QFileDialog.getOpenFileName(self, "顔に重ねる透過 PNG を選ぶ", "", "PNG (*.png)")
        if not path:
            return
        try:
            rgba = load_png(Path(path))
        except UserError as error:
            QMessageBox.warning(self, "PNG を確認してください", str(error))
            return
        except Exception:
            QMessageBox.warning(self, "PNG を確認してください", "この PNG を読み込めませんでした。")
            return
        self.rgba = rgba
        self.png_label.setText(Path(path).name)
        self.repaint_preview()

    def reset_cloud(self):
        self.rgba = cloud()
        self.png_label.setText("白い雲 · 中央は不透明、周囲はふんわり")
        self.repaint_preview()

    def set_draw(self, enabled):
        self.canvas.drawing = enabled
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        if enabled:
            self.stop_play()

    def add_region(self, box):
        self.manual.append(Face(box))
        self.region_note.setText(f"追加 {len(self.manual)} 箇所 · 動画全体の同じ位置を隠します")
        self.repaint_preview()

    def undo_region(self):
        if self.manual:
            self.manual.pop()
            self.region_note.setText(f"追加 {len(self.manual)} 箇所 · 動画全体の同じ位置を隠します")
            self.repaint_preview()

    def format_changed(self):
        if self.analysis and self.analysis.hlg and self.format.currentIndex() == 0:
            self.quality_note.setText("HDR を維持する HEVC 10bit。再圧縮あり。対応音声はコピー。")
            return
        self.quality_note.setText(
            "映像は再圧縮。対応音声はコピー（それ以外は AAC）。"
            if self.format.currentIndex() == 0
            else "加工範囲外の画素を維持。音声はコピー。容量は大きくなります。"
        )

    def toggle_play(self):
        if self.playback.isActive():
            self.stop_play()
        elif self.analysis:
            if self.slider.value() == len(self.analysis.frames) - 1:
                self.slider.setValue(0)
            self.region_button.setChecked(False)
            self.play_start = time.monotonic(), self.analysis.times[self.slider.value()]
            self.play.setText("一時停止")
            self.playback.start()
            self.status.setText("加工後のプレビューを再生中（確認用の間引き再生・音声なし）")

    def tick(self):
        started, timestamp = self.play_start
        index = min(
            len(self.analysis.frames) - 1,
            bisect.bisect_left(self.analysis.times, timestamp + time.monotonic() - started),
        )
        self.slider.setValue(index)
        self.debounce.stop()
        self.load_preview()
        if index == len(self.analysis.frames) - 1:
            self.stop_play()

    def stop_play(self):
        self.playback.stop()
        self.play.setText("再生")

    def write_video(self):
        if not self.analysis or self.busy():
            return
        mp4 = self.format.currentIndex() == 0
        extension = ".mp4" if mp4 else ".mkv"
        output = self.analysis.path.with_name(self.analysis.path.stem + "_mosaic" + extension)
        revealed, rgba, manual = self.revealed(), self.rgba.copy(), list(self.manual)
        self.start_work(
            lambda progress, cancel: engine.export(
                self.analysis, output, revealed, rgba, manual, mp4, progress, cancel
            ),
            self.exported,
        )

    def exported(self, path):
        self.status.setText(f"書き出し完了 · {path.name}")
        dialog = QMessageBox(self)
        dialog.setWindowTitle("書き出しが完了しました")
        dialog.setText(f"{path.name}\n保存した動画を再生してから公開してください。")
        show = dialog.addButton("Finder で表示", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("閉じる", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        if dialog.clickedButton() == show:
            import subprocess

            subprocess.Popen(["/usr/bin/open", "-R", str(path)])

    def failed(self, message):
        self.status.setText(message)
        self.progress.setValue(0)

    def cancel_work(self):
        if self.busy():
            self.worker.cancel.set()
            self.status.setText("キャンセルしています…")

    def closeEvent(self, event):
        self.stop_play()
        if self.busy():
            self.cancel_work()
            self.status.setText("処理が停止してからウィンドウを閉じてください。")
            event.ignore()
        else:
            event.accept()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and not self.busy():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            self.load_video(Path(urls[0].toLocalFile()))
        else:
            self.status.setText("動画を 1 本ずつドロップしてください。")


STYLE = """
QWidget { font-family: '.AppleSystemUIFont'; font-size: 13px; color: #1d3237; }
QMainWindow, QMainWindow > QWidget { background: #f4f6f3; }
QLabel#heading { font-size: 30px; font-weight: 700; }
QPushButton { background: #ffffff; border: 1px solid #cbd7d0; border-radius: 8px; padding: 9px 14px; }
QPushButton:hover { background: #e9f1ec; }
QPushButton:checked { background: #d2eddf; }
QPushButton:disabled { color: #99a8a1; background: #eef1ee; }
QPushButton#primary { background: #216950; color: white; border: none; font-weight: 600; }
QPushButton#primary:disabled { background: #b7ccc1; }
QDialog { background: #f4f6f3; }
QListView { background: #f4f6f3; border: 1px solid #d5dfd8; border-radius: 10px; padding: 5px; }
QComboBox { padding: 8px; background: white; border: 1px solid #cbd7d0; border-radius: 8px; }
QProgressBar { border: none; border-radius: 5px; background: #e1e8e2; height: 12px; text-align: center; }
QProgressBar::chunk { background: #76b49b; border-radius: 5px; }
"""


def main():
    parser = argparse.ArgumentParser(description="動画の顔を白い雲や透過 PNG で隠す")
    parser.add_argument("video", nargs="?", type=Path)
    args = parser.parse_args()
    # Finder's service runner has a minimal PATH.
    os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("People Mosaic")
    app.setStyle("Fusion")
    app.setPalette(app.style().standardPalette())
    app.setStyleSheet(STYLE)
    window = Window(args.video)
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
