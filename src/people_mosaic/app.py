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
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen, QPixmap
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
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import engine
from .domain import DEFAULT_DETECTION_SCORE_THRESHOLD, Face, hidden_faces
from .errors import UserError
from .overlay import PreparedOverlay, blend_bgr, cloud, load_png
from .people_picker import PeoplePicker
from .runtime import frozen


def pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return QPixmap.fromImage(
        QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
    )


def light_palette():
    """Never inherit macOS appearance: this application uses a light UI only."""
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#f6f7f4",
        QPalette.ColorRole.WindowText: "#253c36",
        QPalette.ColorRole.Base: "#ffffff",
        QPalette.ColorRole.AlternateBase: "#edf2ed",
        QPalette.ColorRole.Text: "#253c36",
        QPalette.ColorRole.Button: "#ffffff",
        QPalette.ColorRole.ButtonText: "#253c36",
        QPalette.ColorRole.Highlight: "#d2eddf",
        QPalette.ColorRole.HighlightedText: "#173d2e",
        QPalette.ColorRole.ToolTipBase: "#ffffff",
        QPalette.ColorRole.ToolTipText: "#253c36",
        QPalette.ColorRole.Link: "#216950",
        QPalette.ColorRole.LinkVisited: "#216950",
        QPalette.ColorRole.Light: "#ffffff",
        QPalette.ColorRole.Midlight: "#e7eee8",
        QPalette.ColorRole.Mid: "#a8b8ad",
        QPalette.ColorRole.Dark: "#697a70",
        QPalette.ColorRole.Shadow: "#45594d",
        QPalette.ColorRole.PlaceholderText: "#53665f",
        QPalette.ColorRole.Accent: "#216950",
    }
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        for role, color in colors.items():
            palette.setColor(group, role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#69776f"))
    return palette


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
    cloud_pressed = Signal()
    cloud_clicked = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 80)
        self.image = None
        self.overlays = []
        self.pressed_cloud = None
        self.drawing = False
        self.start = self.end = None
        self.setAccessibleName(
            "加工後の動画プレビュー。クリックした雲を外します。"
            "追加範囲モードではドラッグで隠す範囲を追加できます"
        )
        self.setToolTip("雲をクリックして、人物候補またはその連続した検出を除外します")

    def set_preview(self, bgr, faces, rgba):
        prepared = PreparedOverlay(rgba)
        self.image = pixmap(blend_bgr(bgr, faces, rgba, prepared=prepared))
        self.overlays = []
        for face in faces:
            region = prepared.placement(face.box, bgr.shape[1], bgr.shape[0])
            if region is not None:
                x, y, patch = region
                self.overlays.append((face, x, y, patch[:, :, 3].copy()))
        self.pressed_cloud = None
        self.update()

    def cloud_at(self, position):
        display = self.image_rect()
        if self.image is None or not display.contains(position) or display.isEmpty():
            return None
        x = int((position.x() - display.x()) * self.image.width() / display.width())
        y = int((position.y() - display.y()) * self.image.height() / display.height())
        # Later clouds are drawn on top. An unselectable cloud also blocks clicks.
        for face, left, top, alpha in reversed(self.overlays):
            px, py = x - left, y - top
            if 0 <= py < alpha.shape[0] and 0 <= px < alpha.shape[1] and alpha[py, px] > 0:
                return face
        return None

    def image_rect(self):
        if self.image is None:
            return QRectF()
        size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF(
            QRectF(
                (self.width() - size.width()) / 2,
                (self.height() - size.height()) / 2,
                size.width(),
                size.height(),
            ).toRect()
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#e7eee8"))
        if self.image is not None:
            painter.drawPixmap(self.image_rect().toRect(), self.image)
        else:
            painter.setPen(QColor("#53665f"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "動画をここへドロップ\nまたは「動画を選ぶ」から開始",
            )
        if self.start is not None and self.end is not None:
            painter.setPen(QPen(QColor("#76e3bc"), 2))
            painter.drawRect(QRectF(self.start, self.end).normalized())

    def mousePressEvent(self, event):
        self.pressed_cloud = None
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton:
            return
        if self.drawing and self.image_rect().contains(event.position()):
            self.start = self.end = event.position()
            self.update()
        elif not self.drawing:
            face = self.cloud_at(event.position())
            if face is not None:
                self.pressed_cloud = (face, event.position())
                self.cloud_pressed.emit()

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        pressed, self.pressed_cloud = self.pressed_cloud, None
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.drawing:
            if (
                pressed is not None
                and (event.position() - pressed[1]).manhattanLength()
                <= QApplication.startDragDistance()
                and self.cloud_at(event.position()) is pressed[0]
            ):
                self.cloud_clicked.emit(pressed[0])
            return
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
        QApplication.instance().styleHints().setColorScheme(Qt.ColorScheme.Light)
        QApplication.instance().setPalette(light_palette())
        self.analysis = None
        self.input_path = None
        self.worker = None
        self.rgba = cloud()
        self.manual = []
        self.excluded_tracks = []
        self.cloud_undo = []
        self.preview_raw = None
        self.preview_index = -1
        self.play_start = None
        self.default_threshold = round(DEFAULT_DETECTION_SCORE_THRESHOLD * 100)
        self.setWindowTitle("People Mosaic · 動画の顔を雲で隠す")
        self.setMinimumSize(640, 480)
        screen = self.screen().availableGeometry()
        self.resize(min(1100, screen.width() - 40), min(820, screen.height() - 60))
        self.setAcceptDrops(True)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QVBoxLayout()
        heading = QLabel("動画の顔を、雲で隠す")
        heading.setObjectName("heading")
        title.addWidget(heading)
        self.filename = QLabel("People Mosaic · 動画はこの Mac の中だけで処理します")
        self.filename.setObjectName("muted")
        self.filename.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        title.addWidget(self.filename)
        header.addLayout(title, 1)
        self.open_button = QPushButton("動画を選ぶ…")
        self.open_button.clicked.connect(self.choose_video)
        header.addWidget(self.open_button)
        layout.addLayout(header)
        self.steps = QLabel("① 動画を選ぶ   →   ② 雲を確認する   →   ③ 保存する")
        self.steps.setObjectName("steps")
        layout.addWidget(self.steps)

        self.pages = QStackedWidget()
        self.pages.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        welcome = QWidget()
        welcome_layout = QVBoxLayout(welcome)
        welcome_layout.addStretch()
        self.welcome_title = QLabel("ここに動画をドロップ")
        self.welcome_title.setObjectName("welcomeTitle")
        self.welcome_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(self.welcome_title)
        self.welcome_note = QLabel(
            "または右上の「動画を選ぶ…」から始めましょう。\n顔を見つけて、自動で雲を重ねます。"
        )
        self.welcome_note.setWordWrap(True)
        self.welcome_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(self.welcome_note)
        welcome_layout.addStretch()
        self.pages.addWidget(welcome)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.tabs.setAccessibleName("雲を外す人の選択、調整と設定")
        self.tabs.setMinimumWidth(260)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.editor = QSplitter(Qt.Orientation.Horizontal)
        self.editor.setChildrenCollapsible(False)
        self.pages.addWidget(self.editor)
        layout.addWidget(self.pages, 1)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 12, 12, 0)
        hint = QLabel("最初は全員を隠します。雲を外すときはクリック。")
        hint.setWordWrap(True)
        preview_layout.addWidget(hint)
        self.canvas = Canvas()
        self.canvas.region_added.connect(self.add_region)
        self.canvas.cloud_pressed.connect(self.pause_preview_for_cloud)
        self.canvas.cloud_clicked.connect(self.reveal_cloud)
        preview_layout.addWidget(self.canvas, 1)
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
        preview_layout.addLayout(scrub)

        def scroll_page(widget):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(widget)
            return scroll

        self.preview_scroll = scroll_page(preview)
        self.editor.addWidget(self.preview_scroll)
        self.editor.addWidget(self.tabs)
        self.editor.setStretchFactor(0, 3)
        self.editor.setStretchFactor(1, 1)
        self.editor.setSizes([720, 340])
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(90)
        self.debounce.timeout.connect(self.load_preview)
        self.playback = QTimer(self)
        self.playback.setInterval(100)
        self.playback.timeout.connect(self.tick)

        people_panel = QWidget()
        people_layout = QVBoxLayout(people_panel)
        people_layout.setContentsMargins(0, 12, 0, 0)
        self.overview_button = QPushButton("人物一覧を広げる")
        self.overview_button.clicked.connect(self.show_people_overview)
        self.people = PeoplePicker(compact=True, reset_all=self.hide_everyone)
        self.people.selection_changed.connect(self.repaint_preview)
        self.people.candidate_clicked.connect(self.jump_to_person)
        people_layout.addWidget(self.people, 1)
        self.summary = QLabel(
            "写真をクリックすると動画の登場場面へ移動します。\n"
            "同じ人が複数の写真に分かれることがあります。判定が曖昧な場面は隠します。"
        )
        self.summary.setWordWrap(True)
        self.summary.setObjectName("muted")
        self.summary.hide()
        self.people.setToolTip(self.summary.text())
        self.people_scroll = scroll_page(people_panel)
        self.tabs.addTab(self.people_scroll, "雲を外す人")

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings = QWidget()
        settings.setMinimumWidth(0)
        self.settings_scroll.setWidget(settings)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 12, 12, 12)
        settings_layout.setSpacing(16)
        self.tabs.addTab(self.settings_scroll, "調整・設定")

        def section(title, description):
            card = QWidget()
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 12, 12, 12)
            card_layout.setSpacing(8)
            label = QLabel(title)
            label.setObjectName("sectionTitle")
            card_layout.addWidget(label)
            note = QLabel(description)
            note.setWordWrap(True)
            note.setObjectName("muted")
            card_layout.addWidget(note)
            settings_layout.addWidget(card)
            return card_layout

        detection = section(
            "顔を見つける細かさ",
            "右へ動かすと、看板などの誤検出を減らせます。"
            "実際の顔も隠れなくなるので、動画で確認してください。",
        )
        threshold_row = QHBoxLayout()

        self.detection_threshold = QSlider(Qt.Orientation.Horizontal)
        self.detection_threshold.setRange(30, 99)
        self.detection_threshold.setValue(self.default_threshold)
        self.detection_threshold.setAccessibleName("顔を見つける細かさ。右ほど検出を減らす")
        self.detection_threshold.setToolTip("顔検出の閾値。変更はすぐ反映され、再解析は不要です。")
        self.detection_threshold.valueChanged.connect(self.threshold_changed)
        threshold_row.addWidget(self.detection_threshold, 1)

        detection.addLayout(threshold_row)
        threshold_ends = QHBoxLayout()
        threshold_ends.addWidget(QLabel("多く見つける"))
        threshold_ends.addStretch()
        threshold_ends.addWidget(QLabel("検出を減らす"))
        detection.addLayout(threshold_ends)
        threshold_detail = QHBoxLayout()
        self.threshold_label = QLabel(f"{DEFAULT_DETECTION_SCORE_THRESHOLD:.2f} · 標準")
        threshold_detail.addWidget(self.threshold_label, 1)
        self.threshold_reset = QPushButton("標準に戻す")
        self.threshold_reset.clicked.connect(
            lambda: self.detection_threshold.setValue(self.default_threshold)
        )
        threshold_detail.addWidget(self.threshold_reset)
        detection.addLayout(threshold_detail)

        appearance = section(
            "雲の見た目", "自分の画像も使えます。背景が透明な PNG を選んでください。"
        )
        overlay_row = QVBoxLayout()
        self.png = QPushButton("画像を選ぶ…")
        self.png.clicked.connect(self.choose_png)
        overlay_row.addWidget(self.png)
        self.default_png = QPushButton("元の雲に戻す")
        self.default_png.clicked.connect(self.reset_cloud)
        overlay_row.addWidget(self.default_png)
        appearance.addLayout(overlay_row)
        self.png_label = QLabel("同梱の雲画像 · 元の透過をそのまま使用")
        self.png_label.setWordWrap(True)
        self.png_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        appearance.addWidget(self.png_label)

        regions = section(
            "隠したい場所を足す",
            "顔を見つけられないときに使います。動画の上でドラッグして囲みます。"
            "囲んだ場所は、動画全体の同じ位置で隠れます。",
        )
        region_row = QVBoxLayout()
        self.region_button = QPushButton("動画で範囲を囲む")
        self.region_button.setCheckable(True)
        self.region_button.toggled.connect(self.set_draw)
        region_row.addWidget(self.region_button)
        self.undo = QPushButton("追加した範囲を戻す")
        self.undo.clicked.connect(self.undo_region)
        region_row.addWidget(self.undo)
        regions.addLayout(region_row)
        self.region_note = QLabel("追加した範囲はありません")
        self.region_note.setWordWrap(True)
        regions.addWidget(self.region_note)

        saving = section(
            "保存する動画の形式", "保存先は元の動画と同じフォルダです。元の動画は残ります。"
        )
        self.format = QComboBox()
        self.format.addItems(["MP4 · 共有用（おすすめ）", "MKV · 画質を保つ保存用"])
        self.format.setAccessibleName("保存する動画の形式")
        self.format.currentIndexChanged.connect(self.format_changed)
        saving.addWidget(self.format)
        self.quality_note = QLabel()
        self.quality_note.setWordWrap(True)
        saving.addWidget(self.quality_note)

        analysis = section(
            "小さい顔が見つからないとき",
            "「小さい顔も探す」に変更すると、動画を読み直します。"
            "雲を外した選択と追加した範囲はリセットされます。",
        )
        self.analysis_mode = QComboBox()
        self.analysis_mode.addItems(["標準 · 速く探す", "小さい顔も探す"])
        self.analysis_mode.setAccessibleName("顔の解析方法")
        self.analysis_mode.currentIndexChanged.connect(self.reanalyze)
        analysis.addWidget(self.analysis_mode)
        settings_layout.addStretch()

        # The body can shrink/scroll; saving and progress stay outside it.
        actions = QHBoxLayout()
        self.reset_button = QPushButton("すべての顔を隠す")
        self.reset_button.clicked.connect(self.hide_everyone)
        actions.addWidget(self.reset_button)
        self.undo_exclusion_button = QPushButton("外した雲を1つ戻す")
        self.undo_exclusion_button.clicked.connect(self.undo_exclusion)
        actions.addWidget(self.undo_exclusion_button)
        actions.addWidget(self.overview_button)
        actions.addStretch()
        layout.addLayout(actions)
        self.exclusion_note = QLabel(
            "最初は全員の顔に雲が付きます。選ばずに保存することもできます。"
        )
        self.exclusion_note.setWordWrap(True)
        self.exclusion_note.setObjectName("muted")
        self.exclusion_note.hide()
        layout.addWidget(self.exclusion_note)

        self.footer = QWidget()
        self.footer.setObjectName("footer")
        footer_layout = QVBoxLayout(self.footer)
        footer_layout.setContentsMargins(16, 12, 16, 12)
        footer_layout.setSpacing(8)
        save_row = QHBoxLayout()
        self.save_note = QLabel("動画を選ぶと、保存できるようになります")
        self.save_note.setWordWrap(True)
        self.save_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        save_row.addWidget(self.save_note, 1)
        self.export_button = QPushButton("動画を保存する")
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(self.write_video)
        save_row.addWidget(self.export_button)
        footer_layout.addLayout(save_row)
        self.progress_row = QWidget()
        progress_layout = QHBoxLayout(self.progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        progress_layout.addWidget(self.progress, 1)
        self.cancel_button = QPushButton("処理を中止")
        self.cancel_button.clicked.connect(self.cancel_work)
        progress_layout.addWidget(self.cancel_button)
        footer_layout.addWidget(self.progress_row)
        self.status = QLabel("動画を選んで始めましょう。")
        self.status.setWordWrap(True)
        self.status.setFixedHeight(44)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        footer_layout.addWidget(self.status)
        layout.addWidget(self.footer)
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
            self.detection_threshold,
            self.threshold_reset,
        ):
            widget.setEnabled(ready)
        self.cancel_button.setEnabled(busy)
        self.progress_row.setVisible(busy)
        self.steps.setVisible(busy or self.analysis is None)
        self.steps.setText(
            "② 顔を探しています…"
            if busy and self.analysis is None
            else "③ 動画を保存しています…"
            if busy
            else "② 雲を確認する   →   ③ 保存する"
            if self.analysis is not None
            else "① 動画を選ぶ   →   ② 雲を確認する   →   ③ 保存する"
        )
        self.undo_exclusion_button.setEnabled(ready and bool(self.cloud_undo))
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
        self.pages.setCurrentIndex(0)
        self.welcome_title.setText("顔を探しています…")
        self.welcome_note.setText("このまま少しお待ちください。見つけた顔に雲を重ねます。")
        self.region_button.setChecked(False)
        max_edge = 1920 if self.analysis_mode.currentIndex() == 0 else None
        self.stop_play()
        self.analysis = None
        self.format_changed()
        self.manual.clear()
        self.excluded_tracks.clear()
        self.cloud_undo.clear()
        self.detection_threshold.setValue(self.default_threshold)
        self.update_exclusions()
        self.people.set_candidates([])
        self.canvas.image = None
        self.canvas.overlays.clear()
        self.canvas.pressed_cloud = None
        self.canvas.update()
        self.preview_raw = None
        self.preview_index = -1
        self.filename.setText(path.name)
        self.filename.setToolTip(str(path.resolve()))
        self.setWindowTitle(f"People Mosaic · {path.name}")
        self.region_note.setText("追加した範囲はありません")
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
        self.filename.setText(result.path.name)
        self.filename.setToolTip(str(result.path))
        self.pages.setCurrentIndex(1)
        self.tabs.setCurrentIndex(0)
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
            "準備できました。雲を確認して、よければ「動画を保存する」を押してください。"
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
            hidden_faces(
                self.analysis.frames[self.preview_index],
                self.revealed(),
                score_threshold=self.detection_threshold.value() / 100,
                excluded_tracks=set(self.excluded_tracks),
            )
            + self.manual
        )
        self.canvas.set_preview(self.preview_raw, faces, self.rgba)

    def pause_preview_for_cloud(self):
        self.stop_play()
        self.debounce.stop()
        # A pending scrub may point beyond the frame actually displayed.
        self.slider.blockSignals(True)
        self.slider.setValue(self.preview_index)
        self.slider.blockSignals(False)

    def reveal_cloud(self, face):
        if self.analysis is None or self.busy() or self.canvas.drawing:
            return
        if any(face is region for region in self.manual):
            self.status.setText("手動の追加範囲です。「追加した範囲を戻す」で取り消せます。")
        elif face.confident and face.person is not None and self.people.reveal(face.person):
            self.cloud_undo.append(("person", face.person))
            self.update_exclusions()
            self.status.setText(
                f"候補 {face.person + 1} の雲を外しました。戻すには一覧のチェックを外してください。"
            )
        elif face.track_id is not None:
            if face.track_id not in self.excluded_tracks:
                self.excluded_tracks.append(face.track_id)
                self.cloud_undo.append(("track", face.track_id))
            self.update_exclusions()
            self.repaint_preview()
            self.status.setText(
                "この検出の雲を外しました。連続して追跡できる範囲に反映します。"
                "戻すには「外した雲を1つ戻す」を押してください。"
            )
        else:
            self.status.setText("この検出を除外できません。動画を再解析してください。")

    def threshold_changed(self, value):
        self.threshold_label.setText(
            f"{value / 100:.2f}" + (" · 標準" if value == self.default_threshold else "")
        )
        self.repaint_preview()
        if self.analysis is not None:
            self.status.setText(
                f"検出の細かさを変更しました（{value / 100:.2f}）。動画と保存結果に反映します。"
                "高くすると、実際の顔も隠れなくなることがあります。"
            )

    def update_exclusions(self):
        count = len(self.excluded_tracks)
        self.undo_exclusion_button.setEnabled(
            self.analysis is not None and not self.busy() and bool(self.cloud_undo)
        )
        self.exclusion_note.setVisible(bool(count))
        self.exclusion_note.setText(
            f"クリックで除外した検出 {count} 件 · カットや追跡が途切れた後は再び隠します"
            if count
            else "最初は全員の顔に雲が付きます。選ばずに保存することもできます。"
        )

    def undo_exclusion(self):
        if not self.busy() and self.cloud_undo:
            kind, target = self.cloud_undo.pop()
            if kind == "track":
                if target in self.excluded_tracks:
                    self.excluded_tracks.remove(target)
            else:
                self.people.cover(target)
            self.update_exclusions()
            self.repaint_preview()
            self.status.setText("外した雲を1つ戻しました。")

    def tab_changed(self, index):
        # Pause when changing the inspector, keeping the video visible.
        if hasattr(self, "playback"):
            self.stop_play()

    def jump_to_person(self, person):
        if self.analysis:
            self.slider.setValue(self.analysis.people[person].first_frame)

    def show_people_overview(self):
        if self.analysis and not self.busy():
            self.stop_play()
            self.people.show_overview()

    def hide_everyone(self):
        self.excluded_tracks.clear()
        self.cloud_undo.clear()
        self.detection_threshold.setValue(self.default_threshold)
        self.update_exclusions()
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
        self.png_label.setToolTip(str(path))
        self.repaint_preview()

    def reset_cloud(self):
        self.rgba = cloud()
        self.png_label.setText("同梱の雲画像 · 元の透過をそのまま使用")
        self.repaint_preview()

    def set_draw(self, enabled):
        self.canvas.drawing = enabled
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        if enabled:
            self.stop_play()
            self.status.setText(
                "隠したい場所をドラッグで囲んでください。終了するには下のボタンを押します。"
            )
            self.play.setText("範囲追加を終了")
        else:
            self.play.setText("再生")

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
        mp4 = self.format.currentIndex() == 0
        if self.analysis:
            output = self.analysis.path.with_name(
                self.analysis.path.stem + "_mosaic" + (".mp4" if mp4 else ".mkv")
            )
            self.save_note.setText(f"{'MP4' if mp4 else 'MKV'} · 元の動画と同じフォルダに保存")
            self.save_note.setToolTip(f"{output}\n同名のファイルがあれば、末尾に番号を付けます。")
        if self.analysis and self.analysis.hlg and mp4:
            self.quality_note.setText(
                "元の HDR の明るさを保ち、高画質で圧縮します。画質は少し変わります。"
            )
        else:
            self.quality_note.setText(
                "共有しやすい高画質の動画です。圧縮により画質は少し変わります。"
                if mp4
                else "雲を重ねた部分以外の画質を保ちます。ファイルが大きくなり、再生できるアプリが限られます。"
            )

    def toggle_play(self):
        if self.canvas.drawing:
            self.region_button.setChecked(False)
            return
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
        self.play.setText("範囲追加を終了" if self.canvas.drawing else "再生")

    def write_video(self):
        if not self.analysis or self.busy():
            return
        mp4 = self.format.currentIndex() == 0
        extension = ".mp4" if mp4 else ".mkv"
        output = self.analysis.path.with_name(self.analysis.path.stem + "_mosaic" + extension)
        revealed, rgba, manual = self.revealed(), self.rgba.copy(), list(self.manual)
        threshold = self.detection_threshold.value() / 100
        excluded = set(self.excluded_tracks)
        self.start_work(
            lambda progress, cancel: engine.export(
                self.analysis,
                output,
                revealed,
                rgba,
                manual,
                mp4,
                progress,
                cancel,
                score_threshold=threshold,
                excluded_tracks=excluded,
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
        if self.analysis is None:
            self.welcome_title.setText("動画を読み込めませんでした")
            self.welcome_note.setText(message + "\n右上から別の動画を選べます。")
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
QWidget { font-family: '.AppleSystemUIFont'; font-size: 14px; color: #253c36; }
QMainWindow, QMainWindow > QWidget, QDialog { background: #f6f7f4; }
QLabel { background: transparent; }
QCheckBox { color: #253c36; background: transparent; }
QCheckBox:disabled { color: #69776f; }
QTabWidget::pane, QScrollArea > QWidget > QWidget { background: #f6f7f4; }
QSplitter::handle { background: #d9e0da; width: 5px; }
QScrollBar:vertical { background: #eef2ed; width: 12px; border: none; }
QScrollBar::handle:vertical { background: #a1b0a5; min-height: 28px; border-radius: 5px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QLabel#heading { font-size: 24px; font-weight: 700; }
QLabel#welcomeTitle { font-size: 26px; font-weight: 600; }
QLabel#sectionTitle { font-size: 17px; font-weight: 600; }
QLabel#muted { color: #53665f; }
QLabel#steps { color: #365d50; padding: 8px 12px; background: #e7eee8; border-radius: 8px; }
QWidget#card, QWidget#footer { background: white; border: 1px solid #d9e0da; border-radius: 12px; }
QPushButton { background: white; border: 1px solid #bbc9c0; border-radius: 8px;
              padding: 9px 14px; min-height: 22px; }
QPushButton:hover { background: #e8f0e9; border-color: #527764; }
QPushButton:focus { border: 2px solid #216950; }
QPushButton:checked { background: #d2eddf; }
QPushButton:disabled { color: #69776f; background: #eef1ee; border-color: #d8ded8; }
QPushButton#primary { background: #216950; color: white; border: 2px solid #216950; font-weight: 600; }
QPushButton#primary:hover { background: #174d3a; }
QPushButton#primary:focus { border-color: #071f16; }
QPushButton#primary:disabled { background: #d4dfd7; color: #53665f; border-color: #d4dfd7; }
QListView { background: #f6f7f4; border: 1px solid #d5dfd8; border-radius: 10px; padding: 5px; }
QComboBox { padding: 8px; min-height: 22px; background: white;
            border: 1px solid #bbc9c0; border-radius: 8px; }
QComboBox:focus { border: 2px solid #216950; }
QScrollArea { background: #f6f7f4; border: none; }
QTabWidget::pane { border: none; }
QTabBar::tab { padding: 12px 10px; color: #53665f; border-bottom: 3px solid transparent; }
QTabBar::tab:selected { color: #216950; border-bottom: 3px solid #216950; font-weight: 600; }
QTabBar::tab:hover { background: #e7eee8; }
QProgressBar { border: none; border-radius: 5px; background: #e1e8e2;
               height: 16px; text-align: center; }
QProgressBar::chunk { background: #85b59e; border-radius: 5px; }
QSlider::groove:horizontal { height: 6px; background: #d1ddd3; border-radius: 3px; }
QSlider::handle:horizontal { background: #216950; width: 18px; height: 18px;
                            margin: -6px 0; border-radius: 9px; }
QSlider::handle:horizontal:disabled { background: #94a69b; }
"""


def main():
    parser = argparse.ArgumentParser(description="動画の顔を白い雲や透過 PNG で隠す")
    parser.add_argument("video", nargs="?", type=Path)
    args = parser.parse_args()
    # Finder's service runner has a minimal PATH.
    if not frozen():
        os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("People Mosaic")
    app.setStyle("Fusion")
    app.setPalette(light_palette())
    app.setStyleSheet(STYLE)
    window = Window(args.video)
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
