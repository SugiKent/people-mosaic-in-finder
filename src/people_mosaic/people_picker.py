"""Face cards with one selection model shared by compact and expanded views."""

from PySide6.QtCore import QEvent, QRect, QSize, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QVBoxLayout,
    QWidget,
)

PERSON_ROLE = int(Qt.ItemDataRole.UserRole)
FIRST_ROLE = PERSON_ROLE + 1
COUNT_ROLE = PERSON_ROLE + 2
SECONDS_ROLE = PERSON_ROLE + 3


def is_checked(index):
    return index.data(Qt.ItemDataRole.CheckStateRole) in (2, Qt.CheckState.Checked)


class CandidateView(QListView):
    face_clicked = Signal(int)

    def mouseReleaseEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        person = index.data(PERSON_ROLE) if index.isValid() else None
        inspect_face = (
            person is not None
            and event.button() == Qt.MouseButton.LeftButton
            and not self.itemDelegate()
            .checkbox_rect(self.visualRect(index))
            .contains(event.position().toPoint())
        )
        super().mouseReleaseEvent(event)
        # A checked-only row can disappear during checkbox handling. Never resolve its
        # former visual row to another person after the filter has changed.
        if inspect_face:
            self.face_clicked.emit(person)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            index = self.currentIndex()
            if index.isValid() and not event.isAutoRepeat():
                self.model().setData(
                    index, 0 if is_checked(index) else 2, Qt.ItemDataRole.CheckStateRole
                )
            event.accept()
        else:
            super().keyPressEvent(event)


class CandidateFilter(QSortFilterProxyModel):
    def __init__(self, parent):
        super().__init__(parent)
        self.selected_only = False
        self.frequent_first = False
        self.setFilterRole(Qt.ItemDataRole.CheckStateRole)

    def filterAcceptsRow(self, row, parent):
        index = self.sourceModel().index(row, 0, parent)
        return not self.selected_only or is_checked(index)

    def lessThan(self, left, right):
        def key(index):
            first = index.data(SECONDS_ROLE)
            person = index.data(PERSON_ROLE)
            return (
                (-index.data(COUNT_ROLE), first, person) if self.frequent_first else (first, person)
            )

        return key(left) < key(right)


class FaceCardDelegate(QStyledItemDelegate):
    """Only the explicit checkbox (or Space) authorizes revealing a candidate."""

    def __init__(self, parent, compact=False):
        super().__init__(parent)
        self.photo_size = 80 if compact else 112
        self.card_size = QSize(146, self.photo_size + 90)

    def sizeHint(self, option, index):
        return self.card_size

    def checkbox_rect(self, rect):
        return QRect(rect.left() + 13, rect.top() + self.photo_size + 58, rect.width() - 26, 24)

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        checked = is_checked(index)
        focused = bool(option.state & QStyle.StateFlag.State_Selected)
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        card = option.rect.adjusted(4, 4, -4, -4)
        painter.setBrush(QColor("#e5f4eb" if checked else "#ffffff"))
        painter.setPen(
            QPen(QColor("#216950" if focused or checked else "#d5dfd8"), 2 if focused else 1)
        )
        painter.drawRoundedRect(card, 9, 9)
        photo = QRect(
            option.rect.left() + (option.rect.width() - self.photo_size) // 2,
            card.top() + 9,
            self.photo_size,
            self.photo_size,
        )
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon, QIcon):
            icon.paint(painter, photo)
        painter.setPen(QColor("#1d3237" if enabled else "#99a8a1"))
        painter.drawText(
            QRect(card.left() + 5, card.top() + self.photo_size + 13, card.width() - 10, 20),
            Qt.AlignmentFlag.AlignCenter,
            f"候補 {index.data(PERSON_ROLE) + 1}",
        )
        painter.setPen(QColor("#63766f"))
        painter.drawText(
            QRect(card.left() + 5, card.top() + self.photo_size + 34, card.width() - 10, 18),
            Qt.AlignmentFlag.AlignCenter,
            f"登場 {index.data(FIRST_ROLE)}",
        )
        check = QStyleOptionButton()
        check.rect = self.checkbox_rect(option.rect)
        check.text = "雲を付けない"
        check.state = QStyle.StateFlag.State_On if checked else QStyle.StateFlag.State_Off
        if enabled:
            check.state |= QStyle.StateFlag.State_Enabled
        QApplication.style().drawControl(QStyle.ControlElement.CE_CheckBox, check, painter)
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if not option.state & QStyle.StateFlag.State_Enabled:
            return False
        mouse = (
            event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and self.checkbox_rect(option.rect).contains(event.position().toPoint())
        )
        if mouse:
            checked = is_checked(index)
            return model.setData(
                index,
                0 if checked else 2,
                Qt.ItemDataRole.CheckStateRole,
            )
        # Do not let the standard delegate toggle via its different checkbox geometry.
        return False


class PeoplePicker(QWidget):
    selection_changed = Signal()
    candidate_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = QStandardItemModel(self)
        self.loading = False
        self.proxy = CandidateFilter(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.sort(0)
        self.model.itemChanged.connect(self.changed)
        self.proxy.rowsRemoved.connect(self.update_count)
        self.proxy.rowsInserted.connect(self.update_count)
        self.controls = []
        self.count_labels = []
        self.empty_labels = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.make_controls())
        self.view = self.make_view(compact=True)
        layout.addWidget(self.view, 1)
        self.add_count(layout)
        self.update_count()

    def make_view(self, compact=False):
        view = CandidateView()
        view.setModel(self.proxy)
        delegate = FaceCardDelegate(view, compact)
        view.setItemDelegate(delegate)
        view.setViewMode(QListView.ViewMode.IconMode)
        view.setMovement(QListView.Movement.Static)
        view.setResizeMode(QListView.ResizeMode.Adjust)
        view.setGridSize(delegate.card_size)
        if compact:
            view.setMinimumHeight(delegate.card_size.height() + 14)
        view.setUniformItemSizes(True)
        view.setSpacing(0)
        view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        view.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setAccessibleName("人物候補の顔カード。雲を付けないチェック、またはスペースで選択")
        view.face_clicked.connect(self.candidate_clicked)
        return view

    def make_controls(self):
        row = QHBoxLayout()
        sort = QComboBox()
        sort.addItems(["登場順", "検出回数が多い順"])
        sort.setAccessibleName("人物候補の並べ替え")
        sort.setCurrentIndex(int(self.proxy.frequent_first))
        only = QCheckBox("選択済みだけ")
        only.setChecked(self.proxy.selected_only)
        only.setToolTip("雲を付けない候補だけ表示します。表示を絞っても選択は保持されます。")
        sort.currentIndexChanged.connect(self.sort_changed)
        only.toggled.connect(self.filter_changed)
        self.controls.append((sort, only))
        row.addWidget(sort)
        row.addWidget(only)
        row.addStretch()
        return row

    def add_count(self, layout):
        empty = QLabel()
        empty.setWordWrap(True)
        layout.addWidget(empty)
        self.empty_labels.append(empty)
        count = QLabel()
        count.setWordWrap(True)
        layout.addWidget(count)
        self.count_labels.append(count)

    def set_candidates(self, candidates):
        """candidates: (immutable person ID, QPixmap, first appearance seconds, count)."""
        self.loading = True
        self.model.clear()
        for person, photo, seconds, count in candidates:
            first = f"{int(seconds) // 60}:{int(seconds) % 60:02d}"
            item = QStandardItem(f"候補 {person + 1} · 登場 {first} · 雲を付けない")
            item.setIcon(QIcon(photo))
            item.setEditable(False)
            item.setCheckable(True)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(person, PERSON_ROLE)
            item.setData(first, FIRST_ROLE)
            item.setData(count, COUNT_ROLE)
            item.setData(seconds, SECONDS_ROLE)
            item.setToolTip(f"候補 {person + 1} · 最初の登場 {first} · 検出 {count} 回")
            self.model.appendRow(item)
        self.loading = False
        self.filter_changed(False)
        self.update_count()

    def revealed(self):
        return {
            self.model.item(row).data(PERSON_ROLE)
            for row in range(self.model.rowCount())
            if self.model.item(row).checkState() == Qt.CheckState.Checked
        }

    def changed(self, item):
        if self.loading:
            return
        self.update_count()
        self.selection_changed.emit()

    def update_count(self, *args):
        if self.loading:
            return
        selected = len(self.revealed())
        text = f"全 {self.model.rowCount()} 候補 · 雲なし {selected} 候補"
        if self.proxy.selected_only:
            text += f" · 一覧に {self.proxy.rowCount()} 候補"
        for label in self.count_labels:
            label.setText(text)
        for label in self.empty_labels:
            label.setText(
                "選択済みの候補はありません。「選択済みだけ」を外して選んでください。"
                if self.proxy.selected_only
                else "解析すると顔写真がここに表示されます。"
            )
            label.setVisible(self.proxy.rowCount() == 0)

    def sort_changed(self, mode):
        self.proxy.frequent_first = bool(mode)
        self.proxy.invalidate()
        self.proxy.sort(0)
        for sort, _ in self.controls:
            sort.blockSignals(True)
            sort.setCurrentIndex(mode)
            sort.blockSignals(False)

    def filter_changed(self, selected_only):
        self.proxy.selected_only = selected_only
        self.proxy.invalidate()
        for _, only in self.controls:
            only.blockSignals(True)
            only.setChecked(selected_only)
            only.blockSignals(False)
        self.update_count()

    def hide_everyone(self):
        self.loading = True
        for row in range(self.model.rowCount()):
            self.model.item(row).setCheckState(Qt.CheckState.Unchecked)
        self.loading = False
        self.proxy.invalidate()
        self.update_count()
        self.selection_changed.emit()

    def show_overview(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("雲を付けない人物候補 · 大きな一覧")
        dialog.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        screen = self.screen().availableGeometry()
        dialog.resize(min(1250, screen.width() - 60), min(850, screen.height() - 60))
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 20, 24, 20)
        heading = QLabel("雲を付けない人を選ぶ")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        note = QLabel(
            "チェックはすぐ反映されます。顔写真をクリックすると動画の登場位置へ移動します。\n"
            "同じ人が複数候補に分かれる場合があります。似ているか曖昧なフレームは隠します。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addLayout(self.make_controls())
        view = self.make_view()
        layout.addWidget(view, 1)
        self.add_count(layout)
        buttons = QHBoxLayout()
        reset = QPushButton("全員を隠す")
        reset.clicked.connect(self.hide_everyone)
        buttons.addWidget(reset)
        buttons.addStretch()
        done = QPushButton("動画に戻って確認")
        done.setObjectName("primary")
        done.clicked.connect(dialog.accept)
        buttons.addWidget(done)
        layout.addLayout(buttons)
        self.update_count()
        try:
            dialog.exec()
        finally:
            # The model survives; discard only controls belonging to this temporary window.
            self.controls.pop()
            self.count_labels.pop()
            self.empty_labels.pop()
            dialog.deleteLater()
