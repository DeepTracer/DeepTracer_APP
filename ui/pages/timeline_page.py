from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QScrollArea,
    QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    StrongBodyLabel,
    SubtitleLabel,
    TitleLabel,
)


def conf_color(conf: float) -> str:
    if conf >= 0.8: return "#22c55e"
    if conf >= 0.5: return "#eab308"
    return "#ef4444"


class TimelineEvent(CardWidget):
    def __init__(self, event: dict, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setSpacing(16)

        # 종류 태그
        kind  = event.get("kind", "")
        color = "#ea580c" if kind == "plate" else "#1d4ed8"
        label = "번호판" if kind == "plate" else "표지판"
        tag   = CaptionLabel(f"  {label}  ", self)
        tag.setStyleSheet(
            f"background:{color}; color:#fff; border-radius:10px; padding:2px 6px;"
        )
        tag.setFixedWidth(60)

        # 시간
        time_label = CaptionLabel(
            f"frame {event.get('frame', 0)}\n{_fmt(event.get('time_sec', 0))}", self
        )
        time_label.setFixedWidth(90)
        time_label.setAlignment(Qt.AlignCenter)

        # 텍스트
        text_label = TitleLabel(event.get("label", "-"), self)
        text_label.setStyleSheet("font-family: monospace;")

        # 신뢰도
        conf = event.get("confidence", 0.0)
        conf_label = StrongBodyLabel(f"{conf:.3f}", self)
        conf_label.setStyleSheet(f"color: {conf_color(conf)};")
        conf_label.setFixedWidth(60)
        conf_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(tag)
        layout.addWidget(time_label)
        layout.addWidget(text_label)
        layout.addStretch()
        layout.addWidget(conf_label)


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class TimelinePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("timelinePage")
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 24, 36, 24)
        root.setSpacing(16)

        root.addWidget(SubtitleLabel("통합 타임라인", self))

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none;")

        self.inner  = QWidget()
        self.layout_ = QVBoxLayout(self.inner)
        self.layout_.setAlignment(Qt.AlignTop)
        self.layout_.setSpacing(8)

        self.scroll.setWidget(self.inner)
        root.addWidget(self.scroll)

        self._empty_label = BodyLabel("분석 결과가 없습니다.", self.inner)
        self._empty_label.setAlignment(Qt.AlignCenter)
        self.layout_.addWidget(self._empty_label)

    def load_timeline(self, timeline: list):
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not timeline:
            self.layout_.addWidget(BodyLabel("타임라인 데이터 없음", self.inner))
            return

        for event in timeline[:200]:
            self.layout_.addWidget(TimelineEvent(event, self.inner))