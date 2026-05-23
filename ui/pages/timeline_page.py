import base64
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPainterPath
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
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
    if conf >= 0.8:
        return "#22c55e"
    if conf >= 0.5:
        return "#eab308"
    return "#ef4444"


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _rounded_pixmap(src: QPixmap, w: int, h: int, radius: int = 8) -> QPixmap:
    if src.isNull():
        return QPixmap()
    scaled = src.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    out = QPixmap(scaled.width(), scaled.height())
    out.fill(Qt.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing, True)
    path = QPainterPath()
    path.addRoundedRect(0, 0, scaled.width(), scaled.height(), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, scaled)
    painter.end()
    return out


def _load_thumb_pixmap(event: dict, w: int, h: int) -> QPixmap:
    crop_path = event.get("crop_path")
    if crop_path and Path(crop_path).exists():
        src = QPixmap(crop_path)
        if not src.isNull():
            return _rounded_pixmap(src, w, h)

    b64 = event.get("image_base64")
    if b64:
        try:
            raw = base64.b64decode(b64)
            src = QPixmap()
            src.loadFromData(raw)
            if not src.isNull():
                return _rounded_pixmap(src, w, h)
        except Exception:
            pass

    return QPixmap()


class TimelineEvent(CardWidget):
    THUMB_W = 200
    THUMB_H = 110

    def __init__(self, event: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("timelineEventCard")
        self.setBorderRadius(10)

        self.setStyleSheet("""
            #timelineEventCard {
                background-color: #2b2b2b;
                border: 1px solid #3a3a3a;
            }
            #timelineEventCard:hover {
                background-color: #353535;
                border: 1px solid #505050;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 22, 12)
        layout.setSpacing(20)

        # 썸네일
        thumb = QLabel(self)
        thumb.setFixedSize(self.THUMB_W, self.THUMB_H)
        thumb.setAlignment(Qt.AlignCenter)
        pix = _load_thumb_pixmap(event, self.THUMB_W, self.THUMB_H)
        if pix.isNull():
            thumb.setText("📷")
            thumb.setStyleSheet(
                "QLabel {"
                "  background: rgba(255,255,255,0.05);"
                "  border: 1px solid rgba(255,255,255,0.1);"
                "  border-radius: 8px; font-size: 30px;"
                "  color: rgba(255,255,255,0.4);"
                "}"
            )
        else:
            thumb.setPixmap(pix)
            thumb.setStyleSheet("background: rgba(0,0,0,0.25); border-radius: 8px;")

        # 종류 태그
        kind  = event.get("kind", "")
        color = "#ea580c" if kind == "plate" else "#1d4ed8"
        label = "번호판" if kind == "plate" else "표지판"
        tag = CaptionLabel(label, self)
        tag.setAlignment(Qt.AlignCenter)
        tag.setFixedSize(68, 30)
        tag.setStyleSheet(
            f"background:{color}; color:#fff; border-radius:6px;"
            f"font-weight:700; font-size:14px;"
        )

        # 프레임 + 시간 (한 줄)
        time_label = BodyLabel(
            f"frame {event.get('frame', 0)}  ·  {_fmt(event.get('time_sec', 0))}",
            self,
        )
        time_label.setStyleSheet(
            "color: rgba(255,255,255,0.75); font-size: 15px; font-weight: 500;"
        )
        time_label.setFixedWidth(180)

        # 메인 텍스트
        text_label = TitleLabel(event.get("label", "-") or "(텍스트 없음)", self)
        text_label.setWordWrap(True)
        text_label.setStyleSheet(
            "color: #f5f5f5;"
            "font-family: 'Noto Sans CJK KR', 'Malgun Gothic', sans-serif;"
            "font-size: 21px; font-weight: 700;"
        )
        text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # 신뢰도 (한 줄)
        conf = float(event.get("confidence") or 0.0)
        conf_label = StrongBodyLabel(f"신뢰도  {conf:.3f}", self)
        conf_label.setStyleSheet(
            f"color: {conf_color(conf)}; font-size: 20px;"
            f"font-weight: 700; font-family: 'Consolas', monospace;"
        )
        conf_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        conf_label.setFixedWidth(160)

        layout.addWidget(thumb)
        layout.addWidget(tag)
        layout.addWidget(time_label)
        layout.addWidget(text_label, 1)
        layout.addWidget(conf_label)


class TimelinePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("timelinePage")
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet(
            "#timelinePage { background-color: #1e1e1e; }"
            "QWidget#timelineInner { background-color: transparent; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 24, 36, 24)
        root.setSpacing(16)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        self.title_label = SubtitleLabel("통합 타임라인", self)
        self.title_label.setStyleSheet("color: #f5f5f5;")
        self.count_label = CaptionLabel("", self)
        self.count_label.setStyleSheet(
            "color: rgba(255,255,255,0.5); font-size: 13px; padding-bottom: 4px;"
        )

        header_row.addWidget(self.title_label)
        header_row.addWidget(self.count_label)
        header_row.addStretch()
        root.addLayout(header_row)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { background: transparent; width: 8px; margin: 4px 0; }"
            "QScrollBar::handle:vertical {"
            "  background: rgba(255,255,255,0.2); border-radius: 4px; min-height: 30px;"
            "}"
            "QScrollBar::handle:vertical:hover { background: rgba(255,255,255,0.35); }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )

        self.inner = QWidget()
        self.inner.setObjectName("timelineInner")
        self.layout_ = QVBoxLayout(self.inner)
        self.layout_.setAlignment(Qt.AlignTop)
        self.layout_.setSpacing(10)
        self.layout_.setContentsMargins(2, 2, 2, 2)

        self.scroll.setWidget(self.inner)
        root.addWidget(self.scroll, 1)

        self._empty_label = BodyLabel("분석 결과가 없습니다.", self.inner)
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            "color: rgba(255,255,255,0.4); font-size:14px; padding:60px;"
        )
        self.layout_.addWidget(self._empty_label)

    def load_timeline(self, timeline: list):
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not timeline:
            self._empty_label = BodyLabel("타임라인 데이터가 없습니다.", self.inner)
            self._empty_label.setAlignment(Qt.AlignCenter)
            self._empty_label.setStyleSheet(
                "color: rgba(255,255,255,0.4); font-size:14px; padding:60px;"
            )
            self.layout_.addWidget(self._empty_label)
            self.count_label.setText("")
            return

        n_plate = sum(1 for e in timeline if e.get("kind") == "plate")
        n_sign  = sum(1 for e in timeline if e.get("kind") == "sign")
        self.count_label.setText(
            f"총 {len(timeline)}건 · 번호판 {n_plate} · 표지판 {n_sign}"
        )

        for event in timeline[:200]:
            self.layout_.addWidget(TimelineEvent(event, self.inner))