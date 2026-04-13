import base64
import os
import cv2

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CaptionLabel,
    FluentIcon as FIF,
    PushButton,
    Slider,
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


class FrameViewer(QWidget):
    """OpenCV 프레임을 QLabel로 표시하는 위젯."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path = None
        self._cap = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.img_label = QLabel(self)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(640, 360)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_label.setStyleSheet("""
            background: transparent;
            color: #8e8e93;
        """)
        self.img_label.setText("영상 없음")

        self.frame_label = CaptionLabel("프레임: 0", self)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("""
            color: #8e8e93;
            background: transparent;
        """)

        layout.addWidget(self.img_label, 1)
        layout.addWidget(self.frame_label)

    def set_video(self, path: str):
        if self._cap:
            self._cap.release()
        self._video_path = path
        self._cap = cv2.VideoCapture(path) if path and os.path.exists(path) else None

    def show_frame(self, frame_idx: int):
        if not self._cap:
            self.img_label.setText("영상 없음")
            self.img_label.setPixmap(QPixmap())
            return

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qt_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.img_label.width(),
            self.img_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.img_label.setPixmap(pixmap)
        self.img_label.setText("")
        self.frame_label.setText(f"프레임: {frame_idx}")


class PlateCard(CardWidget):
    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            background-color: #1c1c1e;
            border-radius: 12px;
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        text_label = TitleLabel(item.get("text", "-"), self)
        text_label.setStyleSheet("""
            color: #f5f5f7;
            font-family: monospace;
            letter-spacing: 2px;
            background: transparent;
        """)

        ocr_conf = item.get("ocr_confidence", 0.0)
        conf_label = StrongBodyLabel(f"OCR: {ocr_conf:.3f}", self)
        conf_label.setStyleSheet(f"""
            color: {conf_color(ocr_conf)};
            background: transparent;
        """)

        time_label = CaptionLabel(
            f"{item.get('time_str', '')}  (frame {item.get('frame', 0)})", self
        )
        time_label.setStyleSheet("""
            color: #8e8e93;
            background: transparent;
        """)

        crop_label = QLabel(self)
        crop_label.setAlignment(Qt.AlignCenter)
        crop_label.setMinimumHeight(150)
        crop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        crop_label.setStyleSheet("""
            background: transparent;
            color: #888888;
            border: none;
            padding: 0px;
        """)
        self._load_crop(crop_label, item)

        layout.addWidget(text_label)
        layout.addWidget(conf_label)
        layout.addWidget(time_label)
        layout.addWidget(crop_label)

    def _load_crop(self, label: QLabel, item: dict):
        pixmap = None

        if item.get("image_base64"):
            try:
                data = base64.b64decode(item["image_base64"])
                img = QImage.fromData(data)
                pixmap = QPixmap.fromImage(img)
            except Exception:
                pass
        elif item.get("crop_path") and os.path.exists(item["crop_path"]):
            pixmap = QPixmap(item["crop_path"])

        if pixmap and not pixmap.isNull():
            target_width = 360
            label.setPixmap(
                pixmap.scaled(
                    target_width,
                    180,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        else:
            label.setText("이미지 없음")


class SignCard(CardWidget):
    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            background-color: #1c1c1e;
            border-radius: 12px;
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        tag_row = QHBoxLayout()
        track_tag = CaptionLabel(f"  {item.get('track_id', '-')}  ", self)
        track_tag.setStyleSheet(
            "background:#1d4ed8; color:#fff; border-radius:10px; padding:2px 4px;"
        )
        obj_tag = CaptionLabel(f"  {item.get('object', 'unknown')}  ", self)
        obj_tag.setStyleSheet(
            "background:#3f3f46; color:#fff; border-radius:10px; padding:2px 4px;"
        )
        tag_row.addWidget(track_tag)
        tag_row.addWidget(obj_tag)
        tag_row.addStretch()
        layout.addLayout(tag_row)

        text_label = TitleLabel(item.get("text", "-"), self)
        text_label.setStyleSheet("""
            color: #f5f5f7;
            font-family: monospace;
            letter-spacing: 2px;
            background: transparent;
        """)

        conf = item.get("confidence", 0.0)
        conf_label = StrongBodyLabel(f"confidence: {conf:.3f}", self)
        conf_label.setStyleSheet(f"""
            color: {conf_color(conf)};
            background: transparent;
        """)

        review_text = "⚠ 검토 필요" if item.get("needs_review") else "✓ 자동 확정"
        review_label = CaptionLabel(
            f"{review_text}  |  rep_frame: {item.get('rep_frame', 0)}"
            f"  |  frames: {len(item.get('frames', []))}", self
        )
        review_label.setStyleSheet("""
            color: #8e8e93;
            background: transparent;
        """)

        layout.addWidget(text_label)
        layout.addWidget(conf_label)
        layout.addWidget(review_label)

        best = item.get("best_crop_path")
        if best and os.path.exists(best):
            crop_label = QLabel(self)
            crop_label.setAlignment(Qt.AlignCenter)
            crop_label.setMinimumHeight(180)
            crop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            crop_label.setStyleSheet("""
                background: transparent;
                color: #888888;
                border: none;
                padding: 0px;
            """)
            pm = QPixmap(best).scaledToWidth(280, Qt.SmoothTransformation)
            crop_label.setPixmap(pm)
            layout.addWidget(crop_label)


class ResultPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("resultPage")
        self._integrated = None
        self._plate_items = []
        self._sign_items = []
        self._current_frame = 0
        self._total_frames = 1

        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            QWidget#resultPage {
                background-color: #111111;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 24, 36, 24)
        root.setSpacing(16)

        title = SubtitleLabel("분석 결과", self)
        title.setStyleSheet("""
            color: #f5f5f7;
            background: transparent;
        """)
        root.addWidget(title)

        self.summary_row = QHBoxLayout()
        self._summary_cards = {}

        for key in ("번호판 검출", "표지판 Track", "검토 필요", "분석 프레임"):
            card = CardWidget(self)
            card.setStyleSheet("""
                background-color: #1c1c1e;
                border-radius: 12px;
            """)
            inner = QVBoxLayout(card)
            inner.setContentsMargins(14, 14, 14, 14)
            inner.setSpacing(6)

            val_l = TitleLabel("0", card)
            val_l.setStyleSheet("""
                color: #f5f5f7;
                background: transparent;
            """)
            key_l = CaptionLabel(key, card)
            key_l.setStyleSheet("""
                color: #8e8e93;
                background: transparent;
            """)

            inner.addWidget(val_l, alignment=Qt.AlignCenter)
            inner.addWidget(key_l, alignment=Qt.AlignCenter)

            self._summary_cards[key] = val_l
            self.summary_row.addWidget(card)

        root.addLayout(self.summary_row)

        split = QHBoxLayout()
        split.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)

        self.frame_viewer = FrameViewer(self)

        self.slider = Slider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider)

        nav_row = QHBoxLayout()
        self.prev_btn = PushButton(FIF.LEFT_ARROW, "", self)
        self.next_btn = PushButton(FIF.RIGHT_ARROW, "", self)
        self.prev_btn.setFixedWidth(60)
        self.next_btn.setFixedWidth(60)
        self.prev_btn.clicked.connect(lambda: self._step_frame(-5))
        self.next_btn.clicked.connect(lambda: self._step_frame(5))

        nav_row.addWidget(self.prev_btn)
        nav_row.addStretch()
        nav_row.addWidget(self.next_btn)

        left.addWidget(self.frame_viewer, 1)
        left.addWidget(self.slider)
        left.addLayout(nav_row)

        right = QVBoxLayout()
        self.result_scroll = QScrollArea(self)
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #111111;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #111111;
            }
        """)

        self.result_inner = QWidget()
        self.result_inner.setStyleSheet("background-color: #111111;")

        self.result_layout = QVBoxLayout(self.result_inner)
        self.result_layout.setAlignment(Qt.AlignTop)
        self.result_layout.setSpacing(12)

        self.result_scroll.setWidget(self.result_inner)
        right.addWidget(self.result_scroll)

        split.addLayout(left, 55)
        split.addLayout(right, 45)
        root.addLayout(split)

    def load_result(self, integrated: dict, plate_video_path: str):
        self._integrated = integrated
        self._plate_items = integrated.get("plate", {}).get("items", [])
        self._sign_items = integrated.get("sign", {}).get("items", [])
        self._total_frames = integrated.get("meta", {}).get("total_frames", 1)

        s = integrated.get("summary", {})
        self._summary_cards["번호판 검출"].setText(str(s.get("plate_count", 0)))
        self._summary_cards["표지판 Track"].setText(str(s.get("sign_track_count", 0)))
        self._summary_cards["검토 필요"].setText(str(s.get("sign_review_count", 0)))
        self._summary_cards["분석 프레임"].setText(str(self._total_frames))

        self.slider.setMaximum(max(1, self._total_frames - 1))
        self.slider.setValue(0)

        video = plate_video_path if plate_video_path and os.path.exists(plate_video_path) else None
        self.frame_viewer.set_video(video or "")
        self.frame_viewer.show_frame(0)
        self._refresh_cards(0)

    def _on_slider(self, value: int):
        self._current_frame = value
        self.frame_viewer.show_frame(value)
        self._refresh_cards(value)

    def _step_frame(self, delta: int):
        new_val = max(0, min(self._total_frames - 1, self._current_frame + delta))
        self.slider.setValue(new_val)

    def _refresh_cards(self, frame_idx: int):
        while self.result_layout.count():
            item = self.result_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        RANGE = 5
        plates = [p for p in self._plate_items if abs(p.get("frame", 0) - frame_idx) <= RANGE]
        signs = [
            s for s in self._sign_items
            if any(abs(int(f) - frame_idx) <= RANGE for f in s.get("frames", []))
        ]

        if plates:
            header = StrongBodyLabel("번호판", self.result_inner)
            header.setStyleSheet("""
                color: #f5f5f7;
                background: transparent;
            """)
            self.result_layout.addWidget(header)
            for p in plates:
                self.result_layout.addWidget(PlateCard(p, self.result_inner))

        if signs:
            header = StrongBodyLabel("표지판/간판", self.result_inner)
            header.setStyleSheet("""
                color: #f5f5f7;
                background: transparent;
            """)
            self.result_layout.addWidget(header)
            for s in signs:
                self.result_layout.addWidget(SignCard(s, self.result_inner))

        if not plates and not signs:
            no_label = BodyLabel("현재 프레임 근처 결과 없음", self.result_inner)
            no_label.setAlignment(Qt.AlignCenter)
            no_label.setStyleSheet("""
                color: #8e8e93;
                background: transparent;
            """)
            self.result_layout.addWidget(no_label)