import json
import os
import tempfile

import cv2
from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel,
    QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CardWidget,
    ComboBox,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    ProgressBar,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
    SubtitleLabel,
    TitleLabel,
)

from core.worker         import AnalysisWorker
from core.result_builder import build_integrated_result


class HomePage(QWidget):
    # 분석 완료 시 (integrated_result, plate_video_path) 방출
    analysis_finished = Signal(dict, str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("homePage")

        self._video_path  = None
        self._output_dir  = None
        self._worker      = None
        self._plate_res   = None
        self._sign_res    = None

        self._init_ui()

    # ─────────────────────────────────────────────
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 36, 36, 36)
        root.setSpacing(20)
        root.setAlignment(Qt.AlignTop)

        # 타이틀
        title = TitleLabel("DeepTracer", self)
        sub   = SubtitleLabel("블랙박스 영상 통합 분석 서비스", self)
        root.addWidget(title)
        root.addWidget(sub)

        # 업로드 카드
        upload_card = CardWidget(self)
        upload_layout = QVBoxLayout(upload_card)
        upload_layout.setSpacing(12)

        self.file_label = StrongBodyLabel("영상 파일을 선택해주세요", upload_card)
        self.file_label.setAlignment(Qt.AlignCenter)

        self.upload_btn = PushButton(FIF.FOLDER, "영상 파일 열기", upload_card)
        self.upload_btn.setFixedWidth(200)
        self.upload_btn.clicked.connect(self._open_file)

        upload_layout.addWidget(self.file_label, alignment=Qt.AlignCenter)
        upload_layout.addWidget(self.upload_btn, alignment=Qt.AlignCenter)
        root.addWidget(upload_card)

        # 모드 + 분석 카드
        ctrl_card = CardWidget(self)
        ctrl_layout = QHBoxLayout(ctrl_card)
        ctrl_layout.setSpacing(16)

        mode_label = StrongBodyLabel("분석 모드:", ctrl_card)
        self.mode_combo = ComboBox(ctrl_card)
        self.mode_combo.addItems(["번호판만", "표지판/간판만", "둘 다"])
        self.mode_combo.setFixedWidth(160)

        self.start_btn = PrimaryPushButton(FIF.PLAY, "분석 시작", ctrl_card)
        self.start_btn.setFixedWidth(160)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_analysis)

        ctrl_layout.addWidget(mode_label)
        ctrl_layout.addWidget(self.mode_combo)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.start_btn)
        root.addWidget(ctrl_card)

        # 진행 카드
        prog_card = CardWidget(self)
        prog_layout = QVBoxLayout(prog_card)

        self.progress_label = QLabel("대기 중", prog_card)
        self.progress_label.setStyleSheet("color: #aaaaaa; font-size: 13px;")
        self.progress_bar = ProgressBar(prog_card)
        self.progress_bar.setValue(0)

        prog_layout.addWidget(self.progress_label)
        prog_layout.addWidget(self.progress_bar)
        root.addWidget(prog_card)

        # 결과 저장 버튼
        self.save_btn = PushButton(FIF.SAVE, "통합 JSON 저장", self)
        self.save_btn.setFixedWidth(200)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_json)
        root.addWidget(self.save_btn, alignment=Qt.AlignRight)

        root.addStretch()

    # ─────────────────────────────────────────────
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "영상 파일 선택", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )
        if not path:
            return
        self._video_path = path
        self.file_label.setText(os.path.basename(path))
        self.start_btn.setEnabled(True)

    # ─────────────────────────────────────────────
    def _start_analysis(self):
        if not self._video_path:
            return

        self._output_dir = tempfile.mkdtemp(prefix="deeptracer_")
        mode = self.mode_combo.currentText()

        self.start_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.save_btn.setEnabled(False)

        self._worker = AnalysisWorker(self._video_path, self._output_dir, mode, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ─────────────────────────────────────────────
    def _on_progress(self, msg: str, pct: int):
        self.progress_label.setText(msg)
        self.progress_bar.setValue(pct)

    def _on_finished(self, plate_res: dict, sign_res: dict):
        self._plate_res = plate_res
        self._sign_res  = sign_res

        self.progress_bar.setValue(100)
        self.progress_label.setText("분석 완료!")
        self.start_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        # 비디오 정보
        fps, total_frames = self._get_video_info()

        integrated = build_integrated_result(
            self._video_path,
            self.mode_combo.currentText(),
            fps, total_frames,
            plate_result=plate_res if plate_res.get("success") else None,
            sign_result=sign_res   if sign_res.get("success")  else None,
        )
        self._integrated = integrated

        plate_video = plate_res.get("result_video", "") if plate_res else ""
        self.analysis_finished.emit(integrated, plate_video or "")

        InfoBar.success(
            title="분석 완료",
            content=f"번호판 {integrated['summary']['plate_count']}건 / "
                    f"표지판·간판 {integrated['summary']['sign_track_count']}개 검출",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=4000,
            parent=self,
        )

        if sign_res and sign_res.get("warning"):
            InfoBar.warning(
                title="표지판/간판",
                content=sign_res["warning"],
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=4000,
                parent=self,
            )

    def _on_error(self, msg: str):
        self.progress_label.setText("오류 발생")
        self.start_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)

        InfoBar.error(
            title="분석 오류",
            content=msg,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=6000,
            parent=self,
        )

    # ─────────────────────────────────────────────
    def _save_json(self):
        if not hasattr(self, "_integrated"):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "통합 JSON 저장", "integrated_result.json", "JSON (*.json)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._integrated, f, ensure_ascii=False, indent=2)
        InfoBar.success(
            title="저장 완료", content=path,
            orient=Qt.Horizontal, isClosable=True,
            position=InfoBarPosition.TOP_RIGHT, duration=3000, parent=self,
        )

    # ─────────────────────────────────────────────
    def _get_video_info(self):
        try:
            cap = cv2.VideoCapture(self._video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()
            return fps, total
        except Exception:
            return 30.0, 0