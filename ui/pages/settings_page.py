from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QFormLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CardWidget,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    SubtitleLabel,
)

from core.config import load_settings, save_settings, get_default_paths


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("settingsPage")
        self._settings = load_settings()
        self._defaults = get_default_paths()
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 24, 36, 24)
        root.setSpacing(20)
        root.setAlignment(Qt.AlignTop)

        root.addWidget(SubtitleLabel("설정", self))

        model_card = CardWidget(self)
        form = QFormLayout(model_card)
        form.setSpacing(12)
        form.setContentsMargins(20, 20, 20, 20)

        self._plate_ocr_model_edit = self._make_path_edit(
            "plate_ocr_model",
            self._defaults["plate_ocr_model"],
        )
        self._plate_yolo_model_edit = self._make_path_edit(
            "plate_yolo_model",
            self._defaults["plate_yolo_model"],
        )
        self._sign_model_edit = self._make_path_edit(
            "sign_model",
            self._defaults["sign_model"],
        )
        self._sign_tracker_yaml_edit = self._make_path_edit(
            "sign_tracker_yaml",
            self._defaults["sign_tracker_yaml"],
            file_filter="YAML Files (*.yaml *.yml)"
        )

        form.addRow("번호판 OCR 모델 (.pth)", self._plate_ocr_model_edit)
        form.addRow("번호판 YOLO 모델 (.pt)", self._plate_yolo_model_edit)
        form.addRow("표지판 YOLO 모델 (.pt)", self._sign_model_edit)
        form.addRow("표지판 Tracker 설정 (.yaml)", self._sign_tracker_yaml_edit)

        root.addWidget(model_card)

        save_btn = PrimaryPushButton(FIF.SAVE, "설정 저장", self)
        save_btn.setFixedWidth(160)
        save_btn.clicked.connect(self._save)
        root.addWidget(save_btn, alignment=Qt.AlignRight)

        root.addStretch()

    def _make_path_edit(self, key: str, default: str, file_filter="Model Files (*.pt *.pth *.yaml *.yml)") -> QWidget:
        container = QWidget(self)
        row = QVBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        edit = LineEdit(container)
        edit.setText(self._settings.get(key, default))
        edit.setPlaceholderText(default)

        browse_btn = PushButton(FIF.FOLDER, "찾아보기", container)
        browse_btn.setFixedWidth(110)
        browse_btn.clicked.connect(lambda: self._browse(edit, file_filter))

        row.addWidget(edit)
        row.addWidget(browse_btn, alignment=Qt.AlignRight)

        setattr(self, f"_{key}_edit_ref", edit)
        return container

    def _browse(self, edit: LineEdit, file_filter: str):
        path, _ = QFileDialog.getOpenFileName(
            self, "파일 선택", "", file_filter
        )
        if path:
            edit.setText(path)

    def _save(self):
        self._settings["plate_ocr_model"] = self._plate_ocr_model_edit_ref.text()
        self._settings["plate_yolo_model"] = self._plate_yolo_model_edit_ref.text()
        self._settings["sign_model"] = self._sign_model_edit_ref.text()
        self._settings["sign_tracker_yaml"] = self._sign_tracker_yaml_edit_ref.text()

        save_settings(self._settings)

        InfoBar.success(
            title="저장 완료",
            content="설정이 저장되었습니다.",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=3000,
            parent=self,
        )