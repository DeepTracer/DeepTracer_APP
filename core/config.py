import json
import sys
from pathlib import Path


def get_base_dir() -> Path:
    """
    개발 환경: 프로젝트 루트
    배포(exe) 환경: exe가 있는 폴더
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def get_settings_path() -> Path:
    return Path.home() / ".deeptracer_settings.json"


def load_settings() -> dict:
    path = get_settings_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_settings(data: dict):
    path = get_settings_path()
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def get_default_paths() -> dict:
    base = get_base_dir()

    return {
        # scripts
        "plate_inference_script": str(base / "external" / "inference_module.py"),
        "sign_crop_script": str(base / "external" / "trafficsign_storesign_crop.py"),
        "sign_ocr_script": str(base / "external" / "trafficsign_storesign_ocr.py"),

        # models
        "plate_ocr_model": str(base / "models" / "plate" / "plateocr.pth"),
        "plate_yolo_model": str(base / "models" / "plate" / "plateyolo.pt"),
        "sign_model": str(base / "models" / "sign" / "trafficsign.pt"),
        "sign_tracker_yaml": str(base / "models" / "sign" / "botsort.yaml"),

        # resources
        "icon_path": str(base / "resources" / "icon.png"),
    }


def get_paths() -> dict:
    defaults = get_default_paths()
    saved = load_settings()

    merged = defaults.copy()
    for key, value in saved.items():
        if value:
            merged[key] = value

    return merged