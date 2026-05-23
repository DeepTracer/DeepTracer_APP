import gc
import json
import os
import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal as Signal

from core.config import get_paths


def load_jsonl(path: str) -> list:
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class AnalysisWorker(QThread):
    progress = Signal(str, int)
    finished = Signal(dict, dict)
    error = Signal(str)

    def __init__(self, video_path: str, output_dir: str, mode: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.mode = mode

    def run(self):
        plate_res = None
        sign_res = None

        try:
            if self.mode in ("번호판만", "둘 다"):
                plate_out = os.path.join(self.output_dir, "plate")
                os.makedirs(plate_out, exist_ok=True)
                plate_res = self._run_plate(plate_out)
                if not plate_res.get("success"):
                    self.error.emit(f"번호판 분석 실패: {plate_res.get('error')}")
                    return

            self._cleanup_memory()

            if self.mode in ("표지판/간판만", "둘 다"):
                sign_out = os.path.join(self.output_dir, "sign")
                os.makedirs(sign_out, exist_ok=True)
                sign_res = self._run_sign(sign_out)
                if not sign_res.get("success"):
                    self.error.emit(f"표지판 분석 실패: {sign_res.get('error')}")
                    return

            self.finished.emit(plate_res or {}, sign_res or {})

        except Exception as e:
            self.error.emit(str(e))

    def _run_plate(self, output_dir: str) -> dict:
        paths = get_paths()
        plate_inference_script = Path(paths["plate_inference_script"])

        if not plate_inference_script.exists():
            return {"success": False, "error": f"inference_module.py not found: {plate_inference_script}"}

        file_name = os.path.splitext(os.path.basename(self.video_path))[0]
        result_json = os.path.join(output_dir, f"{file_name}_result.json")
        result_video = os.path.join(output_dir, f"{file_name}_result.mp4")

        print("[DEBUG][WORKER] plate_inference_script =", plate_inference_script, flush=True)
        print("[DEBUG][WORKER] plate_ocr_model =", paths["plate_ocr_model"], flush=True)
        print("[DEBUG][WORKER] plate_yolo_model =", paths["plate_yolo_model"], flush=True)
        print("[DEBUG][WORKER] plate_ocr_model exists =", os.path.exists(paths["plate_ocr_model"]), flush=True)
        print("[DEBUG][WORKER] plate_yolo_model exists =", os.path.exists(paths["plate_yolo_model"]), flush=True)
        self.progress.emit("번호판 분석 중...", 5)

        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            cap.release()
        except Exception:
            total_frames = 1

        env = os.environ.copy()
        env["PLATE_YOLO_MODEL"] = os.path.abspath(paths["plate_yolo_model"])
        env["PLATE_OCR_MODEL"] = os.path.abspath(paths["plate_ocr_model"])

        proc = subprocess.Popen(
            [sys.executable, str(plate_inference_script), self.video_path, output_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )

        last_line = ""
        for line in proc.stdout:
            last_line = line.strip()
            print("[PLATE_SUBPROCESS]", last_line, flush=True)

            if "진행 중..." in line and "/" in line:
                try:
                    after = line.split("진행 중...")[-1].strip()
                    cur = int(after.split("/")[0].strip())
                    pct = int(cur / total_frames * 45)
                    self.progress.emit(f"번호판 분석 중: {cur}/{total_frames}", pct)
                except Exception:
                    pass

        proc.wait()

        if os.path.exists(result_json):
            with open(result_json, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            self.progress.emit("번호판 분석 완료", 50)
            return {
                "success": True,
                "result_video": result_video,
                "result_json": result_json,
                "json_data": json_data,
            }

        return {"success": False, "error": last_line or "result json not found"}

    def _run_sign(self, output_dir: str) -> dict:
        paths = get_paths()

        sign_crop_script = Path(paths["sign_crop_script"])
        sign_ocr_script = Path(paths["sign_ocr_script"])
        sign_tracker_yaml = Path(paths["sign_tracker_yaml"])
        sign_model = Path(paths["sign_model"])

        for label, path in [
            ("crop script", sign_crop_script),
            ("ocr script", sign_ocr_script),
            ("tracker yaml", sign_tracker_yaml),
            ("sign model", sign_model),
        ]:
            if not path.exists():
                return {"success": False, "error": f"{label} not found: {path}"}

        print("[DEBUG][WORKER] sign_crop_script =", sign_crop_script, flush=True)
        print("[DEBUG][WORKER] sign_ocr_script  =", sign_ocr_script, flush=True)
        print("[DEBUG][WORKER] sign_model       =", sign_model, flush=True)
        print("[DEBUG][WORKER] sign_tracker     =", sign_tracker_yaml, flush=True)

        crop_out_root = os.path.join(output_dir, "sign_crop")
        crop_folder_name = "track_id_crops"
        crops_root = os.path.join(crop_out_root, crop_folder_name)
        ocr_out_root = os.path.join(output_dir, "sign_ocr")
        ocr_jsonl = os.path.join(ocr_out_root, "ocr_tracks.jsonl")

        self.progress.emit("표지판/간판 추적 + crop 추출 중...", 55)

        # ─── 크롭 단계 (새 모델 기본값 반영) ──────────────────────
        crop_proc = subprocess.Popen(
            [
                sys.executable, str(sign_crop_script),
                "--model", str(sign_model),
                "--source", self.video_path,
                "--tracker", str(sign_tracker_yaml),
                "--imgsz", "960",                  # 640 → 960
                "--conf", "0.8",                   # 0.5 → 0.8
                "--iou", "0.5",
                "--pad_ratio", "0.18",             # 추가
                "--device", "0",
                "--out_root", crop_out_root,
                "--out_crops", crop_folder_name,
                "--save_meta",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        last_crop_line = ""
        for line in crop_proc.stdout:
            last_crop_line = line.strip()
            print("[SIGN_CROP]", last_crop_line, flush=True)

        crop_proc.wait()
        if crop_proc.returncode != 0:
            return {"success": False, "error": f"crop failed: {last_crop_line}"}

        self.progress.emit("표지판/간판 crop 완료", 75)

        if not any(Path(crops_root).glob("id_*")):
            return {
                "success": True,
                "result_json": None,
                "json_data": [],
                "output_dir": output_dir,
                "crops_root": crops_root,
                "warning": "검출된 표지판/간판 없음",
            }

        self.progress.emit("표지판/간판 OCR fusion 중...", 80)

        # ─── OCR 단계 (다중 변형 + 도로 용어 후처리 활용) ────────
        ocr_proc = subprocess.Popen(
            [
                sys.executable, str(sign_ocr_script),
                "--input", crops_root,
                "--out", ocr_out_root,
                "--lang", "korean",
                "--max_frames_per_track", "80",
                "--min_text_len", "2",
                "--review_consensus_thresh", "0.58",
                "--ocr_variants", "original,enhanced,gray",   # 추가
                "--use_textline_orientation",                  # 추가
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        last_ocr_line = ""
        for line in ocr_proc.stdout:
            last_ocr_line = line.strip()
            print("[SIGN_OCR]", last_ocr_line, flush=True)

        ocr_proc.wait()
        if ocr_proc.returncode != 0:
            return {"success": False, "error": f"ocr failed: {last_ocr_line}"}

        if not os.path.exists(ocr_jsonl):
            return {"success": False, "error": "ocr_tracks.jsonl not found"}

        self.progress.emit("표지판/간판 분석 완료", 95)
        return {
            "success": True,
            "result_json": ocr_jsonl,
            "json_data": load_jsonl(ocr_jsonl),
            "output_dir": output_dir,
            "crops_root": crops_root,
        }

    @staticmethod
    def _cleanup_memory():
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass