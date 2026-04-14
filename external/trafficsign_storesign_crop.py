from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


import sys

# Progress bar (tqdm if available)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepTracer CROP stage: YOLO tracking -> per-track crops")

    # YOLO / tracking
    p.add_argument("--model", required=True, help="YOLO 모델 경로(.pt). OBB는 -obb 모델 권장")
    p.add_argument("--source", required=True, help="입력 비디오 경로/스트림")
    p.add_argument("--tracker", default="botsort.yaml", help="tracker yaml (botsort.yaml/bytetrack.yaml)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0", help="Ultralytics device (0/cpu 등)")

    # output (요청하신 폴더 구조로 고정되게)
    p.add_argument("--out_root", default="runs/cvat_YOLOn_Track", help="출력 루트 폴더")
    p.add_argument("--out_crops", default="", help="out_root 하위 crop 폴더명")
    p.add_argument("--save_meta", action="store_true", help="크롭 메타(jsonl) 저장")

    # OBB warp
    p.add_argument("--use_obb_warp", action="store_true", help="OBB면 4점 warp 크롭 저장")
    p.add_argument("--pad_ratio", type=float, default=0.08, help="AABB fallback padding 비율")

    return p.parse_args()


def log(msg: str) -> None:
    # tqdm progress bar와 함께 써도 줄이 깨지지 않게
    if tqdm is not None:
        try:
            tqdm.write(msg)
            return
        except Exception:
            pass
    print(msg, flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

class _SimpleBar:
    """tqdm가 없을 때 쓰는 아주 가벼운 CLI 진행 막대."""

    def __init__(self, total: Optional[int], desc: str = "", unit: str = "it") -> None:
        self.total = int(total) if total is not None else None
        self.desc = desc
        self.unit = unit
        self.n = 0
        self.t0 = time.time()
        self._last_render = 0.0
        self._render()

    def update(self, inc: int = 1) -> None:
        self.n += int(inc)
        now = time.time()
        if now - self._last_render < 0.05:
            return
        self._render()

    def _render(self, final: bool = False) -> None:
        self._last_render = time.time()
        elapsed = max(self._last_render - self.t0, 1e-6)
        rate = self.n / elapsed
        if self.total:
            frac = min(max(self.n / self.total, 0.0), 1.0)
            bar_w = 28
            filled = int(round(bar_w * frac))
            bar = "█" * filled + "░" * (bar_w - filled)
            pct = int(round(frac * 100))
            msg = f"{self.desc} [{bar}] {pct:3d}%  {self.n}/{self.total} {self.unit}  {rate:5.1f}/{self.unit}/s"
        else:
            spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[self.n % 10]
            msg = f"{self.desc} {spin}  {self.n} {self.unit}  {rate:5.1f}/{self.unit}/s"
        sys.stdout.write("\r" + msg + (" " * 4))
        if final:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def close(self) -> None:
        self._render(final=True)


def _try_get_total_frames(source: str) -> Optional[int]:
    """입력이 로컬 비디오 파일이면 총 프레임 수를 가져와 진행률(%) 표시."""
    try:
        sp = Path(source)
        if not (sp.exists() and sp.is_file()):
            return None
        cap = cv2.VideoCapture(str(sp))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return total if total > 0 else None
    except Exception:
        return None


def _make_pbar(total: Optional[int], desc: str, unit: str) -> Any:
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
    return _SimpleBar(total=total, desc=desc, unit=unit)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """pts: (4,2) corners -> returns ordered [tl, tr, br, bl]"""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def warp_quad_crop(img_bgr: np.ndarray, quad_xy: np.ndarray) -> np.ndarray:
    quad = order_quad_points(quad_xy)
    tl, tr, br, bl = quad

    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)

    width = max(2, int(round(max(w1, w2))))
    height = max(2, int(round(max(h1, h2))))

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img_bgr, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def crop_aabb_with_pad(img_bgr: np.ndarray, xyxy: np.ndarray, pad_ratio: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    pad = pad_ratio * max(bw, bh)
    x1p = int(max(0, math.floor(x1 - pad)))
    y1p = int(max(0, math.floor(y1 - pad)))
    x2p = int(min(w - 1, math.ceil(x2 + pad)))
    y2p = int(min(h - 1, math.ceil(y2 + pad)))
    if x2p <= x1p or y2p <= y1p:
        return img_bgr.copy()
    return img_bgr[y1p:y2p, x1p:x2p].copy()


def run_tracking_and_save_crops(
    model_path: str,
    source: str,
    tracker: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    crops_root: Path,
    use_obb_warp: bool,
    pad_ratio: float,
    save_meta: bool,
) -> Dict[str, Any]:
    if YOLO is None:
        raise RuntimeError("ultralytics가 설치되어 있지 않습니다. `pip install ultralytics` 후 실행하세요.")

    ensure_dir(crops_root)
    meta_path = crops_root / "crops_meta.jsonl"
    meta_f = meta_path.open("w", encoding="utf-8") if save_meta else None

    model = YOLO(model_path)

    results_stream = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker=tracker,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    best_by_id: Dict[int, Tuple[float, Path]] = {}
    frames_processed = 0

    total_frames = _try_get_total_frames(source)
    pbar = _make_pbar(total_frames, desc="CROP", unit="frame")

    t0 = time.time()
    for frame_idx, result in enumerate(results_stream):
        frames_processed += 1
        try:
            pbar.update(1)
        except Exception:
            pass
        img = result.orig_img
        if img is None:
            continue

        obb = getattr(result, "obb", None)

        if obb is not None and getattr(obb, "xyxyxyxy", None) is not None:
            quads = obb.xyxyxyxy
            confs = obb.conf
            clss = obb.cls
            tids = getattr(obb, "id", None)
            if tids is None:
                continue

            quads_np = quads.cpu().numpy() if hasattr(quads, "cpu") else np.asarray(quads)
            confs_np = confs.cpu().numpy() if hasattr(confs, "cpu") else np.asarray(confs)
            clss_np = clss.cpu().numpy() if hasattr(clss, "cpu") else np.asarray(clss)
            tids_np = tids.cpu().numpy() if hasattr(tids, "cpu") else np.asarray(tids)

            for quad_xy, c, cls_id, tid in zip(quads_np, confs_np, clss_np, tids_np):
                if tid is None:
                    continue
                tid_int = int(tid)
                label = result.names[int(cls_id)] if hasattr(result, "names") else str(int(cls_id))
                tid_dir = ensure_dir(crops_root / f"id_{tid_int:05d}")
                crop_path = tid_dir / f"frame_{frame_idx:06d}_{label}_conf{float(c):.2f}.jpg"

                if use_obb_warp:
                    crop = warp_quad_crop(img, quad_xy)
                else:
                    xs, ys = quad_xy[:, 0], quad_xy[:, 1]
                    xyxy = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
                    crop = crop_aabb_with_pad(img, xyxy, pad_ratio)

                cv2.imwrite(str(crop_path), crop)

                area = float(crop.shape[0] * crop.shape[1])
                score = float(c) * area
                prev = best_by_id.get(tid_int)
                if prev is None or score > prev[0]:
                    best_by_id[tid_int] = (score, crop_path)

                if meta_f is not None:
                    meta_f.write(json.dumps({
                        "frame": frame_idx,
                        "track_id": tid_int,
                        "label": label,
                        "conf": float(c),
                        "quad_xy": quad_xy.reshape(-1).tolist(),
                        "crop": str(crop_path),
                        "warp": bool(use_obb_warp),
                    }, ensure_ascii=False) + "\n")

        else:
            boxes = getattr(result, "boxes", None)
            if boxes is None or getattr(boxes, "id", None) is None:
                continue

            xyxy = boxes.xyxy
            confs = boxes.conf
            clss = boxes.cls
            tids = boxes.id

            xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
            confs_np = confs.cpu().numpy() if hasattr(confs, "cpu") else np.asarray(confs)
            clss_np = clss.cpu().numpy() if hasattr(clss, "cpu") else np.asarray(clss)
            tids_np = tids.cpu().numpy() if hasattr(tids, "cpu") else np.asarray(tids)

            for bb, c, cls_id, tid in zip(xyxy_np, confs_np, clss_np, tids_np):
                if tid is None:
                    continue
                tid_int = int(tid)
                label = result.names[int(cls_id)] if hasattr(result, "names") else str(int(cls_id))
                tid_dir = ensure_dir(crops_root / f"id_{tid_int:05d}")
                crop_path = tid_dir / f"frame_{frame_idx:06d}_{label}_conf{float(c):.2f}.jpg"

                crop = crop_aabb_with_pad(img, bb, pad_ratio)
                cv2.imwrite(str(crop_path), crop)

                area = float(crop.shape[0] * crop.shape[1])
                score = float(c) * area
                prev = best_by_id.get(tid_int)
                if prev is None or score > prev[0]:
                    best_by_id[tid_int] = (score, crop_path)

                if meta_f is not None:
                    meta_f.write(json.dumps({
                        "frame": frame_idx,
                        "track_id": tid_int,
                        "label": label,
                        "conf": float(c),
                        "xyxy": [float(v) for v in bb],
                        "crop": str(crop_path),
                        "warp": False,
                    }, ensure_ascii=False) + "\n")

    try:
        pbar.close()
    except Exception:
        pass

    if meta_f is not None:
        meta_f.close()

    # best.jpg 저장
    for tid, (_score, best_path) in best_by_id.items():
        tid_dir = crops_root / f"id_{tid:05d}"
        dst = tid_dir / "best.jpg"
        try:
            im = cv2.imread(str(best_path))
            if im is not None:
                cv2.imwrite(str(dst), im)
        except Exception:
            pass

    dt = time.time() - t0
    log(f"[CROP DONE] frames={frames_processed} tracks~={len(best_by_id)} time={dt:.1f}s crops_root={crops_root}")
    return {
        "crops_root": str(crops_root),
        "meta_path": str(meta_path) if save_meta else "",
        "track_count_est": len(best_by_id),
        "frames_processed": frames_processed,
    }


def main() -> int:
    args = parse_args()

    out_root = ensure_dir(Path(args.out_root))
    crops_root = ensure_dir(out_root / args.out_crops)

    run_tracking_and_save_crops(
        model_path=args.model,
        source=args.source,
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        crops_root=crops_root,
        use_obb_warp=args.use_obb_warp,
        pad_ratio=args.pad_ratio,
        save_meta=args.save_meta,
    )
    log(f"[DONE] out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
