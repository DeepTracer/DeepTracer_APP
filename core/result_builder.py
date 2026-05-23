import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def normalize_plate_item(item: dict, idx: int, fps: float) -> dict:
    frame = int(item.get("frame_index", item.get("frame", 0)))
    text  = item.get("plate_number", item.get("plate", item.get("ocr_result", item.get("text", ""))))
    return {
        "plate_id":       f"plate_{idx+1:05d}",
        "frame":          frame,
        "time_sec":       round(frame / fps, 3) if fps > 0 else 0.0,
        "time_str":       item.get("timestamp", format_time(frame / fps if fps > 0 else 0)),
        "bbox":           item.get("bbox"),
        "text":           text,
        "ocr_confidence": float(item.get("ocr_confidence", 0.0)),
        "det_confidence": float(item.get("det_confidence", 0.0)) if item.get("det_confidence") is not None else None,
        "crop_path":      item.get("crop_path"),
        "image_base64":   item.get("image_base64"),
        "raw":            item,
    }


def normalize_sign_item(item: dict, crops_root):
    frames    = [int(x) for x in (item.get("frames") or [])]
    rep_frame = frames[0] if frames else 0

    best_crop_path = None
    track_id = item.get("track_id", item.get("id"))
    if crops_root and track_id:
        candidate = Path(crops_root) / track_id / "best.jpg"
        if candidate.exists():
            best_crop_path = str(candidate)
    if not best_crop_path:
        best_crop_path = item.get("best_crop_path")

    return {
        "track_id":       track_id,
        "object":         item.get("object", "unknown"),
        "text":           item.get("text", ""),
        "confidence":     float(item.get("main_text_prob", item.get("confidence", 0.0))),
        "consensus":      float(item.get("consensus", item.get("main_text_prob", 0.0))),
        "method":         item.get("method", ""),
        "needs_review":   bool(item.get("needs_review", False)),
        "review_reasons": item.get("needs_review_reasons", []),
        "frames":         frames,
        "rep_frame":      rep_frame,
        "start_frame":    min(frames) if frames else rep_frame,
        "end_frame":      max(frames) if frames else rep_frame,
        "best_crop_path": best_crop_path,
        "raw":            item,
    }


def build_integrated_result(
    video_path: str,
    mode: str,
    fps: float,
    total_frames: int,
    plate_result: Optional[dict] = None,
    sign_result: Optional[dict] = None,
):
    plate_json = (plate_result or {}).get("json_data") or []
    sign_json  = (sign_result  or {}).get("json_data") or []

    plate_items = [normalize_plate_item(item, idx, fps) for idx, item in enumerate(plate_json)]
    sign_items  = [
        normalize_sign_item(item, (sign_result or {}).get("crops_root"))
        for item in sign_json
    ]

    timeline = []
    for p in plate_items:
        timeline.append({
            "frame":        p["frame"],
            "time_sec":     p["time_sec"],
            "kind":         "plate",
            "ref_id":       p["plate_id"],
            "label":        p["text"],
            "confidence":   p["ocr_confidence"],
            "crop_path":    p.get("crop_path"),       # 썸네일
            "image_base64": p.get("image_base64"),    # 썸네일 (base64 fallback)
        })
    for s in sign_items:
        timeline.append({
            "frame":      s["rep_frame"],
            "time_sec":   round(s["rep_frame"] / fps, 3) if fps > 0 else 0.0,
            "kind":       "sign",
            "ref_id":     s["track_id"],
            "label":      s["text"],
            "confidence": s["confidence"],
            "crop_path":  s.get("best_crop_path"),    # 썸네일
        })
    timeline.sort(key=lambda x: (x["frame"], x["kind"]))

    return {
        "meta": {
            "source_video":  os.path.basename(video_path) if video_path else None,
            "analysis_mode": mode,
            "created_at":    datetime.now().astimezone().isoformat(),
            "fps":           fps,
            "total_frames":  total_frames,
            "pipelines": {
                "plate_enabled": mode in ("번호판만", "둘 다"),
                "sign_enabled":  mode in ("표지판/간판만", "둘 다"),
            },
        },
        "summary": {
            "plate_count":           len(plate_items),
            "plate_detected_frames": len(set(p["frame"] for p in plate_items)),
            "sign_track_count":      len(sign_items),
            "sign_detected_frames":  len(set(f for s in sign_items for f in s["frames"])),
            "sign_review_count":     sum(1 for s in sign_items if s["needs_review"]),
        },
        "plate":    {"items": plate_items},
        "sign":     {"items": sign_items},
        "timeline": timeline,
    }