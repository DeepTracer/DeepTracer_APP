from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


import sys

# Progress bar (tqdm if available)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None
# PaddleOCR
try:
    import paddle
    from paddleocr import PaddleOCR
except Exception:
    paddle = None
    PaddleOCR = None

# SR 단계는 제거되었습니다. (Real-ESRGAN 관련 코드/의존성 제거)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepTracer OCR stage: PaddleOCR + temporal fusion (+ optional SR)")

    p.add_argument("--input", default="runs/cvat_YOLOn_Track/track_id_crops", help="crop 입력 폴더")
    p.add_argument("--out", default="runs/cvat_YOLOn_TrackOCR", help="OCR 출력 폴더")

    # OCR / fusion knobs
    p.add_argument("--max_frames_per_track", type=int, default=80, help="트랙당 최대 OCR 프레임(긴 트랙은 샘플링)")
    p.add_argument("--min_text_len", type=int, default=2, help="짧은 텍스트 가중치 다운(완전 제외는 아님)")
    p.add_argument("--rover_char_thresh", type=float, default=0.52)
    p.add_argument("--rover_ins_thresh", type=float, default=0.62)
    p.add_argument("--review_consensus_thresh", type=float, default=0.58)

    # PaddleOCR options
    p.add_argument("--lang", default="korean")
    p.add_argument("--rec_model_dir", default="", help="(선택) PaddleX 형식 rec 모델 폴더(inference.yml 포함 권장)")
    p.add_argument("--det_model_dir", default="", help="(선택) PaddleX 형식 det 모델 폴더")
    p.add_argument("--cls_model_dir", default="", help="(선택) PaddleX 형식 textline orientation 모델 폴더")
    p.add_argument("--use_angle_cls", action="store_true", help="(deprecated) -> 내부에서 use_textline_orientation로 매핑")
    p.add_argument("--use_textline_orientation", action="store_true", help="텍스트 라인 방향 분류 사용")

    # 문서용 옵션(간판/표지판에서는 역효과 가능 → 기본 OFF)
    p.add_argument("--use_doc_orientation", action="store_true")
    p.add_argument("--use_doc_unwarping", action="store_true")

    # PaddleOCR 3.x: 결과 필터 임계 (drop_score 대신 이걸 씀)
    p.add_argument("--text_rec_score_thresh", type=float, default=0.0, help="텍스트 인식 결과 필터 임계(0이면 거의 필터 X)")

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


def _progress_iter(items: List[Any], desc: str, unit: str):
    """tqdm가 있으면 tqdm로, 없으면 간단한 막대로 진행률 표시."""
    if tqdm is not None:
        for x in tqdm(items, desc=desc, unit=unit, dynamic_ncols=True):
            yield x
        return
    bar = _SimpleBar(total=len(items), desc=desc, unit=unit)
    for x in items:
        yield x
        bar.update(1)
    bar.close()


# ----------------------------
# Frame parsing + normalization
# ----------------------------
FRAME_RE = re.compile(r"frame_(\d+)")


def parse_frame_no(filename: str, fallback: int) -> int:
    m = FRAME_RE.search(filename)
    return int(m.group(1)) if m else fallback


def norm_text(s: str) -> str:
    return "".join((s or "").split())


def good_text(s: Any) -> bool:
    return isinstance(s, str) and s.strip() != ""


def image_quality_score(img_bgr: np.ndarray) -> float:
    if img_bgr is None or img_bgr.size == 0:
        return 0.1
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp = np.sqrt(max(sharp, 0.0)) / 20.0
    mean = float(gray.mean()) / 255.0
    bright = 0.2 + 0.8 * min(max((mean - 0.08) / 0.35, 0.0), 1.0)
    return float(max(0.1, min(3.0, sharp * bright)))


# ----------------------------
# Final sanity filter
# ----------------------------
_L_BASE = 0x1100
_V_BASE = 0x1161
_T_BASE = 0x11A7
_S_BASE = 0xAC00
_L_COUNT = 19
_V_COUNT = 21
_T_COUNT = 28
_N_COUNT = _V_COUNT * _T_COUNT
_S_COUNT = _L_COUNT * _N_COUNT


def _is_hangul_syllable(ch: str) -> bool:
    o = ord(ch)
    return _S_BASE <= o < (_S_BASE + _S_COUNT)


def _is_hangul_jamo_like(ch: str) -> bool:
    o = ord(ch)
    return (
        0x1100 <= o <= 0x11FF or
        0x3130 <= o <= 0x318F or
        0xA960 <= o <= 0xA97F or
        0xD7B0 <= o <= 0xD7FF or
        0xFFA0 <= o <= 0xFFDC
    )


def is_valid_final_text(text: str) -> bool:
    """
    Exclude:
      - 공백/빈 문자열
      - 불완전한 한글(자모 포함)  ex) '안ㅛ', 'ㄷㅗㅇ'
      - 숫자 1글자만
      - 특수문자 1글자만
    """
    t = norm_text(text)
    if not t:
        return False
    if any(_is_hangul_jamo_like(ch) for ch in t):
        return False
    if len(t) == 1:
        ch = t[0]
        if ch.isdigit():
            return False
        if (not ch.isalnum()) and (not _is_hangul_syllable(ch)):
            return False
    return True


# ----------------------------
# Hangul-aware distance
# ----------------------------
def hangul_to_jamo(s: str) -> str:
    out: List[str] = []
    for ch in s:
        if _is_hangul_syllable(ch):
            s_index = ord(ch) - _S_BASE
            l = s_index // _N_COUNT
            v = (s_index % _N_COUNT) // _T_COUNT
            t = s_index % _T_COUNT
            out.append(chr(_L_BASE + l))
            out.append(chr(_V_BASE + v))
            if t != 0:
                out.append(chr(_T_BASE + t))
        else:
            out.append(ch)
    return "".join(out)


def lev(a: str, b: str) -> int:
    a = a or ""
    b = b or ""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * m
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def norm_dist_hangul(a: str, b: str) -> float:
    a2 = hangul_to_jamo(norm_text(a))
    b2 = hangul_to_jamo(norm_text(b))
    d = lev(a2, b2)
    denom = max(len(a2), len(b2), 1)
    return float(d / denom)


# ----------------------------
# Baseline subset-merge voting
# ----------------------------
def is_subset(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a in b) and (a != b)


def group_mean_prob(rec: Dict[str, Any]) -> float:
    m = rec["frame_to_best_prob"]
    if not m:
        return 0.0
    return float(sum(m.values()) / len(m))


def merge_into(dst: str, src: str, groups: Dict[str, Dict[str, Any]]) -> None:
    drec = groups[dst]
    srec = groups[src]
    for fr, pr in srec["frame_to_best_prob"].items():
        prev = drec["frame_to_best_prob"].get(fr, -1.0)
        if pr > prev:
            drec["frame_to_best_prob"][fr] = pr


def default_subset_vote(frame_items: List[Tuple[int, str, float]]) -> Tuple[str, float, int]:
    groups: Dict[str, Dict[str, Any]] = {}
    for fr, txt, pr in frame_items:
        if not good_text(txt):
            continue
        n = norm_text(txt)
        if n == "":
            continue
        if n not in groups:
            groups[n] = {"frame_to_best_prob": {}}
        prev = groups[n]["frame_to_best_prob"].get(fr, -1.0)
        if pr > prev:
            groups[n]["frame_to_best_prob"][fr] = float(pr)

    keys_sorted = sorted(groups.keys(), key=lambda x: len(x))
    merged = set()
    for i, a in enumerate(keys_sorted):
        if a in merged:
            continue
        for b in keys_sorted[i + 1:]:
            if b in merged:
                continue
            if is_subset(a, b):
                merge_into(b, a, groups)
                merged.add(a)
                break

    best = ""
    best_mp = -1.0
    best_fc = -1
    for k, rec in groups.items():
        fc = len(rec["frame_to_best_prob"])
        if fc <= 0:
            continue
        mp = group_mean_prob(rec)
        if (fc, mp, len(k)) > (best_fc, best_mp, len(best)):
            best, best_fc, best_mp = k, fc, mp

    if best == "":
        return "", 0.0, 0
    return best, float(best_mp), int(best_fc)


# ----------------------------
# ROVER-like char voting
# ----------------------------
_EPS = "<eps>"


def _align_ops(pivot: str, hyp: str) -> List[Tuple[str, int, int]]:
    a = list(pivot)
    b = list(hyp)
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)

    ops_rev: List[Tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost:
                ops_rev.append(("M" if cost == 0 else "S", i, j))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops_rev.append(("D", i, j))
            i -= 1
            continue
        ops_rev.append(("I", i, j))
        j -= 1

    ops_rev.reverse()
    return ops_rev


def rover_char_vote(hyps: List[Tuple[str, float]], char_thresh: float, ins_thresh: float) -> Tuple[str, float]:
    hyps2 = [(norm_text(t), float(w)) for t, w in hyps if good_text(t) and float(w) > 0]
    if not hyps2:
        return "", 0.0

    hyps2.sort(key=lambda x: x[1], reverse=True)
    top = hyps2[: min(12, len(hyps2))]

    def medoid_score(idx: int) -> float:
        t_i = top[idx][0]
        s = 0.0
        for t_j, w_j in top:
            s += w_j * norm_dist_hangul(t_i, t_j)
        return s

    pivot = min(range(len(top)), key=medoid_score)
    pivot_text, pivot_w = top[pivot]
    pivot_chars = list(pivot_text)
    n = len(pivot_chars)

    ins_slots: List[Dict[str, float]] = [defaultdict(float) for _ in range(n + 1)]
    char_slots: List[Dict[str, float]] = [defaultdict(float) for _ in range(n)]

    total_w = pivot_w
    for i, ch in enumerate(pivot_chars):
        char_slots[i][ch] += pivot_w

    for t, w in hyps2:
        total_w += w
        if t == pivot_text:
            continue
        ops = _align_ops(pivot_text, t)
        for op, i, j in ops:
            if op in ("M", "S"):
                char_slots[i - 1][t[j - 1]] += w
            elif op == "D":
                char_slots[i - 1][_EPS] += w
            else:
                ins_slots[i][t[j - 1]] += w

    for i in range(n + 1):
        s = float(sum(ins_slots[i].values()))
        if s < total_w:
            ins_slots[i][_EPS] += (total_w - s)

    out_chars: List[str] = []
    confs: List[float] = []
    for i in range(n + 1):
        slot = ins_slots[i]
        total = float(sum(slot.values())) or 1.0
        best_ch, best_sc = max(slot.items(), key=lambda kv: kv[1])
        if best_ch != _EPS and (best_sc / total) >= ins_thresh:
            out_chars.append(best_ch)
            confs.append(float(best_sc / total))

        if i == n:
            break

        cslot = char_slots[i]
        total2 = float(sum(cslot.values())) or 1.0
        best2, sc2 = max(cslot.items(), key=lambda kv: kv[1])
        if best2 != _EPS and (sc2 / total2) >= char_thresh:
            out_chars.append(best2)
            confs.append(float(sc2 / total2))

    consensus = "".join(out_chars)
    if not consensus:
        return "", 0.0
    return consensus, float(sum(confs) / max(len(confs), 1))


# ----------------------------
# Candidate selection
# ----------------------------
def score_candidate(candidate: str, frame_hyps: List[Tuple[str, float]], min_text_len: int) -> float:
    c = norm_text(candidate)
    if not c:
        return 10.0
    dsum = 0.0
    wsum = 0.0
    for t, w in frame_hyps:
        t2 = norm_text(t)
        if not t2:
            continue
        d = norm_dist_hangul(c, t2)
        dsum += w * d
        wsum += w
    if wsum <= 1e-9:
        return 10.0
    base = dsum / wsum
    lp = 0.0
    if len(c) >= 18:
        lp += 0.05 * (len(c) - 17)
    # 너무 짧으면 살짝 페널티(하지만 완전 제외 X)
    if len(c) < min_text_len:
        lp += 0.15
    return float(base + lp)


def pick_best_frame(frame_items: List[Tuple[int, str, float, float]]) -> Tuple[str, float]:
    best_t = ""
    best_w = -1.0
    for _, t, pr, q in frame_items:
        if not good_text(t):
            continue
        w = float(pr) * float(q)
        if w > best_w:
            best_t, best_w = norm_text(t), w
    return best_t, float(max(best_w, 0.0))


def adaptive_fusion(
    obj: str,
    frame_items: List[Tuple[int, str, float, float]],
    rover_char_thresh: float,
    rover_ins_thresh: float,
    min_text_len: int,
) -> Tuple[str, float, str, Dict[str, Any]]:
    hyps: List[Tuple[str, float]] = []
    for _, t, pr, q in frame_items:
        if not good_text(t):
            continue
        nt = norm_text(t)
        if not nt:
            continue
        w = float(pr) * float(q)
        if len(nt) < min_text_len:
            w *= 0.35
        hyps.append((nt, w))
    if not hyps:
        return "", 0.0, "empty", {"reason": "no_valid_text"}

    baseline_items = [(fr, t, pr) for fr, t, pr, _q in frame_items if good_text(t)]
    base_text, base_mp, base_fc = default_subset_vote(baseline_items)

    ct = rover_char_thresh
    it = rover_ins_thresh
    if obj in ("trafficsign",):
        ct = max(ct, 0.58)
        it = max(it, 0.70)
    elif obj in ("licenseplate", "platesign"):
        ct = max(ct, 0.60)
        it = max(it, 0.72)

    rover_text, rover_conf = rover_char_vote(hyps, char_thresh=ct, ins_thresh=it)
    best_frame_text, best_w = pick_best_frame(frame_items)

    cands: List[Tuple[str, str]] = []
    if base_text:
        cands.append((base_text, "subset_vote"))
    if rover_text:
        cands.append((rover_text, "rover_vote"))
    if best_frame_text:
        cands.append((best_frame_text, "best_frame"))

    scored = []
    for t, name in cands:
        s = score_candidate(t, hyps, min_text_len=min_text_len)
        scored.append((s, name, t))
    scored.sort(key=lambda x: x[0])
    best_s, best_name, best_t = scored[0]

    baseline_s = None
    for s, name, _t in scored:
        if name == "subset_vote":
            baseline_s = s
            break
    if baseline_s is not None and abs(baseline_s - best_s) <= 0.015 and best_name != "subset_vote":
        best_name = "subset_vote"
        best_t = base_text
        best_s = baseline_s

    final_conf = float(max(0.0, min(1.0, 1.0 - best_s)))
    debug = {
        "candidates": [{"method": name, "text": t, "score": float(s)} for s, name, t in scored[:5]],
        "baseline": {"text": base_text, "mean_prob": float(base_mp), "frame_count": int(base_fc)},
        "rover": {"text": rover_text, "conf": float(rover_conf), "char_thresh": float(ct), "ins_thresh": float(it)},
        "best_frame": {"text": best_frame_text, "weight": float(best_w)},
    }
    return best_t, final_conf, best_name, debug


# ----------------------------
# PaddleOCR init and run (PaddleOCR 3.x friendly)
# ----------------------------
def pick_paddle_device() -> str:
    if paddle is None:
        return "cpu"
    try:
        has_cuda = paddle.is_compiled_with_cuda()
        gpu_count = paddle.device.cuda.device_count() if has_cuda else 0
        dev = "gpu:0" if gpu_count > 0 else "cpu"
    except Exception:
        dev = "cpu"
    try:
        paddle.device.set_device(dev)
    except Exception:
        dev = "cpu"
        paddle.device.set_device("cpu")
    return dev


def _looks_like_paddlex_model_dir(model_dir: str) -> bool:
    p = Path(model_dir)
    if not p.exists() or not p.is_dir():
        return False
    return (p / "inference.yml").exists() or (p / "inference.yaml").exists()


def init_paddleocr(args: argparse.Namespace) -> Any:
    if PaddleOCR is None:
        raise RuntimeError("paddleocr import 실패. `pip install paddleocr` / paddle 설치를 먼저 확인하세요.")

    device_str = pick_paddle_device()

    # PaddleOCR 3.x:
    # - use_angle_cls는 deprecated이며 use_textline_orientation와 배타적이므로 kwargs로 넘기지 않는다.
    use_tlo = bool(args.use_textline_orientation or args.use_angle_cls)

    kwargs: Dict[str, Any] = dict(
        lang=args.lang,
        device=device_str,
        use_textline_orientation=use_tlo,
        use_doc_orientation_classify=bool(args.use_doc_orientation),
        use_doc_unwarping=bool(args.use_doc_unwarping),
        text_rec_score_thresh=float(args.text_rec_score_thresh),
    )

    # 모델 경로는 "PaddleX 형식(inference.yml 포함)"일 때만 적용 (아니면 경고 후 기본 모델로 진행)
    if args.det_model_dir:
        if _looks_like_paddlex_model_dir(args.det_model_dir):
            kwargs["text_detection_model_dir"] = args.det_model_dir
        else:
            log(f"[WARN] det_model_dir에 inference.yml이 없습니다: {args.det_model_dir} (무시하고 기본 det 사용)")

    if args.rec_model_dir:
        if _looks_like_paddlex_model_dir(args.rec_model_dir):
            kwargs["text_recognition_model_dir"] = args.rec_model_dir
        else:
            log(f"[WARN] rec_model_dir에 inference.yml이 없습니다: {args.rec_model_dir} (무시하고 기본 rec 사용)")

    if args.cls_model_dir:
        if _looks_like_paddlex_model_dir(args.cls_model_dir):
            kwargs["textline_orientation_model_dir"] = args.cls_model_dir
        else:
            log(f"[WARN] cls_model_dir에 inference.yml이 없습니다: {args.cls_model_dir} (무시하고 기본 orientation 사용)")

    log(
        f"[INFO] PaddleOCR device={device_str} lang={args.lang} "
        f"use_textline_orientation={use_tlo} text_rec_score_thresh={args.text_rec_score_thresh} "
        f"rec_model_dir={args.rec_model_dir or '(default)'}"
    )
    return PaddleOCR(**kwargs)


def _as_dict(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    # PaddleOCR result object often exposes .json
    j = getattr(x, "json", None)
    if isinstance(j, dict):
        return j
    # Some may have to_dict()
    td = getattr(x, "to_dict", None)
    if callable(td):
        try:
            d = td()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return None


def extract_lines_from_paddleocr_result(res_obj: Any) -> List[Tuple[str, float]]:
    """
    PaddleOCR 3.x predict() output: dict-like with keys:
      res -> { rec_texts: [...], rec_scores: [...] , ... }
    (older style output also handled best-effort)
    """
    d = _as_dict(res_obj)

    # Case A) PaddleOCR 3.x dict format
    if isinstance(d, dict):
        root = d.get("res", d)
        texts = root.get("rec_texts", None)
        scores = root.get("rec_scores", None)

        if texts is not None:
            if not isinstance(texts, list):
                texts = [texts]
            if scores is None:
                scores = [1.0] * len(texts)
            else:
                # scores may be list/np.ndarray
                try:
                    scores = list(scores)
                except Exception:
                    scores = [1.0] * len(texts)

            out: List[Tuple[str, float]] = []
            for t, s in zip(texts, scores):
                if isinstance(t, str) and t.strip():
                    out.append((t, float(s)))
            return out

    # Case B) Older PaddleOCR list format: [ [box, (text, score)], ... ]
    if isinstance(res_obj, list):
        # Sometimes predict returns list with 1 element per input
        if len(res_obj) == 1 and not (isinstance(res_obj[0], (list, tuple)) and len(res_obj[0]) == 2):
            # try unwrap
            inner = res_obj[0]
            return extract_lines_from_paddleocr_result(inner)

        out2: List[Tuple[str, float]] = []
        for item in res_obj:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rec = item[1]
                if isinstance(rec, (list, tuple)) and len(rec) >= 2 and good_text(rec[0]):
                    out2.append((str(rec[0]), float(rec[1])))
        return out2

    return []


def run_ocr_on_input(ocr: Any, inp: Any) -> List[Tuple[str, float]]:
    """
    inp: str path or numpy image
    """
    # Prefer predict_iter() if available (generator)
    if hasattr(ocr, "predict_iter"):
        try:
            it = ocr.predict_iter(input=inp)
        except TypeError:
            it = ocr.predict_iter(inp)
        except Exception:
            it = None
        if it is not None:
            lines_all: List[Tuple[str, float]] = []
            for one in it:
                lines_all.extend(extract_lines_from_paddleocr_result(one))
            return lines_all

    if hasattr(ocr, "predict"):
        try:
            it2 = ocr.predict(input=inp)
        except TypeError:
            it2 = ocr.predict(inp)
        except Exception:
            it2 = None

        if it2 is not None:
            # predict() may return list; unify
            if isinstance(it2, list):
                lines_all: List[Tuple[str, float]] = []
                for one in it2:
                    lines_all.extend(extract_lines_from_paddleocr_result(one))
                return lines_all
            # generator-like
            try:
                lines_all = []
                for one in it2:
                    lines_all.extend(extract_lines_from_paddleocr_result(one))
                return lines_all
            except Exception:
                pass

    # Fallback: deprecated ocr()
    if hasattr(ocr, "ocr"):
        try:
            out = ocr.ocr(inp)
            return extract_lines_from_paddleocr_result(out)
        except Exception:
            return []

    return []


def join_lines(items: List[Tuple[str, float]]) -> Tuple[str, float]:
    if not items:
        return "", 0.0
    texts = [t for t, _ in items if good_text(t)]
    if not texts:
        return "", 0.0
    prob = float(sum(p for _, p in items) / max(len(items), 1))
    return "".join(texts), prob


# ----------------------------
# Object label inference per track
# ----------------------------
def infer_object_label_for_track(track_dir: Path) -> str:
    counts = defaultdict(int)
    for p in track_dir.glob("*.jpg"):
        if p.name.lower() == "best.jpg":
            continue
        parts = p.name.split("_")
        if len(parts) >= 3:
            counts[parts[2]] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else "unknown"


def run_ocr_fusion_on_crops(args: argparse.Namespace, crops_root: Path, out_root: Path) -> Dict[str, Any]:
    ensure_dir(out_root)
    out_main = out_root / "ocr_tracks.jsonl"
    out_review = out_root / "ocr_tracks_needs_review.jsonl"

    ocr = init_paddleocr(args)

    ids = sorted([p for p in crops_root.iterdir() if p.is_dir() and p.name.startswith("id_")])
    log(f"[INFO] tracks={len(ids)} crops_root={crops_root}")

    t0 = time.time()
    n_review = 0

    with out_main.open("w", encoding="utf-8") as f_main, out_review.open("w", encoding="utf-8") as f_rev:
        for tid_dir in _progress_iter(ids, desc="OCR", unit="track"):
            imgs = sorted(list(tid_dir.glob("*.jpg"))) + sorted(list(tid_dir.glob("*.png")))
            imgs = [p for p in imgs if p.name.lower() != "best.jpg"]
            if (tid_dir / "best.jpg").exists():
                imgs.append(tid_dir / "best.jpg")

            if not imgs:
                rec = {
                    "id": tid_dir.name,
                    "object": "unknown",
                    "text": "",
                    "main_text_prob": 0.0,
                    "consensus": 0.0,
                    "method": "empty",
                    "needs_review": True,
                    "needs_review_reasons": ["no_images"],
                    "frames": [],
                    "frame_texts": [],
                }
                f_main.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_rev.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_review += 1
                continue

            if len(imgs) > args.max_frames_per_track:
                idxs = np.linspace(0, len(imgs) - 1, args.max_frames_per_track).round().astype(int).tolist()
                imgs = [imgs[i] for i in idxs]

            frame_items: List[Tuple[int, str, float, float]] = []
            frames: List[int] = []
            frame_texts_out: List[Dict[str, Any]] = []

            # OCR on original crops
            for k, img_path in enumerate(imgs):
                fr = parse_frame_no(img_path.name, fallback=k)
                img = cv2.imread(str(img_path))
                q = image_quality_score(img) if img is not None else 0.1

                frames.append(fr)
                items: List[Dict[str, Any]] = []

                if img is not None:
                    # 안정성을 위해 "파일 경로"로 먼저 시도 (PaddleOCR 3.x에서 경로 입력이 가장 안정적인 편)
                    lines = run_ocr_on_input(ocr, str(img_path))
                    if not lines:
                        # 혹시 경로 입력이 막혀있으면 ndarray로도 시도
                        lines = run_ocr_on_input(ocr, img)

                    joined, jprob = join_lines(lines)
                    if good_text(joined):
                        nt = norm_text(joined)
                        if nt:
                            items.append({"text": nt, "prob": float(jprob), "variant": "orig"})
                            frame_items.append((fr, nt, float(jprob), float(q)))

                frame_texts_out.append({"frame": fr, "file": img_path.name, "quality": float(q), "items": items})


            obj = infer_object_label_for_track(tid_dir)

            final_text, final_conf, method, debug = adaptive_fusion(
                obj=obj,
                frame_items=frame_items,
                rover_char_thresh=args.rover_char_thresh,
                rover_ins_thresh=args.rover_ins_thresh,
                min_text_len=args.min_text_len,
            )

            needs_review = False
            reasons: List[str] = []

            if final_text and (not is_valid_final_text(final_text)):
                final_text = ""
                final_conf = 0.0
                needs_review = True
                reasons.append("filtered_invalid_final_text")

            if final_conf < args.review_consensus_thresh:
                needs_review = True
                reasons.append(f"low_consensus<{args.review_consensus_thresh}")
            if len(frame_items) < 3:
                needs_review = True
                reasons.append("too_few_frames_with_text")
            if not final_text:
                needs_review = True
                reasons.append("empty_final_text")

            rec = {
                "id": tid_dir.name,
                "object": obj,
                "text": final_text,
                "main_text_prob": float(final_conf),
                "consensus": float(final_conf),
                "method": method,
                "needs_review": needs_review,
                "needs_review_reasons": reasons,
                "candidates_top5": debug.get("candidates", []),
                "frames": frames,
                "frame_texts": frame_texts_out
            }

            f_main.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if needs_review:
                f_rev.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_review += 1

    dt = time.time() - t0
    log(f"[OCR DONE] wrote={out_main} needs_review={n_review}/{len(ids)} time={dt:.1f}s")
    return {"ocr_out": str(out_root), "needs_review": int(n_review), "tracks": int(len(ids))}


def main() -> int:
    args = parse_args()
    crops_root = Path(args.input)
    out_root = Path(args.out)

    log(f"[INFO] crops_root={crops_root}")
    log(f"[INFO] out_root={out_root}")

    if not crops_root.exists():
        raise SystemExit(f"[ERROR] crops input not found: {crops_root}")

    ensure_dir(out_root)
    run_ocr_fusion_on_crops(args, crops_root=crops_root, out_root=out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
