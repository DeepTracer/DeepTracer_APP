import cv2
import numpy as np
import os
import re
import torch
import json
import base64
import sys
import subprocess
import shutil
from collections import OrderedDict
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# =========================================================
# 1. 시스템 경로 설정 (하이픈 폴더 및 모듈 로딩 해결)
# =========================================================
benchmark_path = os.path.abspath("./deep-text-recognition-benchmark")
if benchmark_path not in sys.path:
    sys.path.insert(0, benchmark_path)

# 이제 하이픈이 포함된 폴더 안의 파일들을 직접 import 할 수 있습니다.
from utils import AttnLabelConverter
from text_recognition_model import TextRecognitionModel

# =========================================================
# 2. 환경 설정 및 모델 경로
# =========================================================
# YOLO 및 OCR 모델 가중치 경로
YOLO_WEIGHTS = os.environ.get("PLATE_YOLO_MODEL", 'runs/detect/train6/weights/best.pt')
OCR_WEIGHTS = os.environ.get(
    "PLATE_OCR_MODEL",
    os.path.abspath("./deep-text-recognition-benchmark/saved_models/sec_train/best_accuracy.pth")
)

print("[DEBUG][INF] inference_module start", flush=True)
print(f"[DEBUG][INF] YOLO_WEIGHTS = {YOLO_WEIGHTS}", flush=True)
print(f"[DEBUG][INF] OCR_WEIGHTS  = {OCR_WEIGHTS}", flush=True)
print(f"[DEBUG][INF] YOLO exists? {os.path.exists(YOLO_WEIGHTS)}", flush=True)
print(f"[DEBUG][INF] OCR exists?  {os.path.exists(OCR_WEIGHTS)}", flush=True)

# 추론 파라미터
CONFIDENCE_THRESHOLD = 0.3
OCR_CONFIDENCE_THRESHOLD = 0.3
SKIP_FRAMES = 500

# [중요] 학습 시 사용한 96개 글자 (점 '.' 제외)
# 10개 숫자 + 86개 한글 = 총 96개 (모델 num_class는 98이 됨)
CHARACTER_SETS = "0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바배뱌버보부북사산서소수아악안양어연영오용우울원육이인자작저전조주중지차천초추충카타파평포하허호홀히"

# 폰트 경로 설정
if os.name == 'nt': 
    FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
else: 
    FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plate_pattern = re.compile(r"\D{0,5}\d{0,3}\D{1}\d{4}$")

# =========================================================
# 3. 모델 로딩 함수 (가중치 이름표 불일치 해결)
# =========================================================

def load_trained_ocr_model(weights_path, device):
    """
    학습된 STR 모델을 로드합니다. (입력 채널 1, 글자수 96개 최적화)
    """
    converter = AttnLabelConverter(CHARACTER_SETS)
    num_class = len(converter.character) # [GO]와 [s]가 포함된 숫자
    
    # 모델 뼈대 생성 (학습된 체크포인트 사양과 일치시킴)
    model = TextRecognitionModel(
        Transformation='TPS',
        FeatureExtraction='ResNet',
        SequenceModeling='BiLSTM',
        Prediction='Attn',
        num_fiducial=20,
        img_scale=(32, 100), 
        input_channel=1,     # 흑백 모델 (체크포인트 규격)
        output_channel=512,
        hidden_size=256,
        num_class=num_class,
        batch_max_length=25
    )
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights_path}")
    
    # 가중치 로드 및 이름표(module.) 정제
    state_dict = torch.load(weights_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    print(f"✅ OCR 모델 가중치 로드 성공! (입력 채널: 1, 클래스: {num_class})")
    
    # GPU 가속 설정
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    return model, converter

# =========================================================
# 4. 핵심 추론 및 유틸리티 함수
# =========================================================

def run_ocr(model, converter, image):
    """
    학습된 모델을 사용해 번호판 글자 인식 (흑백 변환 포함)
    """
    # 1. 전처리: 흑백 변환 및 리사이즈
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    img = cv2.resize(img, (100, 32))
    
    # 2. 텐서 변환 및 정규화
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    img_tensor.sub_(0.5).div_(0.5)

    batch_size = 1
    with torch.no_grad():
        # Attention 예측을 위한 빈 텐서 준비
        length_for_pred = torch.IntTensor([25] * batch_size).to(DEVICE)
        text_for_pred = torch.LongTensor(batch_size, 25 + 1).fill_(0).to(DEVICE)
        
        # 모델 추론
        preds = model(img_tensor, text_for_pred, is_train=False)
        
        # 결과 디코딩
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        
        # 결과 텍스트 후처리 ([s] 제거)
        text = preds_str[0]
        if '[s]' in text:
            text = text[:text.find('[s]')]
            
        # 신뢰도 점수 계산
        preds_prob = torch.nn.functional.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        conf = preds_max_prob[0].cumprod(dim=0)[-1].item()
        
    return text, float(conf)

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def get_time_str(msec):
    seconds = int(msec / 1000)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def detect_with_slicing(model, frame, confidence=0.25):
    """ YOLO를 사용하여 프레임을 반으로 나눠 정밀 검출 """
    h, w, _ = frame.shape
    mid_x, overlap = w // 2, 100
    results_coords = []
    
    # 왼쪽/오른쪽 슬라이싱 검출
    for start_x, end_x, offset in [(0, mid_x + overlap, 0), (mid_x - overlap, w, mid_x - overlap)]:
        slice_img = frame[:, start_x:end_x]
        results = model(slice_img, conf=confidence, verbose=False)
        for res in results:
            for box in res.boxes:
                c = box.xyxy[0].cpu().tolist()
                c[0] += offset; c[2] += offset
                results_coords.append(c + [box.conf.item()])
    return results_coords

def transform_vertical_plate(plate_img):
    """ 2단 번호판을 1단으로 변환 """
    h, w, _ = plate_img.shape
    if w / h > 2.5: return plate_img
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([40, 255, 255]))
    if cv2.countNonZero(mask) / (w * h) > 0.3:
        split = int(w * 0.25)
        left, right = plate_img[:, :split], plate_img[:, split:]
        top = cv2.resize(left[:h//2, :], (w//4, h))
        bot = cv2.resize(left[h//2:, :], (w//4, h))
        return np.hstack([cv2.resize(np.hstack([top, bot]), (int(w*0.3), h)), right])
    return plate_img

def draw_text(img, text, x, y, color=(0,255,0)):
    """ 이미지 위에 한글 번호판 텍스트 그리기 """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype(FONT_PATH, 30)
    except: font = ImageFont.load_default()
    draw.rectangle(draw.textbbox((x, y-35), text, font=font), fill=(0,0,0,150))
    draw.text((x, y-35), text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def enhance_for_motion_blur(img):
    """ 화질 개선 (CLAHE 및 샤프닝) """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    res = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return cv2.addWeighted(res, 1.5, cv2.GaussianBlur(res, (0,0), 3.0), -0.5, 0)

# =========================================================
# 5. 메인 실행 함수 (run_plate_detection)
# =========================================================

def run_plate_detection(video_path, output_dir, progress_callback=None):
    print(f"🚀 실행 장치: {DEVICE}")
    
    # 모델 로드
    yolo = YOLO(YOLO_WEIGHTS)
    ocr_model, ocr_converter = load_trained_ocr_model(OCR_WEIGHTS, DEVICE)

    # 비디오 준비
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    capture_dir = os.path.join(output_dir, "captured_plates")
    os.makedirs(capture_dir, exist_ok=True)
    
    result_video_path = os.path.join(output_dir, f"{file_name}_result.mp4")
    result_json_path = os.path.join(output_dir, f"{file_name}_result.json")
    
    # 임시 AVI 파일 작성 (OpenCV 에러 방지를 위해 mp4v 사용)
    temp_avi = result_video_path.replace('.mp4', '_temp.avi')
    out = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_cnt, json_results, rects_to_draw = 0, [], []

    print("▶️ 분석 시작...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_cnt += 1
        time_str = get_time_str(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        if frame_cnt % SKIP_FRAMES == 0:
            rects_to_draw = []
            detections = detect_with_slicing(yolo, frame, confidence=CONFIDENCE_THRESHOLD)

            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                det_conf = det[4]

                # ROI 추출 및 OCR 실행
                roi = frame[max(0, y1-5):min(height, y2+5), max(0, x1-5):min(width, x2+5)]
                if roi.size == 0: continue
                
                roi = enhance_for_motion_blur(transform_vertical_plate(roi))
                text, ocr_conf = run_ocr(ocr_model, ocr_converter, roi)

                if text != "invalid" and ocr_conf >= OCR_CONFIDENCE_THRESHOLD:
                    print(f"✨ [{time_str}] {text} (OCR: {ocr_conf:.2f})")
                    cv2.imwrite(os.path.join(capture_dir, f"{text}_{frame_cnt}.jpg"), roi)
                    
                    json_results.append({
                        "plate_number": text, "timestamp": time_str,
                        "ocr_confidence": round(ocr_conf, 4), "frame_index": frame_cnt,
                        "image_base64": image_to_base64(roi)
                    })
                    rects_to_draw.append({'coords': (x1, y1, x2, y2), 'text': text})

        # 프레임에 결과 그리기
        for item in rects_to_draw:
            ix1, iy1, ix2, iy2 = item['coords']
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 3)
            frame = draw_text(frame, item['text'], ix1, iy1)

        out.write(frame)
        if progress_callback: progress_callback(frame_cnt, total_frames)
        if frame_cnt % 50 == 0: print(f"진행 중... {frame_cnt}/{total_frames}")

    cap.release()
    out.release()
    
    # 6. 최종 MP4 변환 (H.264 코덱으로 웹 호환성 확보)
    print("🎬 MP4 변환 및 최종 저장 중...")
    try:
        ffmpeg_path = shutil.which('ffmpeg') or 'ffmpeg'
        subprocess.run([ffmpeg_path, '-y', '-i', temp_avi, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', result_video_path], capture_output=True)
        if os.path.exists(temp_avi): os.remove(temp_avi)
    except:
        os.rename(temp_avi, result_video_path)

    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 모든 작업 완료! 결과: {result_video_path}")
    return result_video_path, result_json_path, json_results

if __name__ == "__main__":
    import sys
    
    # 아규먼트(인자)가 들어왔을 때만 실행
    if len(sys.argv) >= 3:
        input_video = sys.argv[1]
        output_path = sys.argv[2]
        print(f"DEBUG: 앱으로부터 인자 받음 - 파일: {input_video}, 경로: {output_path}")
        run_plate_detection(input_video, output_path)
    else:
        # 인자가 없을 때(터미널에서 그냥 실행할 때)만 기본값 사용
        print("DEBUG: 기본값으로 실행합니다.")
        run_plate_detection('./주행.mp4', 'results')