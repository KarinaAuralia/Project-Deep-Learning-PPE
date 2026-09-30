import os
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ---- Paths: lokal vs Vercel ----
IS_VERCEL = os.environ.get("VERCEL") == "1"

if IS_VERCEL:
    WRITE_ROOT = tempfile.gettempdir()
else:
    WRITE_ROOT = STATIC_DIR

UPLOAD_DIR = os.path.join(WRITE_ROOT, "uploads")
RESULT_DIR = os.path.join(WRITE_ROOT, "results")
DB_PATH    = os.path.join(WRITE_ROOT, "ppe.db")

SAMPLE_DIR = os.path.join(STATIC_DIR, "samples")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.onnx")


def ensure_dir(p):
    if os.path.exists(p) and not os.path.isdir(p):
        os.remove(p)
    os.makedirs(p, exist_ok=True)


ensure_dir(UPLOAD_DIR)
ensure_dir(RESULT_DIR)

# ---------- Kelas (urutan HARUS sama dengan model ONNX) ----------
CLASS_NAMES  = ["Helmet", "Mask", "Safety Vest", "boots", "glove"]
REQUIRED_PPE = ["Helmet", "Mask", "Safety Vest", "boots", "glove"]

COLORS = {
    "Helmet":      (59, 130, 246),
    "Mask":        (234, 179, 8),
    "Safety Vest": (16, 185, 129),
    "boots":       (168, 85, 247),
    "glove":       (6, 182, 212),
}

SAMPLES = [
    {"label": "Sample 01", "file": "sample1.jpg", "sub": "Rig Floor"},
    {"label": "Sample 02", "file": "sample2.jpg", "sub": "Dermaga"},
    {"label": "Sample 03", "file": "sample3.jpg", "sub": "Kilang"},
]

TEAM = [
    {"nama": "Difta Alzena Sakhi", "nim": "23083010061", "initsial": "DA",
     "peran": "Ketua · Model Architect & Quantization",
     "linkedin": "https://www.linkedin.com/in/difta-alzena-sakhi-09a2b1315/",
     "foto": "anggota1.jpg"},
    {"nama": "Azizah Zalfa Assyadida", "nim": "23083010062", "initsial": "AZ",
     "peran": "Data Pipeline & Augmentation",
     "linkedin": "https://www.linkedin.com/in/azalfassyadida/",
     "foto": "anggota2.jpg"},
    {"nama": "Ni Luh Ayu Nariswari Dewi", "nim": "23083010068", "initsial": "NA",
     "peran": "Edge Runtime & Inference Optimization",
     "linkedin": "https://www.linkedin.com/in/ayu-nariswari/",
     "foto": "anggota3.jpg"},
    {"nama": "Karina Auralia", "nim": "23083010072", "initsial": "KA",
     "peran": "Frontend & Integrasi Web",
     "linkedin": "https://www.linkedin.com/in/karinaauralia/",
     "foto": "anggota4.jpg"},
    {"nama": "Tiara Audrey Anugerah Hadin", "nim": "23083010079", "initsial": "TA",
     "peran": "Technical Writer & Validasi HSE",
     "linkedin": "https://www.linkedin.com/in/tiara-audrey/",
     "foto": "anggota5.jpg"},
]

# ---------- Perbandingan Model ----------
MODEL_COMPARISON = [
    {"nama": "YOLOv8n FP32 Baseline", "p": 89.42, "r": 79.72, "f1": 84.29,
     "map50": 86.04, "map": 57.62, "size": 5.96, "best": True},
    {"nama": "YOLOv8n INT8 ONNX",     "p": 89.52, "r": 77.52, "f1": 83.09,
     "map50": 79.22, "map": 53.08, "size": 3.27, "best": False},
    {"nama": "YOLOv8n Pruned",        "p": 88.54, "r": 77.46, "f1": 82.63,
     "map50": 84.69, "map": 56.07, "size": 5.99, "best": False},
]

# ---------- Akurasi per kelas (dari model terbaik) ----------
ACCURACY = [
    {"kelas": "Mask",        "sub": "Masker / Respirator", "p": 93.5, "r": 89.8, "map50": 94.1, "map": 64.7},
    {"kelas": "Helmet",      "sub": "Helm Keselamatan",    "p": 91.9, "r": 78.4, "map50": 83.9, "map": 54.8},
    {"kelas": "Safety Vest", "sub": "Rompi High-Vis",      "p": 90.9, "r": 76.3, "map50": 83.8, "map": 60.0},
    {"kelas": "boots",       "sub": "Sepatu Safety",       "p": 82.1, "r": 78.2, "map50": 85.4, "map": 56.7},
    {"kelas": "glove",       "sub": "Sarung Tangan",       "p": 88.7, "r": 75.9, "map50": 83.0, "map": 51.9},
]
ACCURACY_AVG = {"p": 89.4, "r": 79.7, "map50": 86.0, "map": 57.6}

# ---------- Load ONNX ----------
session     = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME  = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name
INPUT_SIZE  = 640


# ---------- DB ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            area TEXT, image_path TEXT, result_path TEXT,
            total_person INTEGER, total_compliant INTEGER, total_violation INTEGER,
            created_at TEXT
        )
    """)
    conn.commit(); conn.close()


def save_analysis(area, img, res, total, comp, viol):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO analysis (area,image_path,result_path,total_person,total_compliant,total_violation,created_at) VALUES (?,?,?,?,?,?,?)",
        (area, img, res, total, comp, viol, datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    conn.commit(); conn.close()


# ---------- Preprocess ----------
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    ow, oh = img.size
    scale = min(INPUT_SIZE / ow, INPUT_SIZE / oh)
    nw, nh = int(ow * scale), int(oh * scale)
    px, py = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2

    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))
    canvas.paste(img.resize((nw, nh)), (px, py))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr, img, scale, px, py


# ---------- NMS ----------
def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    A = (box[2] - box[0]) * (box[3] - box[1])
    B = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (A + B - inter + 1e-9)


def nms(xyxy, scores, class_ids, iou_thr):
    keep_all = []
    for c in np.unique(class_ids):
        mask = class_ids == c
        c_boxes, c_scores = xyxy[mask], scores[mask]
        c_idx = np.where(mask)[0]
        order = c_scores.argsort()[::-1]
        while len(order):
            i = order[0]
            keep_all.append(c_idx[i])
            if len(order) == 1: break
            ious = compute_iou(c_boxes[i], c_boxes[order[1:]])
            order = order[1:][ious < iou_thr]
    return keep_all


def decode(raw, ow, oh, scale, px, py, conf_thr=0.25, iou_thr=0.45):
    """
    Format output model: (1, 300, 6) = [x1, y1, x2, y2, confidence, class_id]
    Model sudah include NMS.
    """
    preds = raw[0] if raw.ndim == 3 else raw  # (300, 6)

    if preds.shape[1] != 6:
        # Fallback: format lama (jika model berubah)
        return _decode_legacy(raw, ow, oh, scale, px, py, conf_thr, iou_thr)

    confs = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    # Filter by confidence
    mask = confs > conf_thr
    if not mask.any():
        return []

    xyxy = preds[mask, :4].copy()
    confs = confs[mask]
    class_ids = class_ids[mask]

    # Reverse letterbox: dari space 640 → gambar asli
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / scale
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, ow)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, oh)

    # Filter bbox valid (x1 < x2, y1 < y2)
    valid = (xyxy[:, 0] < xyxy[:, 2]) & (xyxy[:, 1] < xyxy[:, 3])
    xyxy = xyxy[valid]
    confs = confs[valid]
    class_ids = class_ids[valid]

    # Filter class_id valid (0..len(CLASS_NAMES)-1)
    valid_cls = (class_ids >= 0) & (class_ids < len(CLASS_NAMES))
    xyxy = xyxy[valid_cls]
    confs = confs[valid_cls]
    class_ids = class_ids[valid_cls]

    return [
        {"class": CLASS_NAMES[int(class_ids[i])],
         "conf":  float(confs[i]),
         "bbox":  [float(v) for v in xyxy[i]]}
        for i in range(len(confs))
    ]


def _decode_legacy(raw, ow, oh, scale, px, py, conf_thr, iou_thr):
    """Fallback untuk format YOLOv8 standar [4+nc, 8400]."""
    preds = raw[0] if raw.ndim == 3 else raw
    nc = len(CLASS_NAMES)

    if preds.shape[0] < preds.shape[1]:
        boxes  = preds[:4, :]
        scores = preds[4:4+nc, :]
    else:
        boxes  = preds[:, :4].T
        scores = preds[:, 4:4+nc].T

    class_ids = np.argmax(scores, axis=0)
    confs     = np.max(scores, axis=0)
    mask = confs > conf_thr
    if not mask.any():
        return []

    boxes, class_ids, confs = boxes[:, mask].T, class_ids[mask], confs[mask]

    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / scale
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, ow)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, oh)

    keep = nms(xyxy, confs, class_ids, iou_thr)
    return [
        {"class": CLASS_NAMES[int(class_ids[i])],
         "conf":  float(confs[i]),
         "bbox":  [float(v) for v in xyxy[i]]}
        for i in keep
    ]


# ---------- Drawing ----------
def draw_boxes(img, detections):
    draw = ImageDraw.Draw(img)
    font_size = max(16, min(28, img.width // 40))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = COLORS.get(d["class"], (255, 255, 255))
        label = f'{d["class"]} {d["conf"]*100:.0f}%'

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ly = max(0, y1 - th - 8)
        draw.rectangle([x1, ly, x1 + tw + 12, ly + th + 8], fill=color)
        draw.text((x1 + 6, ly + 4), label, fill="white", font=font)
    return img


# ---------- Analyze ----------
def analyze_image(image_path, conf_thr=0.25):
    inp, orig_img, scale, px, py = preprocess(image_path)
    raw = session.run([OUTPUT_NAME], {INPUT_NAME: inp})[0]
    all_dets = decode(raw, orig_img.width, orig_img.height,
                      scale, px, py, conf_thr=conf_thr)

    # Ambil confidence tertinggi per kelas
    best = {}
    for d in all_dets:
        k = d["class"]
        if k not in best or d["conf"] > best[k]["conf"]:
            best[k] = d

    # Status tiap APD wajib
    ppe_status = []
    for req in REQUIRED_PPE:
        if req in best:
            ppe_status.append({"name": req, "detected": True, "conf": best[req]["conf"]})
        else:
            ppe_status.append({"name": req, "detected": False, "conf": 0.0})

    detected_count = sum(1 for p in ppe_status if p["detected"])
    missing_count  = len(REQUIRED_PPE) - detected_count

    result_img = draw_boxes(orig_img.copy(), all_dets)
    res_name = f"{uuid.uuid4().hex[:8]}.jpg"
    result_img.save(os.path.join(RESULT_DIR, res_name), quality=85)

    return {
        "ppe_status":     ppe_status,
        "total_detected": detected_count,
        "total_required": len(REQUIRED_PPE),
        "total_missing":  missing_count,
        "result_name":    res_name,
        "conf_used":      conf_thr,
    }


# ---------- Static files ----------
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)


# ---------- Routes ----------
def render(result=None):
    return render_template("index.html",
        team=TEAM, samples=SAMPLES, required=REQUIRED_PPE,
        accuracy=ACCURACY, acc_avg=ACCURACY_AVG,
        model_comparison=MODEL_COMPARISON,
        result=result)


@app.route("/")
def home():
    return render(None)


@app.route("/analyze", methods=["POST"])
def analyze_upload():
    f = request.files.get("image")
    if not f or f.filename == "":
        return redirect(url_for("home"))
    conf_thr = float(request.form.get("conf", 0.25))
    ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
    name = f"{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)
    r = analyze_image(path, conf_thr)
    r["image_name"] = name
    save_analysis("-", name, r["result_name"],
                  r["total_detected"], r["total_detected"], r["total_missing"])
    return render(r)


@app.route("/try/<sample_file>", methods=["POST"])
def try_sample(sample_file):
    src = os.path.join(SAMPLE_DIR, sample_file)
    if not os.path.exists(src):
        return redirect(url_for("home"))
    conf_thr = float(request.form.get("conf", 0.25))
    name = f"{uuid.uuid4().hex[:8]}.jpg"
    dst = os.path.join(UPLOAD_DIR, name)
    shutil.copy(src, dst)
    r = analyze_image(dst, conf_thr)
    r["image_name"] = name
    save_analysis("Sample", name, r["result_name"],
                  r["total_detected"], r["total_detected"], r["total_missing"])
    return render(r)


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)