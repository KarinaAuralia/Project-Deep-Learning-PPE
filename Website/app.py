import os
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image, ImageDraw
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ---- Paths: lokal vs Vercel ----
IS_VERCEL = os.environ.get("VERCEL") == "1"

if IS_VERCEL:
    # Vercel filesystem read-only kecuali /tmp
    WRITE_ROOT = tempfile.gettempdir()          # /tmp
else:
    WRITE_ROOT = STATIC_DIR                     # lokal: static/

UPLOAD_DIR = os.path.join(WRITE_ROOT, "uploads")
RESULT_DIR = os.path.join(WRITE_ROOT, "results")
DB_PATH    = os.path.join(WRITE_ROOT, "ppe.db")

# Sample & model tetap dibaca dari repo (read-only, tidak masalah)
SAMPLE_DIR = os.path.join(STATIC_DIR, "samples")
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")


def ensure_dir(p):
    if os.path.exists(p) and not os.path.isdir(p):
        os.remove(p)
    os.makedirs(p, exist_ok=True)


ensure_dir(UPLOAD_DIR)
ensure_dir(RESULT_DIR)

CLASS_NAMES  = ["Mask", "Vest", "Person", "Gloves", "Hard_hat", "Safety_boots"]
REQUIRED_PPE = ["Mask", "Vest", "Gloves", "Hard_hat", "Safety_boots"]
COLORS = {
    "Person":       (34, 197, 94),
    "Hard_hat":     (59, 130, 246),
    "Vest":         (234, 179, 8),
    "Gloves":       (239, 68, 68),
    "Safety_boots": (168, 85, 247),
    "Mask":         (6, 182, 212),
}

SAMPLES = [
    {"label": "Sample 01", "file": "sample1.jpg"},
    {"label": "Sample 02", "file": "sample2.jpg"},
    {"label": "Sample 03", "file": "sample3.jpg"},
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

ACCURACY = [
    {"kelas": "Hard_hat",     "sub": "Helm Keselamatan",   "p": 95.3, "r": 92.7, "map50": 97.0, "map": 76.3},
    {"kelas": "Vest",         "sub": "Rompi High-Vis",     "p": 91.8, "r": 91.4, "map50": 96.4, "map": 76.1},
    {"kelas": "Person",       "sub": "Kru Lapangan",       "p": 91.4, "r": 91.3, "map50": 97.0, "map": 76.0},
    {"kelas": "Safety_boots", "sub": "Sepatu Safety",      "p": 87.3, "r": 76.7, "map50": 85.2, "map": 48.6},
    {"kelas": "Gloves",       "sub": "Sarung Tangan",      "p": 80.9, "r": 85.9, "map50": 84.9, "map": 50.2},
    {"kelas": "Mask",         "sub": "Masker / Respirator","p": 80.0, "r": 80.2, "map50": 80.1, "map": 48.7},
]
ACCURACY_AVG = {"p": 87.8, "r": 86.4, "map50": 90.1, "map": 62.6}

# ---- Load ONNX sekali (bukan ultralytics) ----
session    = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name
INPUT_SIZE = 640


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


# ---------- ONNX Inference ----------
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    ow, oh = img.size
    scale = min(INPUT_SIZE / ow, INPUT_SIZE / oh)
    nw, nh = int(ow * scale), int(oh * scale)
    px, py = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2

    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))
    canvas.paste(img.resize((nw, nh)), (px, py))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]  # NCHW
    return arr, img, scale, px, py


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


def decode(raw, ow, oh, scale, px, py, conf_thr=0.35, iou_thr=0.45):
    preds = raw[0]           # [4+nc, 8400]
    nc = len(CLASS_NAMES)
    boxes  = preds[:4, :]
    scores = preds[4:4+nc, :]

    class_ids = np.argmax(scores, axis=0)
    confs     = np.max(scores, axis=0)
    mask = confs > conf_thr
    if not mask.any(): return []

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


def draw_boxes(img, detections):
    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = COLORS.get(d["class"], (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, max(0, y1 - 14)),
                  f'{d["class"]} {d["conf"]*100:.0f}%', fill=color)
    return img


# ---------- Logic ----------
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    A = (a[2]-a[0])*(a[3]-a[1]); B = (b[2]-b[0])*(b[3]-b[1])
    return inter / (A + B - inter + 1e-9)


def pt_in(px, py, b):
    return b[0] <= px <= b[2] and b[1] <= py <= b[3]


def analyze_image(image_path):
    inp, orig_img, scale, px, py = preprocess(image_path)
    raw = session.run([OUTPUT_NAME], {INPUT_NAME: inp})[0]
    detections = decode(raw, orig_img.width, orig_img.height, scale, px, py)

    persons = [d for d in detections if d["class"] == "Person"]
    ppes    = [d for d in detections if d["class"] != "Person"]

    persons.sort(key=lambda p: p["bbox"][0])
    for i, p in enumerate(persons):
        p["number"] = i + 1
        p["ppe"] = {}

    for ppe in ppes:
        cx = (ppe["bbox"][0] + ppe["bbox"][2]) / 2
        cy = (ppe["bbox"][1] + ppe["bbox"][3]) / 2
        best, bs = None, 0
        for p in persons:
            s = 1.0 if pt_in(cx, cy, p["bbox"]) else iou(ppe["bbox"], p["bbox"])
            if s > bs and s > 0.3:
                best, bs = p, s
        if best is not None:
            k = ppe["class"]
            if k not in best["ppe"] or ppe["conf"] > best["ppe"][k]["conf"]:
                best["ppe"][k] = ppe

    for p in persons:
        miss = [r_ for r_ in REQUIRED_PPE if r_ not in p["ppe"]]
        p["violations"] = miss
        p["status"] = "COMPLIANT" if not miss else "NON-COMPLIANT"

    result_img = draw_boxes(orig_img.copy(), detections)
    res_name = f"{uuid.uuid4().hex[:8]}.jpg"
    result_img.save(os.path.join(RESULT_DIR, res_name), quality=85)

    total = len(persons)
    comp = sum(1 for p in persons if p["status"] == "COMPLIANT")
    return {"persons": persons, "total_person": total, "total_compliant": comp,
            "total_violation": total - comp, "result_name": res_name}


# ---------- Static files dari /tmp ----------
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
        accuracy=ACCURACY, acc_avg=ACCURACY_AVG, result=result)


@app.route("/")
def home():
    return render(None)


@app.route("/analyze", methods=["POST"])
def analyze_upload():
    f = request.files.get("image")
    if not f or f.filename == "":
        return redirect(url_for("home"))
    ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
    name = f"{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)
    r = analyze_image(path)
    r["image_name"] = name
    r["area"] = "-"
    save_analysis("-", name, r["result_name"],
                  r["total_person"], r["total_compliant"], r["total_violation"])
    return render(r)


@app.route("/try/<sample_file>")
def try_sample(sample_file):
    src = os.path.join(SAMPLE_DIR, sample_file)
    if not os.path.exists(src):
        return redirect(url_for("home"))
    name = f"{uuid.uuid4().hex[:8]}.jpg"
    dst = os.path.join(UPLOAD_DIR, name)
    shutil.copy(src, dst)
    r = analyze_image(dst)
    r["image_name"] = name
    r["area"] = "Sample"
    save_analysis("Sample", name, r["result_name"],
                  r["total_person"], r["total_compliant"], r["total_violation"])
    return render(r)


# Vercel butuh objek `app`, bukan app.run()
if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)