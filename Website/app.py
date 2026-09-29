import os
import shutil
import sqlite3
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
SAMPLE_DIR = os.path.join(BASE_DIR, "static", "samples")
DB_PATH    = os.path.join(BASE_DIR, "ppe.db")


def ensure_dir(p):
    if os.path.exists(p) and not os.path.isdir(p):
        os.remove(p)
    os.makedirs(p, exist_ok=True)


ensure_dir(UPLOAD_DIR)
ensure_dir(RESULT_DIR)
ensure_dir(SAMPLE_DIR)

CLASS_NAMES  = ["Mask", "Vest", "Person", "Gloves", "Hard_hat", "Safety_boots"]
REQUIRED_PPE = ["Mask", "Vest", "Gloves", "Hard_hat", "Safety_boots"]

SAMPLES = [
    {"label": "Sample 01", "file": "sample1.jpg"},
    {"label": "Sample 02", "file": "sample2.jpg"},
    {"label": "Sample 03", "file": "sample3.jpg"},
]

# ====== EDIT DATA TIM DI SINI ======
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

# ====== AKURASI MODEL (placeholder — edit sesuai hasil training) ======
ACCURACY = [
    {"kelas": "Hard_hat",     "sub": "Helm Keselamatan",   "p": 95.3, "r": 92.7, "map50": 97.0, "map": 76.3},
    {"kelas": "Vest",         "sub": "Rompi High-Vis",     "p": 91.8, "r": 91.4, "map50": 96.4, "map": 76.1},
    {"kelas": "Person",       "sub": "Kru Lapangan",       "p": 91.4, "r": 91.3, "map50": 97.0, "map": 76.0},
    {"kelas": "Safety_boots", "sub": "Sepatu Safety",      "p": 87.3, "r": 76.7, "map50": 85.2, "map": 48.6},
    {"kelas": "Gloves",       "sub": "Sarung Tangan",      "p": 80.9, "r": 85.9, "map50": 84.9, "map": 50.2},
    {"kelas": "Mask",         "sub": "Masker / Respirator","p": 80.0, "r": 80.2, "map50": 80.1, "map": 48.7},
]
ACCURACY_AVG = {"p": 87.8, "r": 86.4, "map50": 90.1, "map": 62.6}

model = YOLO(os.path.join(BASE_DIR, "best.pt"))


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


# ---------- Detection ----------
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    A = (a[2]-a[0])*(a[3]-a[1]); B = (b[2]-b[0])*(b[3]-b[1])
    return inter / (A + B - inter + 1e-9)


def pt_in(px, py, b):
    return b[0] <= px <= b[2] and b[1] <= py <= b[3]


def analyze_image(image_path):
    r = model.predict(image_path, conf=0.35, verbose=False)[0]
    persons, ppes = [], []
    for box in r.boxes:
        name = model.names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        det = {"class": name, "conf": float(box.conf), "bbox": [x1, y1, x2, y2]}
        (persons if name == "Person" else ppes).append(det)

    persons.sort(key=lambda p: p["bbox"][0])
    for i, p in enumerate(persons):
        p["number"] = i + 1; p["ppe"] = {}

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

    ann = r.plot()
    res_img = Image.fromarray(ann[:, :, ::-1])
    res_name = f"{uuid.uuid4().hex[:8]}.jpg"
    res_img.save(os.path.join(RESULT_DIR, res_name), quality=88)

    total = len(persons)
    comp = sum(1 for p in persons if p["status"] == "COMPLIANT")
    return {"persons": persons, "total_person": total, "total_compliant": comp,
            "total_violation": total - comp, "result_name": res_name}


# ---------- Routes (semua render index.html) ----------
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
    save_analysis("-", name, r["result_name"], r["total_person"], r["total_compliant"], r["total_violation"])
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
    r["image_name"] = name; r["area"] = "Sample"
    save_analysis("Sample", name, r["result_name"], r["total_person"], r["total_compliant"], r["total_violation"])
    return render(r)


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)