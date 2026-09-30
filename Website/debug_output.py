import onnxruntime as ort
import numpy as np
from PIL import Image

session = ort.InferenceSession("best_model.onnx")
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name

print("Input  shape:", session.get_inputs()[0].shape)
print("Output shape:", session.get_outputs()[0].shape)

# Load sample
img = Image.open("static/samples/sample1.jpg").convert("RGB")
ow, oh = img.size
print(f"\nImage: {ow}x{oh}")

# Preprocess sama seperti app.py
scale = min(640/ow, 640/oh)
nw, nh = int(ow*scale), int(oh*scale)
px, py = (640-nw)//2, (640-nh)//2
canvas = Image.new("RGB", (640,640), (0,0,0))
canvas.paste(img.resize((nw, nh)), (px, py))
arr = np.array(canvas, dtype=np.float32) / 255.0
arr = arr.transpose(2,0,1)[None]

raw = session.run([out_name], {inp_name: arr})[0]
print(f"\nRaw shape: {raw.shape}")
preds = raw[0]
print(f"preds shape: {preds.shape}")

# Cek nilai numerik di beberapa baris pertama
print("\n--- Sample nilai raw (5 kolom pertama) ---")
if preds.shape[0] < preds.shape[1]:
    print("Format: [4+nc, N] (transposed)")
    print("Baris 0-3 (bbox):")
    for i in range(4):
        print(f"  Row {i}: {preds[i, :5]}")
    print(f"\nBaris 4-8 (class scores):")
    for i in range(4, min(9, preds.shape[0])):
        print(f"  Row {i}: {preds[i, :5]}")
    
    # Ambil 1 sample kolom, tampilkan semua nilai
    print(f"\n--- Kolom pertama (semua nilai) ---")
    print(preds[:, 0])
    
    # Range nilai bbox
    bbox_vals = preds[:4, :]
    print(f"\nBBox values range: min={bbox_vals.min():.3f}, max={bbox_vals.max():.3f}")
    print(f"Kalau max < 1.5 → normalized (perlu × 640)")
    print(f"Kalau max 100-700 → pixel space 640 (OK)")
    print(f"Kalau max > 1000 → pixel space gambar asli")
else:
    print("Format: [N, 4+nc] (non-transposed)")
    print("Kolom 0-3 (bbox):", preds[:5, :4])
    bbox_vals = preds[:, :4]
    print(f"BBox values range: min={bbox_vals.min():.3f}, max={bbox_vals.max():.3f}")