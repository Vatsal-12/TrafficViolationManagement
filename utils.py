"""
utils.py  —  TrafficSentinel shared utilities
"""

import os
import re
import json
import glob

import numpy as np
import cv2
from PIL import Image

# ── Chaotic encryption helpers (inlined from Key_Generation + Selective_encryption_decryption) ──

def _generate_chaotic_sequence(a, b, c, d, x0, y0, z0, steps, dt):
    x, y, z = x0, y0, z0
    X = [x]
    for _ in range(steps):
        dx = a * x - b * y * z
        dy = x * z - c * y
        dz = x - d * z + x * y
        x += dx * dt
        y += dy * dt
        z += dz * dt
        X.append(x)
    return np.array(X)


def _generate_key_stream(chaotic_signal, image_size, discard_steps):
    if len(chaotic_signal) < image_size + discard_steps:
        raise ValueError("Chaotic signal too short for requested image size.")
    raw = chaotic_signal[discard_steps: image_size + discard_steps]
    scaled = raw * 1e8
    return np.mod(np.floor(scaled).astype(np.uint64), 256).astype(np.uint8)


def build_global_key_stream_for_image(img_shape, chaotic_params):
    H, W = img_shape[0], img_shape[1]
    channels    = 1 if len(img_shape) == 2 else img_shape[2]
    total_bytes = H * W * channels
    discard     = int(chaotic_params.get("discard_steps", 15000))   # must be int
    total_steps = total_bytes + discard + 10
    X = _generate_chaotic_sequence(
        float(chaotic_params["a"]), float(chaotic_params["b"]),
        float(chaotic_params["c"]), float(chaotic_params["d"]),
        float(chaotic_params["x0"]), float(chaotic_params["y0"]), float(chaotic_params["z0"]),
        total_steps, float(chaotic_params["dt"]),
    )
    return _generate_key_stream(X, total_bytes, discard)


def _xor_region(image_array, global_key_stream, bbox):
    """XOR-encrypt (or decrypt — same op) a bbox region of a numpy image."""
    enc = image_array.copy()
    xmin, ymin, xmax, ymax = (int(v) for v in bbox)
    region = enc[ymin:ymax, xmin:xmax]
    if region.size == 0:
        return enc
    region_len = region.size
    max_offset = len(global_key_stream) - region_len
    if max_offset <= 0:
        raise ValueError("Global key stream too small for region.")
    offset = (
        (xmin * 73856093) ^
        (ymin * 19349663) ^
        (xmax * 83492791) ^
        (ymax * 15485863)
    ) % max_offset
    key_segment = global_key_stream[offset: offset + region_len].reshape(region.shape)
    enc[ymin:ymax, xmin:xmax] = np.bitwise_xor(
        region.astype(np.uint8), key_segment.astype(np.uint8)
    )
    return enc


# ── Region detectors ──────────────────────────────────────────────────────────

def _detect_plates_roboflow(image_path: str, api_key: str) -> list:
    """Return list of (xmin,ymin,xmax,ymax) tuples for licence plates."""
    try:
        from inference_sdk import InferenceHTTPClient
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        result = client.infer(image_path, model_id="license-plate-detector-ogxxg/1")
        boxes = []
        for p in result.get("predictions", []):
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            boxes.append((int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)))
        return boxes
    except Exception as e:
        print("[encrypt] Roboflow plate detection failed: %s" % e)
        return []


_FACE_CASCADE = None

def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _FACE_CASCADE


def _detect_faces(img_np: np.ndarray) -> list:
    """Return list of (xmin,ymin,xmax,ymax) expanded face bboxes."""
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if not len(faces):
        return []
    img_h, img_w = img_np.shape[:2]
    boxes = []
    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        nw, nh = int(w * 1.6), int(h * 1.6)
        boxes.append((
            max(0, cx - nw // 2), max(0, cy - nh // 2),
            min(img_w, cx + nw // 2), min(img_h, cy + nh // 2),
        ))
    return boxes


# ── Public encrypt / decrypt API ─────────────────────────────────────────────

def _serialisable_params(chaotic_params: dict) -> dict:
    """Return a JSON-safe copy of chaotic_params with correct types preserved."""
    result = {}
    for k, v in chaotic_params.items():
        if k == "discard_steps":
            result[k] = int(v)          # must always be int
        else:
            result[k] = float(v)        # all other params are floats
    return result

DEFAULT_CHAOTIC_PARAMS = {
    "a": 5.2, "b": 6.0, "c": 10.0, "d": 5.0,
    "x0": 1.0, "y0": 1.0, "z0": 0.0,
    "dt": 0.005, "discard_steps": 15000,
}


def encrypted_path(image_path: str) -> str:
    """Return the path used for the encrypted copy of a snapshot."""
    base, ext = os.path.splitext(image_path)
    return base + ".enc" + ext


def encrypt_snapshot(image_path: str, chaotic_params: dict, roboflow_api_key: str = "") -> dict:
    """
    Detect number plates + faces in the snapshot at image_path,
    XOR-encrypt those regions and save to a NEW side-by-side copy
    (<name>.enc.jpg).  The original file is never modified.

    Returns a result dict:
        {"ok": True,  "plates": N, "faces": N}
        {"ok": False, "error": "..."}
    """
    try:
        img_np = np.array(Image.open(image_path).convert("RGB"))
        key_stream = build_global_key_stream_for_image(img_np.shape, chaotic_params)

        # Detect regions
        plate_boxes = _detect_plates_roboflow(image_path, roboflow_api_key) if roboflow_api_key else []
        face_boxes  = _detect_faces(img_np)
        regions     = plate_boxes + face_boxes

        if not regions:
            return {"ok": True, "plates": 0, "faces": len(face_boxes), "warning": "no_regions"}

        enc = img_np.copy()
        for bbox in regions:
            enc = _xor_region(enc, key_stream, bbox)

        # Save the ORIGINAL pixels as <n>.enc.jpg — this is the backup used by decrypt
        enc_file = encrypted_path(image_path)
        cv2.imwrite(enc_file, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # Overwrite the original file with encrypted pixels so opening it
        # locally shows the scrambled (protected) image
        cv2.imwrite(image_path, cv2.cvtColor(enc, cv2.COLOR_RGB2BGR))

        # Write sidecar so decrypt knows exact bboxes + params used
        # Cast every coord to plain Python int — numpy int32/int64 are not JSON-serialisable
        sidecar = image_path + ".enc.json"
        with open(sidecar, "w") as f:
            json.dump({
                "regions"       : [[int(v) for v in b] for b in regions],
                "plates"        : int(len(plate_boxes)),
                "faces"         : int(len(face_boxes)),
                "chaotic_params": _serialisable_params(chaotic_params),
                "enc_file"      : enc_file,
            }, f)

        return {"ok": True, "plates": len(plate_boxes), "faces": len(face_boxes)}

    except Exception as e:
        return {"ok": False, "error": str(e)}


def decrypt_snapshot(image_path: str, chaotic_params: dict, roboflow_api_key: str = "") -> dict:
    """
    Restore the original snapshot by copying the clean backup (.enc.jpg) back
    over the original path, then removing the sidecar.
    No XOR is needed — .enc.jpg already holds the untouched original pixels.
    """
    sidecar = image_path + ".enc.json"
    if not os.path.exists(sidecar):
        return {"ok": False, "error": "No encryption record found — encrypt first."}

    try:
        with open(sidecar) as f:
            meta = json.load(f)

        enc_file = meta.get("enc_file") or encrypted_path(image_path)

        if not os.path.exists(enc_file):
            # Backup already gone — clean up stale sidecar
            os.remove(sidecar)
            return {"ok": True, "plates": meta.get("plates", 0), "faces": meta.get("faces", 0)}

        # .enc.jpg holds the original clean pixels — just move it back
        os.replace(enc_file, image_path)

        # Remove the sidecar so is_encrypted() returns False
        os.remove(sidecar)

        return {"ok": True, "plates": meta.get("plates", 0), "faces": meta.get("faces", 0)}

    except Exception as e:
        return {"ok": False, "error": str(e)}


def is_encrypted(image_path: str) -> bool:
    """True if a sidecar encryption record exists for this snapshot."""
    return os.path.exists(image_path + ".enc.json")


def get_display_path(image_path: str) -> str:
    """Return the path that should be shown in the UI.
    The original .jpg always holds the current pixels (encrypted or not),
    so we always display it directly."""
    return image_path


def scan_violation_images(directory: str) -> list[dict]:
    """
    Scan directory for violation_*.jpg snapshots and parse metadata
    from filenames: violation_ID{id}_{TYPE}_f{frame}.jpg
    """
    pattern = os.path.join(directory, "violation_*.jpg")
    # Exclude encrypted copies (violation_*.enc.jpg) — those are side-car files, not originals
    files = sorted(fp for fp in glob.glob(pattern) if not fp.endswith(".enc.jpg"))
    items = []
    for fp in files:
        fname = os.path.basename(fp)
        m = re.match(r"violation_ID(\d+)_([A-Z_]+)_f(\d+)\.jpg", fname)
        if m:
            items.append({
                "path"     : fp,
                "filename" : fname,
                "track_id" : int(m.group(1)),
                "vtype"    : m.group(2),
                "frame"    : int(m.group(3)),
            })
        else:
            items.append({
                "path"     : fp,
                "filename" : fname,
                "track_id" : None,
                "vtype"    : "UNKNOWN",
                "frame"    : None,
            })
    return items


def load_violations_json(directory: str) -> list[dict]:
    """Load violations.json written by the detector at end of run."""
    path = os.path.join(directory, "violations.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    # Normalise: ensure 'vtype' key mirrors 'type'
    for entry in data:
        entry.setdefault("vtype", entry.get("type", "UNKNOWN"))
    return data


def merge_violations(json_violations: list[dict], images: list[dict]) -> list[dict]:
    """
    Merge JSON log entries with snapshot image paths.
    Prioritises JSON for timestamps; images fill in snapshot paths.
    """
    # Build lookup: (track_id, vtype) → image path
    img_lookup = {}
    for img in images:
        if img["track_id"] is not None:
            img_lookup[(img["track_id"], img["vtype"])] = img["path"]

    seen   = set()
    merged = []

    for entry in json_violations:
        key = (entry["track_id"], entry.get("vtype", entry.get("type", "")))
        if key in seen:
            continue
        seen.add(key)
        entry = dict(entry)
        entry["vtype"]    = key[1]
        entry["has_image"] = key in img_lookup
        entry["snapshot"] = img_lookup.get(key, entry.get("snapshot"))
        merged.append(entry)

    # Add any images not in JSON
    for img in images:
        if img["track_id"] is None:
            continue
        key = (img["track_id"], img["vtype"])
        if key not in seen:
            seen.add(key)
            merged.append({
                "track_id" : img["track_id"],
                "type"     : img["vtype"],
                "vtype"    : img["vtype"],
                "timestamp": "—",
                "frame"    : img["frame"],
                "has_image": True,
                "snapshot" : img["path"],
            })

    merged.sort(key=lambda x: x.get("frame") or 0)
    return merged


def badge_class(vtype: str) -> str:
    return "badge-ww" if vtype == "WRONG_WAY" else "badge-nh"


def badge_label(vtype: str) -> str:
    return "WRONG WAY" if vtype == "WRONG_WAY" else "NO HELMET"


def accent_color(vtype: str) -> str:
    return "#ff3b30" if vtype == "WRONG_WAY" else "#ff9500"