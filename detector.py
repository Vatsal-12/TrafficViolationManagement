"""
detector.py  —  TrafficSentinel violation detector (headless mode)
Communicates with Streamlit via two files written to output_dir:
  - latest_frame.jpg  : most recent annotated frame (updated every N frames)
  - progress.json     : frame index, progress %, status, violations so far
"""

import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import defaultdict, deque
from typing import Optional, Callable
import os
import json
import threading

# ============================================================
# CONFIGURATION DEFAULTS
# ============================================================

DEFAULT_CONFIG = {
    "min_frames"         : 2,
    "conf_vehicle"       : 0.3,
    "conf_helmet"        : 0.20,
    "helmet_vote_frames" : 5,
    "helmet_iou_thresh"  : 0.10,
    "lane_p1"            : [500, 600],
    "lane_p2"            : [500, 100],
}

VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck
PREVIEW_EVERY   = 5               # save preview frame every N frames

# BGR color palette
COLOR_OK          = (0,   220,   0)
COLOR_WRONG_WAY   = (0,     0, 255)
COLOR_HELMET_OK   = (50,  205,  50)
COLOR_NO_HELMET   = (0,   140, 255)
COLOR_HEAD_REGION = (255, 255,   0)
COLOR_TRAJECTORY  = (0,   255, 255)
COLOR_BIKE        = (255, 100, 100)


# ============================================================
# HELPERS
# ============================================================

def format_video_time(frame_idx, fps):
    if fps <= 0:
        fps = 25.0
    total_ms  = int((frame_idx / fps) * 1000)
    ms        = total_ms % 1000
    total_sec = total_ms // 1000
    secs      = total_sec % 60
    mins      = (total_sec // 60) % 60
    hrs       = total_sec // 3600
    return "%02d:%02d:%02d.%03d" % (hrs, mins, secs, ms)


def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def get_head_crop(frame, px1, py1, px2, py2):
    h = py2 - py1
    w = px2 - px1
    if h < 40 or w < 10 or (w / max(h, 1)) > 3.0:
        return None, None
    pad_top = int(0.20 * h)
    crop_y1 = max(0, py1 - pad_top)
    crop_y2 = py1 + int(0.40 * h)
    crop_x1 = max(0, px1 - int(0.05 * w))
    crop_x2 = min(frame.shape[1], px2 + int(0.05 * w))
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
        return None, None
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2)


def cell_key(cx, cy, grid=40):
    return (cx // grid, cy // grid)


def draw_label_with_bg(frame, text, pos, font_scale, color, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 2, y - th - 4), (x + tw + 2, y + baseline), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


# ============================================================
# DETECTOR CLASS
# ============================================================

class ViolationDetector:

    def __init__(
        self,
        video_path         : str,
        output_dir         : str,
        config             : Optional[dict] = None,
        frame_callback     : Optional[Callable] = None,
        violation_callback : Optional[Callable] = None,
        stop_event         : Optional[threading.Event] = None,
    ):
        self.video_path         = video_path
        self.output_dir         = output_dir
        self.cfg                = dict(DEFAULT_CONFIG)
        if config:
            self.cfg.update(config)
        self.frame_callback     = frame_callback
        self.violation_callback = violation_callback
        self.stop_event         = stop_event if stop_event is not None else threading.Event()

        os.makedirs(output_dir, exist_ok=True)

        # Pre-compute lane direction vector
        p1  = self.cfg["lane_p1"]
        p2  = self.cfg["lane_p2"]
        dx  = p2[0] - p1[0]
        dy  = p2[1] - p1[1]
        mag = math.sqrt(dx**2 + dy**2) or 1.0
        self.lane_dx = dx / mag
        self.lane_dy = dy / mag

        # Per-track state
        self.track_history      = defaultdict(lambda: deque(maxlen=12))
        self.violation_counter  = defaultdict(int)
        self.helmet_vote_buffer = defaultdict(
            lambda: deque(maxlen=self.cfg["helmet_vote_frames"])
        )
        self.violation_log  = {}   # (track_id, vtype) -> entry dict
        self.snapshot_saved = set()

        # Public status
        self.total_frames = 0
        self.processed    = 0
        self.fps          = 25.0
        self.finished     = False
        self.error        = None

        print("[Detector] Loading models...")
        self.vehicle_model = YOLO("yolov8n.pt")
        self.helmet_model  = YOLO("helmet_model.pt")
        print("[Detector] Models ready.")

    # ── Violation recording ────────────────────────────────────

    def record_violation(self, track_id, vtype, frame_idx, frame_snap, box):
        key = (track_id, vtype)
        ts  = format_video_time(frame_idx, self.fps)
        if key not in self.violation_log:
            entry = {
                "track_id" : track_id,
                "type"     : vtype,
                "vtype"    : vtype,
                "timestamp": ts,
                "frame"    : frame_idx,
                "snapshot" : None,
                "has_image": False,
            }
            self.violation_log[key] = entry
            print("[VIOLATION] ID=%d  Type=%s  Time=%s  Frame=%d"
                  % (track_id, vtype, ts, frame_idx))
            if self.violation_callback:
                self.violation_callback(dict(entry))

        if key not in self.snapshot_saved:
            self.snapshot_saved.add(key)
            snap_path = self._save_snapshot(frame_snap, track_id, vtype, box, frame_idx)
            self.violation_log[key]["snapshot"]  = snap_path
            self.violation_log[key]["has_image"] = True

    def _save_snapshot(self, frame, track_id, vtype, box, frame_idx):
        snap        = frame.copy()
        x1, y1, x2, y2 = box
        color       = COLOR_WRONG_WAY if vtype == "WRONG_WAY" else COLOR_NO_HELMET
        cv2.rectangle(snap, (x1, y1), (x2, y2), color, 4)
        ts    = format_video_time(frame_idx, self.fps)
        label = "ID%d | %s | %s" % (track_id, vtype, ts)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(snap, (x1, y1 - th - 12), (x1 + tw + 8, y1), (0, 0, 0), -1)
        cv2.putText(snap, label, (x1 + 4, y1 - 6), font, 0.7, color, 2, cv2.LINE_AA)
        fname = "violation_ID%d_%s_f%d.jpg" % (track_id, vtype, frame_idx)
        fpath = os.path.join(self.output_dir, fname)
        cv2.imwrite(fpath, snap)
        print("[SNAPSHOT] Saved -> %s" % fpath)
        return fpath

    # ── Frame annotation ──────────────────────────────────────

    def _annotate_frame(self, frame, results, frame_idx):
        cfg = self.cfg
        P1  = tuple(cfg["lane_p1"])
        P2  = tuple(cfg["lane_p2"])

        cv2.arrowedLine(frame, P1, P2, COLOR_HEAD_REGION, 3, tipLength=0.03)
        draw_label_with_bg(frame, "Correct Direction",
                           (P2[0] - 10, P2[1] - 12), 0.5, COLOR_HEAD_REGION)

        person_boxes = []
        bike_boxes   = []

        if results.boxes.id is not None:
            boxes     = results.boxes.xyxy.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()

            # Pass 1 — bucket detections by class
            for box, cls_id, tid in zip(boxes, class_ids, track_ids):
                name = self.vehicle_model.names[int(cls_id)]
                b    = tuple(map(int, box))
                if name == "person":
                    person_boxes.append(b)
                elif name in ("motorcycle", "bicycle"):
                    bike_boxes.append((b, int(tid)))

            # Pass 2 — wrong-way detection for tracked vehicles
            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                if int(cls_id) not in VEHICLE_CLASSES:
                    continue
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                self.track_history[track_id].append((cx, cy))

                color = COLOR_OK
                label = "OK"

                if len(self.track_history[track_id]) >= 2:
                    px_prev, py_prev = list(self.track_history[track_id])[-2]
                    vdx = cx - px_prev
                    vdy = cy - py_prev
                    vmag = math.sqrt(vdx**2 + vdy**2)
                    if vmag > 0:
                        vdx /= vmag
                        vdy /= vmag
                        dot = vdx * self.lane_dx + vdy * self.lane_dy
                        draw_label_with_bg(frame, "dot:%.2f" % dot,
                                           (x1, y1 - 32), 0.40, (200, 200, 200))
                        if dot < -0.5:
                            self.violation_counter[track_id] += 1
                            if self.violation_counter[track_id] > cfg["min_frames"]:
                                color = COLOR_WRONG_WAY
                                label = "WRONG WAY | ID %d" % track_id
                                self.record_violation(
                                    track_id, "WRONG_WAY",
                                    frame_idx, frame, (x1, y1, x2, y2)
                                )
                        else:
                            self.violation_counter[track_id] = 0

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label_with_bg(frame, "ID%d %s" % (track_id, label),
                                   (x1, y1 - 10), 0.55, color)
                cv2.circle(frame, (cx, cy), 4,
                           COLOR_WRONG_WAY if color == COLOR_WRONG_WAY else (255, 0, 0), -1)
                pts = list(self.track_history[track_id])
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i-1], pts[i], COLOR_TRAJECTORY, 2)

            # Draw bike bounding boxes
            for (bx1, by1, bx2, by2), btid in bike_boxes:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_BIKE, 1)
                draw_label_with_bg(frame, "BK%d" % btid, (bx1, by1 - 6), 0.40, COLOR_BIKE)

        # ── Helmet detection ──────────────────────────────────
        for person_box in person_boxes:
            px1, py1, px2, py2 = person_box
            best_iou  = 0.0
            best_btid = None
            for (bbox, btid) in bike_boxes:
                iou = compute_iou(person_box, bbox)
                if iou > best_iou:
                    best_iou  = iou
                    best_btid = btid
            if best_iou <= cfg["helmet_iou_thresh"]:
                continue

            crop, crop_coords = get_head_crop(frame, px1, py1, px2, py2)
            if crop is None:
                continue

            res       = self.helmet_model(crop, conf=cfg["conf_helmet"], verbose=False)[0]
            best_conf = 0.0
            for hbox in res.boxes:
                cls_name = self.helmet_model.names[int(hbox.cls[0])].lower()
                conf     = float(hbox.conf[0])
                if "helmet" in cls_name and conf > best_conf:
                    best_conf = conf
            found = best_conf > cfg["conf_helmet"]

            hcx = (px1 + px2) // 2
            hcy = py1 + (py2 - py1) // 4
            key = cell_key(hcx, hcy)
            self.helmet_vote_buffer[key].append(1 if found else 0)
            votes    = list(self.helmet_vote_buffer[key])
            majority = (sum(votes) / len(votes)) >= 0.5

            if majority:
                h_color = COLOR_HELMET_OK
                h_label = "Helmet: YES (%.0f%%)" % (best_conf * 100)
                if best_btid:
                    h_label += " (BK%d)" % best_btid
            else:
                h_color = COLOR_NO_HELMET
                h_label = "NO HELMET"
                if best_btid:
                    h_label += " (BK%d)" % best_btid
                if best_btid is not None:
                    self.record_violation(
                        best_btid, "NO_HELMET",
                        frame_idx, frame, (px1, py1, px2, py2)
                    )

            cv2.rectangle(frame, (px1, py1), (px2, py2), h_color, 2)
            draw_label_with_bg(frame, h_label, (px1, py2 + 18), 0.55, h_color)
            if crop_coords:
                cx1, cy1, cx2, cy2 = crop_coords
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), COLOR_HEAD_REGION, 1)

        # ── Frame overlays ────────────────────────────────────
        ts = format_video_time(frame_idx, self.fps)
        draw_label_with_bg(frame, "Frame: %d  |  %s" % (frame_idx, ts),
                           (10, frame.shape[0] - 10), 0.45, (200, 200, 200))
        total_v = len(self.violation_log)
        draw_label_with_bg(frame, "Violations: %d" % total_v,
                           (frame.shape[1] - 200, 30), 0.6,
                           COLOR_WRONG_WAY if total_v > 0 else (200, 200, 200))
        return frame

    # ── Progress file ─────────────────────────────────────────

    def _write_progress(self, frame_idx, pct, status, error=""):
        """Write progress.json atomically. Streamlit polls this on each rerun."""
        data = {
            "frame"        : frame_idx,
            "total_frames" : self.total_frames,
            "pct"          : round(pct, 1),
            "status"       : status,
            "error"        : error,
            "violations"   : [
                {
                    "track_id" : v["track_id"],
                    "type"     : v["type"],
                    "vtype"    : v["type"],
                    "timestamp": v["timestamp"],
                    "frame"    : v["frame"],
                    "snapshot" : v.get("snapshot"),
                    "has_image": v.get("has_image", False),
                }
                for v in self.violation_log.values()
            ],
        }
        path = os.path.join(self.output_dir, "progress.json")
        tmp  = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    # ── Main loop ─────────────────────────────────────────────

    def run_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error = "Cannot open video: %s" % self.video_path
            print("[ERROR] %s" % self.error)
            self.finished = True
            self._write_progress(0, 0, "error", self.error)
            return

        self.fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("[Detector] FPS=%.1f  Total frames=%d" % (self.fps, self.total_frames))

        frame_idx    = 0
        preview_path = os.path.join(self.output_dir, "latest_frame.jpg")

        while cap.isOpened():
            if self.stop_event.is_set():
                print("[Detector] Stop requested.")
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx     += 1
            self.processed = frame_idx

            results = self.vehicle_model.track(
                frame, persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=self.cfg["conf_vehicle"]
            )[0]

            annotated = self._annotate_frame(frame, results, frame_idx)

            # Save preview frame to disk every N frames
            if frame_idx % PREVIEW_EVERY == 0 or frame_idx == 1:
                cv2.imwrite(preview_path, annotated)

            # Write progress (Streamlit reads this on each rerun)
            pct = (frame_idx / self.total_frames * 100.0) if self.total_frames > 0 else 0.0
            self._write_progress(frame_idx, pct, "running")

            if self.frame_callback:
                self.frame_callback(annotated, pct, frame_idx)

        cap.release()
        self.finished = True
        self._write_json_log()

        stopped_pct = (frame_idx / self.total_frames * 100.0) if self.total_frames else 0.0
        final_pct   = 100.0 if not self.stop_event.is_set() else stopped_pct
        self._write_progress(frame_idx, final_pct, "done")
        self._print_summary()

    # ── Persistence ───────────────────────────────────────────

    def _write_json_log(self):
        out = []
        for (tid, vtype), entry in sorted(self.violation_log.items()):
            out.append({
                "track_id" : entry["track_id"],
                "type"     : entry["type"],
                "vtype"    : entry["type"],
                "timestamp": entry["timestamp"],
                "frame"    : entry["frame"],
                "snapshot" : entry.get("snapshot"),
                "has_image": entry.get("has_image", False),
            })
        path = os.path.join(self.output_dir, "violations.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print("[Detector] Violation log -> %s" % path)

    def _print_summary(self):
        print("\n" + "="*60)
        print("VIOLATION SUMMARY")
        print("="*60)
        if self.violation_log:
            for (tid, vtype), e in sorted(self.violation_log.items()):
                print("  ID %4d  |  %-12s  |  %s  |  Frame %d"
                      % (tid, vtype, e["timestamp"], e["frame"]))
        else:
            print("  No violations detected.")
        print("="*60)

    @property
    def violations(self):
        return [dict(v, vtype=v["type"]) for v in self.violation_log.values()]