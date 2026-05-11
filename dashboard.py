"""
dashboard.py  —  TrafficViolationDetection · Streamlit UI
Run:  streamlit run dashboard.py

Architecture:
  - Detector runs in a background thread and writes two files to output_dir:
      latest_frame.jpg  : most recent annotated frame
      progress.json     : frame index, %, status, violations list
  - Dashboard polls those files on each st.rerun() (every 1s while running)
  - No queues, no shared memory — fully safe with Streamlit's execution model
"""

import streamlit as st
import threading
import time
import os
import json
import tempfile
import cv2
import numpy as np
from PIL import Image as PILImage
from datetime import datetime

from detector import ViolationDetector
from utils import (
    scan_violation_images,
    load_violations_json,
    merge_violations,
    badge_class,
    badge_label,
    accent_color,
    encrypt_snapshot,
    decrypt_snapshot,
    is_encrypted,
    get_display_path,
    DEFAULT_CHAOTIC_PARAMS,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficSentinel",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;900&family=Barlow:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #f0f2f5 !important; color: #1a2332 !important;
    font-family: 'Barlow', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
[data-testid="block-container"] { padding: 1.6rem 2.2rem 3rem !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { display: block; background:#ffffff !important; border-right:1px solid #d8e2ee !important; }

/* scanline */
[data-testid="stAppViewContainer"]::before {
    content:''; position:fixed; top:0;left:0;right:0;bottom:0;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.02) 2px,rgba(0,0,0,0.02) 4px);
    pointer-events:none; z-index:9999;
}

/* masthead */
.masthead { display:flex; align-items:flex-end; justify-content:space-between;
            padding-bottom:1.1rem; border-bottom:1px solid #d0d8e4; margin-bottom:1.6rem; }
.masthead-brand { font-family:'Barlow Condensed',sans-serif; font-size:2.4rem; font-weight:900;
                  letter-spacing:0.14em; text-transform:uppercase; color:#1a2332; line-height:1; }
.masthead-brand span { color:#ff3b30; }
.masthead-sub { font-family:'Share Tech Mono',monospace; font-size:0.65rem; letter-spacing:0.22em;
                color:#7a90a8; text-transform:uppercase; margin-top:0.3rem; }
.masthead-right { font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                  color:#8a9db0; text-align:right; line-height:2; }
.live-dot { display:inline-block; width:7px; height:7px; background:#ff3b30; border-radius:50%;
            animation:blink 1.1s infinite; margin-right:5px; vertical-align:middle; }
@keyframes blink { 0%,100%{opacity:1}50%{opacity:0.1} }

/* stat cards */
.stat-row { display:grid; grid-template-columns:repeat(5,1fr); gap:0.85rem; margin-bottom:1.5rem; }
.stat-card { background:#ffffff; border:1px solid #d8e2ee; border-top:2px solid var(--ac);
             padding:0.9rem 1.1rem; position:relative; overflow:hidden; }
.stat-card::after { content:''; position:absolute; bottom:0;right:0; width:55px;height:55px;
    background:radial-gradient(circle at bottom right,rgba(var(--ac-rgb),0.08),transparent 70%); }
.stat-value { font-family:'Barlow Condensed',sans-serif; font-size:2.1rem; font-weight:700;
              color:var(--ac-color,#1a2332); line-height:1; }
.stat-label { font-family:'Share Tech Mono',monospace; font-size:0.58rem; letter-spacing:0.15em;
              color:#9aaabb; text-transform:uppercase; margin-top:0.3rem; }
.stat-icon  { font-size:1.2rem; line-height:1; margin-bottom:0.4rem; opacity:0.7; }

/* section header */
.sec-hdr { font-family:'Share Tech Mono',monospace; font-size:0.62rem; letter-spacing:0.22em;
           text-transform:uppercase; color:#9aaabb; border-left:2px solid #ff3b30;
           padding-left:0.55rem; margin-bottom:0.85rem; margin-top:0.15rem; }

/* progress */
.prog-track { height:3px; background:#d8e2ee; margin-top:0.4rem; border-radius:1px; overflow:hidden; }
.prog-fill  { height:100%; background:#ff3b30; border-radius:1px; }

/* violation table */
.vtable { background:#ffffff; border:1px solid #d8e2ee; overflow:hidden; }
.vt-head { display:grid; grid-template-columns:55px 110px 150px 110px 80px;
           padding:0.5rem 0.9rem; background:#f5f7fa; border-bottom:1px solid #d8e2ee; }
.vt-head span { font-family:'Share Tech Mono',monospace; font-size:0.55rem;
                letter-spacing:0.18em; color:#9aaabb; text-transform:uppercase; }
.vt-row  { display:grid; grid-template-columns:55px 110px 150px 110px 80px;
           padding:0.6rem 0.9rem; border-bottom:1px solid #eaeff5; align-items:center;
           border-left:2px solid var(--row-ac,transparent); transition:background 0.12s; }
.vt-row:last-child { border-bottom:none; }
.vt-row:hover { background:#f5f7fa; }
.cell-id { font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#5a7080; }
.badge { display:inline-block; padding:2px 7px; border-radius:2px;
         font-family:'Share Tech Mono',monospace; font-size:0.58rem;
         letter-spacing:0.06em; text-transform:uppercase; font-weight:600; }
.badge-ww { background:rgba(255,59,48,.12); color:#ff3b30; border:1px solid rgba(255,59,48,.28); }
.badge-nh { background:rgba(255,149,0,.12); color:#ff9500; border:1px solid rgba(255,149,0,.28); }
.cell-ts { font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#6a8090; }
.cell-fr { font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#8a9db0; }

/* timeline */
.tl-item { display:flex; gap:0.85rem; padding:0.6rem 0;
           border-bottom:1px solid #eaeff5; align-items:flex-start; }
.tl-item:last-child { border-bottom:none; }
.tl-dot { width:7px; height:7px; border-radius:50%; margin-top:4px; flex-shrink:0; }
.tl-ts  { font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:#9aaabb; min-width:80px; }
.tl-msg { font-size:0.74rem; color:#4a6070; line-height:1.4; }
.tl-msg strong { color:#1a2332; font-weight:500; }

/* sev bars */
.sev-wrap { background:#ffffff; border:1px solid #d8e2ee; padding:0.85rem 1.1rem; }
.sev-lbl { display:flex; justify-content:space-between; font-family:'Share Tech Mono',monospace;
           font-size:0.58rem; color:#9aaabb; text-transform:uppercase; letter-spacing:0.12em;
           margin-bottom:0.4rem; }
.sev-track { height:3px; background:#d8e2ee; border-radius:1px; overflow:hidden; }
.sev-fill  { height:100%; border-radius:1px; }

/* status chips */
.status-chip { display:inline-flex; align-items:center; gap:0.4rem; padding:3px 10px;
               border-radius:2px; font-family:'Share Tech Mono',monospace; font-size:0.6rem;
               letter-spacing:0.1em; text-transform:uppercase; }
.chip-idle    { background:rgba(180,200,220,.4);  color:#4a6070; border:1px solid #c0d0e0; }
.chip-running { background:rgba(255,59,48,.1); color:#ff3b30; border:1px solid rgba(255,59,48,.3); }
.chip-done    { background:rgba(0,200,80,.1);  color:#00c850; border:1px solid rgba(0,200,80,.3); }
.chip-error   { background:rgba(255,200,0,.1); color:#ffc800; border:1px solid rgba(255,200,0,.3); }

/* viewer */
.viewer-meta { background:#ffffff; border:1px solid #d8e2ee; padding:1.1rem 1.2rem; }
.viewer-id   { font-family:'Barlow Condensed',sans-serif; font-size:2rem; font-weight:700;
               color:#1a2332; line-height:1; margin-bottom:0.5rem; }

/* empty state */
.empty { text-align:center; padding:2.5rem 1rem; color:#b0c0d0;
         font-family:'Share Tech Mono',monospace; font-size:0.72rem; letter-spacing:0.1em; }
.empty-icon { font-size:2rem; margin-bottom:0.6rem; opacity:0.25; }

/* streamlit overrides */
div[data-testid="stButton"] button {
    background:#f0f4f8 !important; border:1px solid #c8d8e8 !important;
    color:#5a7080 !important; border-radius:2px !important;
    font-family:'Share Tech Mono',monospace !important; font-size:0.6rem !important;
    letter-spacing:0.15em !important; text-transform:uppercase !important;
    transition:all 0.15s !important; padding:0.4rem 0.9rem !important;
}
div[data-testid="stButton"] button:hover {
    background:#ff3b30 !important; border-color:#ff3b30 !important; color:#fff !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] > div input {
    background:#ffffff !important; border:1px solid #c8d8e8 !important;
    color:#1a2332 !important; border-radius:2px !important;
    font-family:'Share Tech Mono',monospace !important; font-size:0.72rem !important;
}
label[data-testid="stWidgetLabel"] p {
    font-family:'Share Tech Mono',monospace !important; font-size:0.6rem !important;
    letter-spacing:0.15em !important; color:#8a9db0 !important; text-transform:uppercase !important;
}
div[data-testid="stFileUploader"] {
    background:#f8fafc !important; border:1px dashed #c8d8e8 !important; border-radius:2px !important;
}
div[data-testid="stExpander"] details { background:#f8fafc !important; border:1px solid #d8e2ee !important; }
div[data-testid="stExpander"] summary {
    font-family:'Share Tech Mono',monospace !important; font-size:0.62rem !important;
    letter-spacing:0.15em !important; color:#8a9db0 !important; text-transform:uppercase !important;
}
[data-testid="stTabs"] [role="tablist"] { gap:0 !important; border-bottom:1px solid #d8e2ee !important; background:transparent !important; }
[data-testid="stTabs"] [role="tab"] {
    font-family:'Share Tech Mono',monospace !important; font-size:0.62rem !important;
    letter-spacing:0.18em !important; text-transform:uppercase !important;
    color:#8a9db0 !important; background:transparent !important;
    border:none !important; border-bottom:2px solid transparent !important; padding:0.5rem 1.3rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#e8edf2 !important; border-bottom-color:#ff3b30 !important; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top:1rem !important; }
[data-testid="stHorizontalBlock"] { gap:1rem !important; }
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:#f0f2f5; }
::-webkit-scrollbar-thumb { background:#c0d0e0; border-radius:2px; }
hr { border-color:#e0e8f0 !important; margin:1.3rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
def ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

ss("run_status",     "idle")   # idle | running | done | error
ss("output_dir",     None)
ss("tmp_video_path", None)
ss("selected_img",   None)
ss("gal_filter",     "ALL")
ss("log_filter",     "ALL")
ss("stop_event",     None)
ss("thread",         None)
ss("enc_status",     {})       # filename → "encrypted" | "decrypted" | "error:<msg>"
ss("roboflow_key",   "RXFxV1D39OFWVkf28guS")
ss("snap_jump_to",   None)     # (track_id, vtype) → jump to this snapshot in gallery tab


# ─────────────────────────────────────────────────────────────
# HELPERS: poll disk files written by detector thread
# ─────────────────────────────────────────────────────────────
def read_progress(output_dir):
    """Read progress.json. Returns dict or None."""
    if not output_dir:
        return None
    path = os.path.join(output_dir, "progress.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def read_latest_frame(output_dir):
    """Read latest_frame.jpg as PIL Image. Returns None if not available."""
    if not output_dir:
        return None
    path = os.path.join(output_dir, "latest_frame.jpg")
    if not os.path.exists(path):
        return None
    try:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return None
        return PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# MASTHEAD
# ─────────────────────────────────────────────────────────────
status    = st.session_state.run_status
chip_cls  = {"idle":"chip-idle","running":"chip-running","done":"chip-done","error":"chip-error"}.get(status,"chip-idle")
chip_lbl  = {"idle":"● IDLE","running":"▶ PROCESSING","done":"✓ COMPLETE","error":"⚠ ERROR"}.get(status,"IDLE")
now_str   = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

st.markdown(f"""
<div class="masthead">
  <div>
    <div class="masthead-brand">Traffic<span>Sentinel</span></div>
    <div class="masthead-sub">Automated Violation Detection · YOLOv8 + ByteTrack</div>
  </div>
  <div class="masthead-right">
    <span class="live-dot"></span>MONITOR ACTIVE<br>{now_str}<br>
    <span class="status-chip {chip_cls}">{chip_lbl}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — ENCRYPTION PARAMETERS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔐 Chaotic Encryption")
    st.markdown('<span style="font-family:monospace;font-size:0.62rem;color:#8a9db0;">PARAMETERS</span>',
                unsafe_allow_html=True)
    _cp = DEFAULT_CHAOTIC_PARAMS
    enc_params = {
        "a"            : st.number_input("a",             value=_cp["a"],             format="%.6f", key="ep_a"),
        "b"            : st.number_input("b",             value=_cp["b"],             format="%.6f", key="ep_b"),
        "c"            : st.number_input("c",             value=_cp["c"],             format="%.6f", key="ep_c"),
        "d"            : st.number_input("d",             value=_cp["d"],             format="%.6f", key="ep_d"),
        "x0"           : st.number_input("x0",            value=_cp["x0"],            format="%.6f", key="ep_x0"),
        "y0"           : st.number_input("y0",            value=_cp["y0"],            format="%.6f", key="ep_y0"),
        "z0"           : st.number_input("z0",            value=_cp["z0"],            format="%.6f", key="ep_z0"),
        "dt"           : st.number_input("dt",            value=_cp["dt"],            format="%.6f", key="ep_dt"),
        "discard_steps": st.number_input("discard_steps", value=_cp["discard_steps"], step=1,        key="ep_ds"),
    }
    st.markdown("---")
    st.markdown('<span style="font-family:monospace;font-size:0.62rem;color:#8a9db0;">ROBOFLOW API KEY</span>',
                unsafe_allow_html=True)
    st.session_state.roboflow_key = st.text_input(
        "Roboflow key", value=st.session_state.roboflow_key,
        type="password", label_visibility="collapsed", key="rb_key_input"
    )


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_run, tab_results, tab_gallery, tab_analysis, tab_config = st.tabs([
    "▶  RUN DETECTOR", "📋  VIOLATIONS LOG", "📷  SNAPSHOTS", "📊  ANALYSIS", "⚙  CONFIGURATION",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — RUN DETECTOR
# ══════════════════════════════════════════════════════════════
with tab_run:

    up_col, ctrl_col = st.columns([2, 1], gap="large")

    with up_col:
        st.markdown('<div class="sec-hdr">VIDEO INPUT</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload video", type=["mp4","avi","mov","mkv"],
            label_visibility="collapsed"
        )
        if uploaded:
            # Save to a persistent temp file (only when new upload)
            cur_name = os.path.basename(st.session_state.tmp_video_path or "")
            if cur_name != uploaded.name or not os.path.exists(st.session_state.tmp_video_path or ""):
                suffix = os.path.splitext(uploaded.name)[-1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.read())
                tmp.flush(); tmp.close()
                st.session_state.tmp_video_path = tmp.name

            st.markdown(f"""
            <div style="background:#f0f4f8;border:1px solid #d0dcea;padding:0.6rem 1rem;
                        font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:#5a7080;
                        letter-spacing:0.1em;margin-top:0.5rem;">
              ✓ &nbsp; {uploaded.name} &nbsp;·&nbsp; {uploaded.size/1024/1024:.1f} MB
            </div>""", unsafe_allow_html=True)

    with ctrl_col:
        st.markdown('<div class="sec-hdr">CONTROLS</div>', unsafe_allow_html=True)
        can_run = (
            st.session_state.tmp_video_path is not None
            and os.path.exists(st.session_state.tmp_video_path or "")
            and st.session_state.run_status != "running"
        )

        if st.button("▶  START DETECTION", disabled=not can_run, key="btn_start"):
            out_dir = os.path.join(os.getcwd(), "violation_snapshots")
            os.makedirs(out_dir, exist_ok=True)
            st.session_state.output_dir  = out_dir
            st.session_state.run_status  = "running"

            stop_ev = threading.Event()
            st.session_state.stop_event  = stop_ev

            cfg = {
                "min_frames"         : st.session_state.get("cfg_min_frames", 2),
                "conf_vehicle"       : st.session_state.get("cfg_conf_vehicle", 0.30),
                "conf_helmet"        : st.session_state.get("cfg_conf_helmet", 0.20),
                "helmet_vote_frames" : st.session_state.get("cfg_vote_frames", 5),
                "helmet_iou_thresh"  : st.session_state.get("cfg_iou_thresh", 0.10),
                "lane_p1"            : [st.session_state.get("cfg_p1x",500), st.session_state.get("cfg_p1y",600)],
                "lane_p2"            : [st.session_state.get("cfg_p2x",500), st.session_state.get("cfg_p2y",100)],
            }

            # Capture into plain locals — background threads CANNOT access st.session_state
            _video_path = st.session_state.tmp_video_path
            _out_dir    = out_dir
            _cfg        = cfg
            _stop_ev    = stop_ev

            def run_thread():
                try:
                    det = ViolationDetector(
                        video_path  = _video_path,
                        output_dir  = _out_dir,
                        config      = _cfg,
                        stop_event  = _stop_ev,
                    )
                    det.run_video()
                except Exception as e:
                    err_data = {"frame":0,"total_frames":0,"pct":0,"status":"error",
                                "error":str(e),"violations":[]}
                    err_path = os.path.join(_out_dir, "progress.json")
                    with open(err_path, "w") as f:
                        json.dump(err_data, f)
                    print("[ERROR] %s" % e)

            t = threading.Thread(target=run_thread, daemon=True)
            t.start()
            st.session_state.thread = t
            time.sleep(0.5)   # give thread a moment to start
            st.rerun()

        if st.session_state.run_status == "running":
            if st.button("⏹  STOP", key="btn_stop"):
                if st.session_state.stop_event:
                    st.session_state.stop_event.set()
                st.session_state.run_status = "done"
                st.rerun()

        if st.session_state.run_status in ("done", "error"):
            if st.button("↺  NEW VIDEO", key="btn_reset"):
                if st.session_state.stop_event:
                    st.session_state.stop_event.set()
                st.session_state.run_status     = "idle"
                st.session_state.tmp_video_path = None
                st.session_state.output_dir     = None
                st.session_state.selected_img   = None
                st.rerun()

    st.markdown("---")

    # ── Live feed ──────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">LIVE DETECTION FEED</div>', unsafe_allow_html=True)

    prog = read_progress(st.session_state.output_dir)

    # Sync run_status from progress file (handles thread finishing)
    if prog and st.session_state.run_status == "running":
        if prog.get("status") in ("done", "error"):
            st.session_state.run_status = prog["status"]

    if st.session_state.run_status == "running":
        pct       = prog["pct"]       if prog else 0.0
        cur_frame = prog["frame"]     if prog else 0
        total_fr  = prog["total_frames"] if prog else 0

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;
                    font-family:'Share Tech Mono',monospace;font-size:0.58rem;
                    color:#9aaabb;margin-bottom:0.3rem;">
          <span>PROCESSING</span>
          <span>Frame {cur_frame} / {total_fr} &nbsp;·&nbsp; {pct:.1f}%</span>
        </div>
        <div class="prog-track"><div class="prog-fill" style="width:{pct:.1f}%;"></div></div>
        """, unsafe_allow_html=True)

        feed_col, stats_col = st.columns([3, 1], gap="large")

        with feed_col:
            frame_img = read_latest_frame(st.session_state.output_dir)
            if frame_img:
                st.image(frame_img, use_container_width=True)
            else:
                st.markdown("""
                <div style="background:#f5f7fa;border:1px solid #d8e2ee;height:280px;
                            display:flex;align-items:center;justify-content:center;">
                  <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                               color:#b0c4d8;letter-spacing:0.15em;">WAITING FOR FIRST FRAME…</span>
                </div>""", unsafe_allow_html=True)

        with stats_col:
            viols = prog.get("violations", []) if prog else []
            ww = sum(1 for v in viols if v.get("vtype") == "WRONG_WAY")
            nh = sum(1 for v in viols if v.get("vtype") == "NO_HELMET")
            st.markdown(f"""
            <div style="display:flex;flex-direction:column;gap:0.7rem;">
              <div class="stat-card" style="--ac:#ff3b30;--ac-rgb:255,59,48;--ac-color:#ff3b30;">
                <div class="stat-icon">🚨</div>
                <div class="stat-value">{len(viols)}</div>
                <div class="stat-label">Violations Found</div>
              </div>
              <div class="stat-card" style="--ac:#ff3b30;--ac-rgb:255,59,48;--ac-color:#ff3b30;">
                <div class="stat-icon">↩️</div>
                <div class="stat-value">{ww}</div>
                <div class="stat-label">Wrong-Way</div>
              </div>
              <div class="stat-card" style="--ac:#ff9500;--ac-rgb:255,149,0;--ac-color:#ff9500;">
                <div class="stat-icon">⛑️</div>
                <div class="stat-value">{nh}</div>
                <div class="stat-label">No Helmet</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Auto-refresh every second
        time.sleep(1)
        st.rerun()

    elif st.session_state.run_status == "done":
        prog = read_progress(st.session_state.output_dir)
        vcount = len(prog.get("violations", [])) if prog else 0
        frame_img = read_latest_frame(st.session_state.output_dir)
        if frame_img:
            st.image(frame_img, use_container_width=True)
        st.markdown(f"""
        <div style="background:#f0fdf5;border:1px solid #c0e8d0;border-left:2px solid #00c850;
                    padding:0.8rem 1.2rem;font-family:'Share Tech Mono',monospace;
                    font-size:0.65rem;color:#1a6040;letter-spacing:0.1em;margin-top:0.8rem;">
          ✓ &nbsp; PROCESSING COMPLETE &nbsp;·&nbsp; {vcount} VIOLATION(S) DETECTED
          &nbsp;·&nbsp; VIEW RESULTS IN TABS ABOVE →
        </div>""", unsafe_allow_html=True)

    elif st.session_state.run_status == "error":
        err = (prog.get("error","") if prog else "") or "Unknown error"
        st.markdown(f"""
        <div style="background:#fffbf0;border:1px solid #e8d890;border-left:2px solid #ffc800;
                    padding:0.8rem 1.2rem;font-family:'Share Tech Mono',monospace;
                    font-size:0.65rem;color:#806020;letter-spacing:0.1em;">
          ⚠ &nbsp; ERROR: {err}
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty">
          <div class="empty-icon">🎬</div>
          UPLOAD A VIDEO AND PRESS START DETECTION
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — VIOLATIONS LOG
# ══════════════════════════════════════════════════════════════
with tab_results:

    # Pull violations from progress.json (live during run, or violations.json after)
    prog = read_progress(st.session_state.output_dir)
    if prog:
        violations = prog.get("violations", [])
    elif st.session_state.output_dir:
        violations = load_violations_json(st.session_state.output_dir)
    else:
        violations = []

    total_v    = len(violations)
    ww_count   = sum(1 for v in violations if v.get("vtype") == "WRONG_WAY")
    nh_count   = sum(1 for v in violations if v.get("vtype") == "NO_HELMET")
    unique_ids = len(set(v["track_id"] for v in violations))
    cur_frame  = prog["frame"] if prog else 0

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card" style="--ac:#ff3b30;--ac-rgb:255,59,48;--ac-color:#ff3b30;">
        <div class="stat-icon">🚨</div><div class="stat-value">{total_v}</div>
        <div class="stat-label">Total Violations</div>
      </div>
      <div class="stat-card" style="--ac:#ff3b30;--ac-rgb:255,59,48;--ac-color:#ff3b30;">
        <div class="stat-icon">↩️</div><div class="stat-value">{ww_count}</div>
        <div class="stat-label">Wrong-Way</div>
      </div>
      <div class="stat-card" style="--ac:#ff9500;--ac-rgb:255,149,0;--ac-color:#ff9500;">
        <div class="stat-icon">⛑️</div><div class="stat-value">{nh_count}</div>
        <div class="stat-label">No Helmet</div>
      </div>
      <div class="stat-card" style="--ac:#30a0ff;--ac-rgb:48,160,255;--ac-color:#30a0ff;">
        <div class="stat-icon">🆔</div><div class="stat-value">{unique_ids}</div>
        <div class="stat-label">Unique Offenders</div>
      </div>
      <div class="stat-card" style="--ac:#50d090;--ac-rgb:80,208,144;--ac-color:#50d090;">
        <div class="stat-icon">🎞</div><div class="stat-value">{cur_frame}</div>
        <div class="stat-label">Frames Processed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if total_v > 0:
        ww_pct = int(ww_count / total_v * 100)
        nh_pct = int(nh_count / total_v * 100)
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-bottom:1.5rem;">
          <div class="sev-wrap">
            <div class="sev-lbl"><span>Wrong-Way Share</span><span>{ww_pct}%</span></div>
            <div class="sev-track"><div class="sev-fill" style="width:{ww_pct}%;background:#ff3b30;"></div></div>
          </div>
          <div class="sev-wrap">
            <div class="sev-lbl"><span>No-Helmet Share</span><span>{nh_pct}%</span></div>
            <div class="sev-track"><div class="sev-fill" style="width:{nh_pct}%;background:#ff9500;"></div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Filter
    fc1, fc2, fc3, _ = st.columns([0.7,1.1,1.1,5])
    with fc1:
        if st.button("ALL",       key="lf_all"): st.session_state.log_filter = "ALL"
    with fc2:
        if st.button("WRONG WAY", key="lf_ww"):  st.session_state.log_filter = "WRONG_WAY"
    with fc3:
        if st.button("NO HELMET", key="lf_nh"):  st.session_state.log_filter = "NO_HELMET"

    lf       = st.session_state.log_filter
    filtered = violations if lf == "ALL" else [v for v in violations if v.get("vtype") == lf]

    log_col, tl_col = st.columns([3, 2], gap="large")

    with log_col:
        st.markdown('<div class="sec-hdr">VIOLATION LOG</div>', unsafe_allow_html=True)
        if filtered:
            rows = ""
            for v in filtered:
                vtype = v.get("vtype","UNKNOWN")
                ac    = "#ff3b30" if vtype == "WRONG_WAY" else "#ff9500"
                bl    = badge_label(vtype)
                bc    = badge_class(vtype)
                tid   = v.get("track_id","?")
                ts    = v.get("timestamp","—")
                fr    = v.get("frame","—")
                rows += (
                    f'<div class="vt-row" style="--row-ac:{ac};grid-template-columns:55px 110px 150px 1fr;padding-left:0.4rem;">'
                    f'<span class="cell-id">#{tid}</span>'
                    f'<span><span class="badge {bc}">{bl}</span></span>'
                    f'<span class="cell-ts">{ts}</span>'
                    f'<span class="cell-fr">frame {fr}</span>'
                    f'</div>'
                )
            html = (
                '<div class="vtable">'
                '<div class="vt-head" style="grid-template-columns:55px 110px 150px 1fr;padding-left:0.4rem;">'
                '<span>ID</span><span>TYPE</span><span>TIME</span><span>FRAME</span>'
                '</div>'
                + rows +
                '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty"><div class="empty-icon">📋</div>NO VIOLATIONS YET</div>',
                        unsafe_allow_html=True)

    with tl_col:
        st.markdown('<div class="sec-hdr">TIMELINE</div>', unsafe_allow_html=True)
        if filtered:
            items = ""
            for v in filtered[:25]:
                vtype = v.get("vtype","UNKNOWN")
                dc    = "#ff3b30" if vtype == "WRONG_WAY" else "#ff9500"
                desc  = ("Vehicle travelling in wrong direction"
                         if vtype == "WRONG_WAY" else "Motorcycle rider without helmet")
                items += f"""
                <div class="tl-item">
                  <div class="tl-dot" style="background:{dc};"></div>
                  <div class="tl-ts">{v.get('timestamp','—')}</div>
                  <div class="tl-msg"><strong>ID {v['track_id']}</strong> — {desc}, frame {v.get('frame','?')}</div>
                </div>"""
            st.markdown(f'<div style="background:#ffffff;border:1px solid #d8e2ee;'
                        f'padding:0.4rem 0.85rem;">{items}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty"><div class="empty-icon">🕐</div>NO EVENTS</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — SNAPSHOTS GALLERY
# ══════════════════════════════════════════════════════════════
with tab_gallery:

    st.markdown('<div class="sec-hdr">SNAPSHOT GALLERY</div>', unsafe_allow_html=True)

    images = []
    if st.session_state.output_dir:
        images = scan_violation_images(st.session_state.output_dir)

    gfc1, gfc2, gfc3, _ = st.columns([0.7,1.1,1.1,5])
    with gfc1:
        if st.button("ALL",       key="gf_all"): st.session_state.gal_filter = "ALL"
    with gfc2:
        if st.button("WRONG WAY", key="gf_ww2"): st.session_state.gal_filter = "WRONG_WAY"
    with gfc3:
        if st.button("NO HELMET", key="gf_nh2"): st.session_state.gal_filter = "NO_HELMET"

    gf   = st.session_state.gal_filter
    imgs = images if gf == "ALL" else [i for i in images if i["vtype"] == gf]

    if not imgs:
        st.markdown("""
        <div class="empty" style="padding:3rem;">
          <div class="empty-icon">📷</div>NO SNAPSHOTS YET<br><br>
          <span style="font-size:0.55rem;color:#c0d0e0;">Run the detector to generate violation snapshots</span>
        </div>""", unsafe_allow_html=True)
    else:
        COLS = 3
        for row_start in range(0, len(imgs), COLS):
            gcols = st.columns(COLS, gap="small")
            for ci, img_data in enumerate(imgs[row_start:row_start+COLS]):
                with gcols[ci]:
                    bc   = badge_class(img_data["vtype"])
                    bl   = badge_label(img_data["vtype"])
                    id_s = "ID %d" % img_data["track_id"] if img_data["track_id"] else "???"
                    fr_s = "f%d" % img_data["frame"]      if img_data["frame"]     else ""
                    fname = img_data["filename"]
                    fpath = img_data["path"]

                    # Sync encrypted state from disk on each render
                    # Disk is authoritative: sidecar present → encrypted, sidecar absent → not encrypted
                    if is_encrypted(fpath):
                        st.session_state.enc_status[fname] = "encrypted"
                    else:
                        # No sidecar: either never encrypted, or successfully decrypted
                        # Only update session state if it was previously marked encrypted
                        # (avoids overwriting an "error:..." state with "decrypted")
                        prev = st.session_state.enc_status.get(fname, "decrypted")
                        if prev == "encrypted":
                            st.session_state.enc_status[fname] = "decrypted"

                    enc_state = st.session_state.enc_status.get(fname, "decrypted")
                    currently_encrypted = (enc_state == "encrypted")

                    # Pre-compute badge style
                    if img_data["vtype"] == "WRONG_WAY":
                        badge_style = ("background:rgba(255,59,48,.12);color:#ff3b30;"
                                       "border:1px solid rgba(255,59,48,.28);")
                    else:
                        badge_style = ("background:rgba(255,149,0,.12);color:#ff9500;"
                                       "border:1px solid rgba(255,149,0,.28);")

                    lock_badge = ""
                    if currently_encrypted:
                        lock_badge = (
                            '<span style="background:rgba(255,59,48,.15);color:#ff3b30;' +
                            'border:1px solid rgba(255,59,48,.35);border-radius:2px;' +
                            'padding:2px 6px;font-size:0.5rem;font-family:monospace;">🔒 ENCRYPTED</span>'
                        )

                    # Render metadata bar ABOVE the image — prevents Streamlit from
                    # leaking raw <span> tag text below the image element
                    st.markdown(
                        f'<div style="background:#f8fafc;border:1px solid #d0d8e8;border-bottom:none;' +
                        f'padding:0.3rem 0.55rem;display:flex;justify-content:space-between;align-items:center;">' +
                        f'<span style="font-family:Share Tech Mono,monospace;font-size:0.57rem;color:#8a9db0;">' +
                        f'{id_s} &middot; {fr_s}</span>' +
                        f'<span style="display:flex;gap:0.3rem;align-items:center;">' +
                        f'{lock_badge}' +
                        f'<span style="display:inline-block;padding:2px 7px;border-radius:2px;' +
                        f'font-family:Share Tech Mono,monospace;font-size:0.5rem;' +
                        f'letter-spacing:0.06em;text-transform:uppercase;font-weight:600;' +
                        f'{badge_style}">{bl}</span></span></div>',
                        unsafe_allow_html=True,
                    )

                    # Show encrypted copy when encrypted, original otherwise
                    display_path = get_display_path(fpath)
                    try:
                        img_pil = PILImage.open(display_path)
                        st.image(img_pil, use_container_width=True)
                    except Exception:
                        st.markdown(
                            '<div style="background:#f5f7fa;border:1px solid #d0d8e8;' +
                            'height:90px;display:flex;align-items:center;justify-content:center;' +
                            'color:#b0c0d0;font-family:monospace;font-size:0.6rem;">IMG ERROR</div>',
                            unsafe_allow_html=True,
                        )

                    # Action buttons row
                    btn_enc, btn_dec = st.columns([1, 1])

                    with btn_enc:
                        enc_disabled = currently_encrypted
                        if st.button("🔒 ENC", key="enc_%s" % fname, disabled=enc_disabled):
                            with st.spinner("Encrypting…"):
                                result = encrypt_snapshot(
                                    fpath,
                                    enc_params,
                                    st.session_state.roboflow_key,
                                )
                            if result["ok"] and result.get("warning") != "no_regions":
                                st.session_state.enc_status[fname] = "encrypted"
                                st.rerun()
                            elif result["ok"] and result.get("warning") == "no_regions":
                                st.warning("No faces or licence plates detected — nothing to encrypt.")
                            else:
                                st.session_state.enc_status[fname] = "error:%s" % result.get("error","")
                                st.error("Encrypt failed: %s" % result.get("error",""))

                    with btn_dec:
                        dec_disabled = not currently_encrypted
                        if st.button("🔓 DEC", key="dec_%s" % fname, disabled=dec_disabled):
                            with st.spinner("Decrypting…"):
                                result = decrypt_snapshot(
                                    fpath,
                                    enc_params,
                                    st.session_state.roboflow_key,
                                )
                            if result["ok"]:
                                st.session_state.enc_status[fname] = "decrypted"
                                st.rerun()
                            else:
                                st.error("Decrypt failed: %s" % result.get("error",""))


# ══════════════════════════════════════════════════════════════
# TAB 4 — ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab_analysis:

    # Pull violations data (same pattern as other tabs)
    prog_an = read_progress(st.session_state.output_dir)
    if prog_an:
        an_violations = prog_an.get("violations", [])
        an_total_frames = prog_an.get("total_frames", 0)
        an_fps = 25.0  # fallback; detector always writes at this default
    elif st.session_state.output_dir:
        an_violations = load_violations_json(st.session_state.output_dir)
        an_total_frames = 0
        an_fps = 25.0
    else:
        an_violations = []
        an_total_frames = 0
        an_fps = 25.0

    # Try to recover fps & total_frames from the progress file
    if prog_an:
        an_total_frames = prog_an.get("total_frames", 0) or 0

    total_an = len(an_violations)

    if total_an == 0:
        st.markdown("""
        <div class="empty" style="margin-top:3rem;">
          <div class="empty-icon">📊</div>
          RUN DETECTION FIRST TO SEE ANALYSIS
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="sec-hdr">VIOLATION ANALYSIS</div>', unsafe_allow_html=True)

        # ── Compute counts ──────────────────────────────────────
        ww_an = sum(1 for v in an_violations if v.get("vtype") == "WRONG_WAY")
        nh_an = sum(1 for v in an_violations if v.get("vtype") == "NO_HELMET")

        # ── Helpers ─────────────────────────────────────────────
        def ts_to_seconds(ts_str):
            """Convert HH:MM:SS.mmm → float seconds."""
            try:
                parts = ts_str.split(":")
                h, m = int(parts[0]), int(parts[1])
                s = float(parts[2])
                return h * 3600 + m * 60 + s
            except Exception:
                return None

        # Compute total video duration in seconds
        if an_total_frames > 0 and an_fps > 0:
            video_duration_s = an_total_frames / an_fps
        else:
            # Fallback: use the last violation's timestamp + a small buffer
            sec_vals = [ts_to_seconds(v.get("timestamp","")) for v in an_violations]
            sec_vals = [s for s in sec_vals if s is not None]
            video_duration_s = max(sec_vals) * 1.05 if sec_vals else 60.0

        # ── Time-interval label helper ───────────────────────────
        def fmt_duration(seconds):
            """Format a duration in seconds into the most readable unit string."""
            if seconds < 1:
                ms = int(seconds * 1000)
                return "%dms" % ms
            elif seconds < 60:
                return "%.1fs" % seconds if seconds != int(seconds) else "%ds" % int(seconds)
            elif seconds < 3600:
                m = seconds / 60
                return "%.1fmin" % m if m != int(m) else "%dmin" % int(m)
            else:
                h = seconds / 3600
                return "%.1fhr" % h if h != int(h) else "%dhr" % int(h)

        # Each bin covers 1/8 of the total duration
        bin_size_s   = video_duration_s / 8
        bin_labels   = []
        for i in range(8):
            start = i * bin_size_s
            end   = (i + 1) * bin_size_s
            bin_labels.append("%s–%s" % (fmt_duration(start), fmt_duration(end)))

        # Count violations per bin (split by type)
        bin_ww = [0] * 8
        bin_nh = [0] * 8
        for v in an_violations:
            s = ts_to_seconds(v.get("timestamp", ""))
            if s is None:
                continue
            idx = min(int(s / bin_size_s), 7)
            if v.get("vtype") == "WRONG_WAY":
                bin_ww[idx] += 1
            else:
                bin_nh[idx] += 1
        bin_total = [bin_ww[i] + bin_nh[i] for i in range(8)]

        # ── Colours matching the UI palette ─────────────────────
        CLR_WW   = "#ff3b30"   # red — wrong-way (matches badges)
        CLR_NH   = "#ff9500"   # amber — no-helmet (matches badges)
        CLR_AXIS = "#9aaabb"   # muted blue-grey — axis text
        CLR_GRID = "#e0e8f0"   # light line — grid
        CLR_BG   = "#ffffff"   # card bg

        # ── Percentage values for pie ────────────────────────────
        ww_pct_f = ww_an / total_an * 100
        nh_pct_f = nh_an / total_an * 100

        # ── Plotly charts ────────────────────────────────────────
        import plotly.graph_objects as go

        chart_col, bar_col = st.columns([1, 1.6], gap="large")

        # ---- PIE CHART ----------------------------------------
        with chart_col:
            st.markdown('<div class="sec-hdr">VIOLATION TYPE BREAKDOWN</div>', unsafe_allow_html=True)

            pie_fig = go.Figure(data=[go.Pie(
                labels=["Wrong-Way", "No Helmet"],
                values=[ww_an, nh_an],
                hole=0.52,
                marker=dict(
                    colors=[CLR_WW, CLR_NH],
                    line=dict(color="#f0f2f5", width=3),
                ),
                textinfo="percent",
                textfont=dict(family="Share Tech Mono, monospace", size=13, color="#ffffff"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
                direction="clockwise",
                sort=False,
            )])

            pie_fig.add_annotation(
                text=f"<b>{total_an}</b><br><span style='font-size:10px'>TOTAL</span>",
                x=0.5, y=0.5,
                font=dict(family="Barlow Condensed, sans-serif", size=22, color="#1a2332"),
                showarrow=False,
                align="center",
            )

            pie_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    x=0.5, xanchor="center",
                    y=-0.06,
                    font=dict(family="Share Tech Mono, monospace", size=10, color=CLR_AXIS),
                ),
                height=320,
            )
            st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

            # Summary stats below pie
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.2rem;">
              <div style="background:#fff5f5;border:1px solid rgba(255,59,48,.2);border-left:2px solid #ff3b30;
                          padding:0.6rem 0.8rem;">
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#9aaabb;
                            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem;">Wrong-Way</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;
                            color:#ff3b30;line-height:1;">{ww_an}</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#cc3020;">
                  {ww_pct_f:.1f}%</div>
              </div>
              <div style="background:#fff9f0;border:1px solid rgba(255,149,0,.2);border-left:2px solid #ff9500;
                          padding:0.6rem 0.8rem;">
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#9aaabb;
                            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem;">No Helmet</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;
                            color:#ff9500;line-height:1;">{nh_an}</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#cc7800;">
                  {nh_pct_f:.1f}%</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ---- BAR CHART ----------------------------------------
        with bar_col:
            st.markdown('<div class="sec-hdr">VIOLATIONS OVER TIME  ·  8 EQUAL INTERVALS</div>',
                        unsafe_allow_html=True)

            bar_fig = go.Figure()

            # Stacked bars — wrong-way bottom, no-helmet on top
            bar_fig.add_trace(go.Bar(
                name="Wrong-Way",
                x=bin_labels,
                y=bin_ww,
                marker_color=CLR_WW,
                marker_line=dict(width=0),
                hovertemplate="<b>%{x}</b><br>Wrong-Way: %{y}<extra></extra>",
            ))
            bar_fig.add_trace(go.Bar(
                name="No Helmet",
                x=bin_labels,
                y=bin_nh,
                marker_color=CLR_NH,
                marker_line=dict(width=0),
                hovertemplate="<b>%{x}</b><br>No Helmet: %{y}<extra></extra>",
            ))

            bar_fig.update_layout(
                barmode="stack",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=CLR_BG,
                margin=dict(t=10, b=10, l=10, r=10),
                height=370,
                xaxis=dict(
                    title=dict(
                        text="TIME INTERVAL  (total duration: %s)" % fmt_duration(video_duration_s),
                        font=dict(family="Share Tech Mono, monospace", size=9, color=CLR_AXIS),
                    ),
                    tickfont=dict(family="Share Tech Mono, monospace", size=8, color=CLR_AXIS),
                    tickangle=-35,
                    showgrid=False,
                    linecolor=CLR_GRID,
                    linewidth=1,
                ),
                yaxis=dict(
                    title=dict(
                        text="VIOLATIONS",
                        font=dict(family="Share Tech Mono, monospace", size=9, color=CLR_AXIS),
                    ),
                    tickfont=dict(family="Share Tech Mono, monospace", size=9, color=CLR_AXIS),
                    gridcolor=CLR_GRID,
                    gridwidth=1,
                    zeroline=False,
                    showline=False,
                    dtick=1,
                ),
                legend=dict(
                    orientation="h",
                    x=1, xanchor="right",
                    y=1.02, yanchor="bottom",
                    font=dict(family="Share Tech Mono, monospace", size=9, color=CLR_AXIS),
                    bgcolor="rgba(0,0,0,0)",
                ),
                hoverlabel=dict(
                    bgcolor="#1a2332",
                    font=dict(family="Share Tech Mono, monospace", size=11, color="#ffffff"),
                    bordercolor="#1a2332",
                ),
            )

            # Subtle border around plot area
            bar_fig.update_xaxes(showline=True, linecolor=CLR_GRID)
            bar_fig.update_yaxes(showline=False)

            st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

            # Peak bin callout
            peak_idx = bin_total.index(max(bin_total))
            st.markdown(f"""
            <div style="background:#f5f7fa;border:1px solid #d8e2ee;border-left:2px solid #30a0ff;
                        padding:0.55rem 1rem;font-family:'Share Tech Mono',monospace;
                        font-size:0.6rem;color:#5a7080;letter-spacing:0.1em;margin-top:0.3rem;">
              ⚡ &nbsp; PEAK INTERVAL: &nbsp;
              <span style="color:#1a2332;font-weight:600;">{bin_labels[peak_idx]}</span>
              &nbsp;·&nbsp; {bin_total[peak_idx]} violation(s)
              &nbsp;·&nbsp; interval size: {fmt_duration(bin_size_s)}
            </div>""", unsafe_allow_html=True)

        # ── INTELLIGENCE SUMMARY ────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sec-hdr">INTELLIGENCE SUMMARY</div>', unsafe_allow_html=True)

        # Determine dominant violation type
        if ww_an >= nh_an:
            dominant_type  = "Wrong-Way driving"
            dominant_count = ww_an
            dominant_pct   = ww_pct_f
            dominant_color = "#ff3b30"
            dominant_badge = "WRONG-WAY"
        else:
            dominant_type  = "No-Helmet riding"
            dominant_count = nh_an
            dominant_pct   = nh_pct_f
            dominant_color = "#ff9500"
            dominant_badge = "NO-HELMET"

        # Second peak (if there are at least 2 non-zero bins)
        sorted_bins = sorted(enumerate(bin_total), key=lambda x: x[1], reverse=True)
        second_peak_idx   = sorted_bins[1][0] if len(sorted_bins) > 1 and sorted_bins[1][1] > 0 else None
        second_peak_label = bin_labels[second_peak_idx] if second_peak_idx is not None else None

        # Severity level
        severity = "LOW"
        sev_color = "#50d090"
        if total_an >= 20 or (ww_an > 0 and ww_pct_f > 50):
            severity  = "HIGH"
            sev_color = "#ff3b30"
        elif total_an >= 8 or (ww_an > 0 and ww_pct_f > 25):
            severity  = "MEDIUM"
            sev_color = "#ff9500"

        # Build recommendation list dynamically
        recommendations = []
        if ww_an > 0:
            recommendations.append({
                "icon": "🚧",
                "title": "Install physical lane dividers or median barriers",
                "detail": "Prevents wrong-way entry at high-risk segments identified in the peak interval.",
            })
            recommendations.append({
                "icon": "👮",
                "title": "Deploy traffic officer at peak interval (%s)" % bin_labels[peak_idx],
                "detail": "Stationing an officer during %s can deter wrong-way entries and enable immediate intervention." % bin_labels[peak_idx],
            })
            recommendations.append({
                "icon": "🚦",
                "title": "Add high-visibility wrong-way signage and road markings",
                "detail": "Retroreflective 'DO NOT ENTER' signs and chevron road markings reduce wrong-way incursions by up to 60%.",
            })
            if second_peak_label:
                recommendations.append({
                    "icon": "📷",
                    "title": "Increase camera coverage at secondary hotspot (%s)" % second_peak_label,
                    "detail": "A secondary spike was detected here — additional monitoring can confirm if it's a recurring problem area.",
                })
        if nh_an > 0:
            recommendations.append({
                "icon": "⛑️",
                "title": "Enforce helmet checks at entry/exit points",
                "detail": "Position enforcement personnel or automated alerts at key intersections where no-helmet riders enter.",
            })
            recommendations.append({
                "icon": "📢",
                "title": "Run targeted awareness campaigns",
                "detail": "Digital signage and outreach campaigns near the monitored road can improve helmet compliance over time.",
            })
        if ww_an > 0 and nh_an > 0:
            recommendations.append({
                "icon": "🔁",
                "title": "Review detection footage during peak interval for road design flaws",
                "detail": "Concurrent wrong-way and no-helmet violations suggest informal cut-throughs or poorly signed junctions — consider closing them.",
            })

        # Render summary card
        second_peak_html = (
            f' A secondary concentration of violations was observed at <strong>{second_peak_label}</strong>.'
            if second_peak_label else ""
        )

        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #d8e2ee;border-top:2px solid {dominant_color};
                    padding:1.2rem 1.4rem;margin-bottom:1rem;">

          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">
            <div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#9aaabb;
                          letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.4rem;">
                ANALYSIS REPORT
              </div>
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;font-weight:700;
                          color:#1a2332;line-height:1.2;">
                {total_an} violation{'s' if total_an != 1 else ''} detected across {fmt_duration(video_duration_s)} of footage
              </div>
            </div>
            <div style="text-align:right;flex-shrink:0;margin-left:1rem;">
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#9aaabb;
                          letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem;">SEVERITY</div>
              <div style="background:rgba(0,0,0,0.04);border:1px solid {sev_color};border-radius:2px;
                          padding:4px 12px;font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                          font-weight:700;color:{sev_color};letter-spacing:0.1em;">{severity}</div>
            </div>
          </div>

          <div style="font-size:0.82rem;color:#3a5060;line-height:1.75;margin-bottom:0.5rem;">
            The predominant violation is
            <span style="font-weight:600;color:{dominant_color};">{dominant_type}</span>,
            accounting for <strong>{dominant_count} ({dominant_pct:.0f}%)</strong> of all recorded incidents.
            The highest concentration of violations occurs in the
            <strong>{bin_labels[peak_idx]}</strong> interval
            with <strong>{bin_total[peak_idx]} violation{'s' if bin_total[peak_idx] != 1 else ''}</strong>.{second_peak_html}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Render recommendations
        st.markdown('<div class="sec-hdr">RECOMMENDED INTERVENTIONS</div>', unsafe_allow_html=True)

        rec_cols = st.columns(2, gap="medium")
        for ri, rec in enumerate(recommendations):
            with rec_cols[ri % 2]:
                st.markdown(f"""
                <div style="background:#ffffff;border:1px solid #d8e2ee;padding:0.85rem 1rem;
                            margin-bottom:0.7rem;display:flex;gap:0.8rem;align-items:flex-start;">
                  <div style="font-size:1.3rem;flex-shrink:0;margin-top:0.1rem;">{rec['icon']}</div>
                  <div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.95rem;font-weight:700;
                                color:#1a2332;margin-bottom:0.25rem;">{rec['title']}</div>
                    <div style="font-size:0.75rem;color:#6a8090;line-height:1.55;">{rec['detail']}</div>
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — CONFIGURATION
# ══════════════════════════════════════════════════════════════
with tab_config:

    st.markdown('<div class="sec-hdr">DETECTION PARAMETERS</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("**VEHICLE DETECTION**")
        st.session_state["cfg_conf_vehicle"] = st.slider(
            "Confidence threshold", 0.1, 0.9,
            float(st.session_state.get("cfg_conf_vehicle", 0.30)), 0.05, key="sl_cv")
        st.session_state["cfg_min_frames"] = st.number_input(
            "Min consecutive frames (wrong-way)", 1, 20,
            int(st.session_state.get("cfg_min_frames", 2)), key="ni_mf")

    with c2:
        st.markdown("**HELMET DETECTION**")
        st.session_state["cfg_conf_helmet"] = st.slider(
            "Helmet confidence threshold", 0.05, 0.9,
            float(st.session_state.get("cfg_conf_helmet", 0.20)), 0.05, key="sl_ch")
        st.session_state["cfg_vote_frames"] = st.number_input(
            "Helmet vote window (frames)", 1, 20,
            int(st.session_state.get("cfg_vote_frames", 5)), key="ni_vf")
        st.session_state["cfg_iou_thresh"] = st.slider(
            "Bike-person IoU threshold", 0.01, 0.5,
            float(st.session_state.get("cfg_iou_thresh", 0.10)), 0.01, key="sl_iou")

    with c3:
        st.markdown("**LANE DIRECTION**")
        st.caption("P1 → P2 defines the correct direction of travel")
        pc1, pc2 = st.columns(2)
        with pc1:
            st.session_state["cfg_p1x"] = st.number_input("P1 X", 0, 3840, int(st.session_state.get("cfg_p1x",500)), key="ni_p1x")
            st.session_state["cfg_p2x"] = st.number_input("P2 X", 0, 3840, int(st.session_state.get("cfg_p2x",500)), key="ni_p2x")
        with pc2:
            st.session_state["cfg_p1y"] = st.number_input("P1 Y", 0, 2160, int(st.session_state.get("cfg_p1y",600)), key="ni_p1y")
            st.session_state["cfg_p2y"] = st.number_input("P2 Y", 0, 2160, int(st.session_state.get("cfg_p2y",100)), key="ni_p2y")

    st.markdown("---")
    st.markdown('<div class="sec-hdr">CURRENT CONFIG PREVIEW</div>', unsafe_allow_html=True)
    st.code(json.dumps({
        "conf_vehicle"      : st.session_state.get("cfg_conf_vehicle", 0.30),
        "conf_helmet"       : st.session_state.get("cfg_conf_helmet", 0.20),
        "min_frames"        : st.session_state.get("cfg_min_frames", 2),
        "helmet_vote_frames": st.session_state.get("cfg_vote_frames", 5),
        "helmet_iou_thresh" : st.session_state.get("cfg_iou_thresh", 0.10),
        "lane_p1"           : [st.session_state.get("cfg_p1x",500), st.session_state.get("cfg_p1y",600)],
        "lane_p2"           : [st.session_state.get("cfg_p2x",500), st.session_state.get("cfg_p2y",100)],
    }, indent=2), language="json")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid #e0e8f0;
            display:flex;justify-content:space-between;">
  <span style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#b0c0d0;letter-spacing:0.15em;">
    TRAFFIC SENTINEL · YOLO v8 + BYTETRACK · AUTOMATED VIOLATION MONITOR
  </span>
  <span style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#c0d0e0;">
    OpenCV · Ultralytics · Streamlit
  </span>
</div>""", unsafe_allow_html=True)