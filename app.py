"""
MedGuard AI — Premium Medical Image Integrity Platform
Hybrid Residual U-Net++ · Locality-Sensitive Hashing · DWT-LSB · Grad-CAM
"""
import os, re, sys, json, hashlib, __main__

import cv2
import numpy as np
import pandas as pd
import pywt
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.metrics  import all_metrics
from  utils.grad_cam import GradCAM, overlay_heatmap
from utils.history  import add_record, get_records, clear_records
from utils.report   import generate_pdf, REPORTLAB_OK

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MedGuard AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# THEME TOKENS
# ══════════════════════════════════════════════════════════════════════════════
_DARK = dict(
    bg="#0F172A", bg2="#1E293B", card="#1E293B", card2="#162032",
    accent="#3B82F6", accent2="#8B5CF6", cyan="#06B6D4",
    success="#10B981", warning="#F59E0B", error="#EF4444",
    text="#F8FAFC", muted="#94A3B8", muted2="#64748B",
    border="#334155", border2="#1E293B",
    sidebar="#0B1120",
    input_bg="#0F172A", input_border="#334155",
    shadow="rgba(0,0,0,0.45)",
    glass_bg="rgba(30,41,59,0.80)",
    glass_border="rgba(255,255,255,0.08)",
    badge_bg="rgba(59,130,246,0.12)",
    badge_text="#93C5FD",
    gradient="linear-gradient(135deg,#3B82F6 0%,#8B5CF6 50%,#06B6D4 100%)",
)
_LIGHT = dict(
    bg="#F8FAFC", bg2="#F1F5F9", card="#FFFFFF", card2="#F8FAFC",
    accent="#2563EB", accent2="#7C3AED", cyan="#0891B2",
    success="#059669", warning="#D97706", error="#DC2626",
    text="#0F172A", muted="#64748B", muted2="#94A3B8",
    border="#E2E8F0", border2="#CBD5E1",
    sidebar="#F1F5F9",
    input_bg="#FFFFFF", input_border="#CBD5E1",
    shadow="rgba(15,23,42,0.08)",
    glass_bg="rgba(255,255,255,0.85)",
    glass_border="rgba(15,23,42,0.06)",
    badge_bg="rgba(37,99,235,0.08)",
    badge_text="#2563EB",
    gradient="linear-gradient(135deg,#2563EB 0%,#7C3AED 50%,#0891B2 100%)",
)

def T() -> dict:
    return _DARK if st.session_state.get("dark_mode", True) else _LIGHT


# ══════════════════════════════════════════════════════════════════════════════
# CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    t = T()
    st.markdown(f"""
<style>
/* ── GOOGLE FONT ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── RESET ────────────────────────────────────────────────────────────────── */
*,*::before,*::after{{box-sizing:border-box;}}

/* ── STREAMLIT BASE ────────────────────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stApp"]{{
  background:{t["bg"]} !important;
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
}}
.main,.block-container,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"]{{
  background:{t["bg"]} !important;
  padding:2rem 2.5rem !important;
}}
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"]{{
  background:transparent !important;
}}
/* Hide Streamlit chrome */
[data-testid="stToolbar"],
[data-testid="stDeployButton"],
#MainMenu,footer,.viewerBadge_container__1QSob{{display:none !important;}}
[data-testid="stHeader"]{{
  background:{t["bg"]}cc !important;
  border-bottom:1px solid {t["border"]} !important;
  backdrop-filter:blur(20px);
}}

/* ── GLOBAL TEXT ──────────────────────────────────────────────────────────── */
/* NOTE: do NOT include span/div here — that overrides Streamlit's Material
   Icons font and causes icon names to render as plain text (overlapping UI). */
p,li,td,th,h1,h2,h3,h4,h5,h6,label,small,
[data-testid="stMarkdownContainer"] *{{
  color:{t["text"]} !important;
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif !important;
}}
/* anchor colour only */
a{{color:{t["accent"]} !important;}}
a:hover{{opacity:.8;}}
code,pre{{
  color:{t["accent"]} !important;
  background:{t["badge_bg"]} !important;
  border-radius:6px;
  font-size:.85em;
}}

/* ── SIDEBAR ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"]{{
  background:{t["sidebar"]} !important;
  border-right:1px solid {t["border"]} !important;
}}
[data-testid="stSidebar"]>div:first-child{{padding:1.25rem 1rem !important;}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *{{
  color:{t["text"]} !important;
}}
[data-testid="stSidebar"] hr{{border-color:{t["border"]} !important;margin:.875rem 0 !important;}}
[data-testid="stSidebar"] .stButton>button{{
  width:100% !important; text-align:left !important;
  background:transparent !important; border:none !important;
  color:{t["muted"]} !important; font-size:.875rem !important;
  font-weight:500 !important; padding:.6rem .875rem !important;
  border-radius:8px !important; margin-bottom:2px !important;
  transition:all .2s ease !important; letter-spacing:0 !important;
  box-shadow:none !important;
}}
[data-testid="stSidebar"] .stButton>button:hover{{
  background:{t["badge_bg"]} !important; color:{t["accent"]} !important;
  transform:none !important; filter:none !important;
}}

/* ── BUTTONS ──────────────────────────────────────────────────────────────── */
.stButton>button,.stFormSubmitButton>button{{
  background:{t["gradient"]} !important;
  color:#FFFFFF !important; border:none !important;
  border-radius:10px !important; font-weight:600 !important;
  font-size:.875rem !important; padding:.625rem 1.5rem !important;
  transition:all .25s cubic-bezier(.4,0,.2,1) !important;
  letter-spacing:.01em !important;
  box-shadow:0 4px 15px {t["accent"]}30 !important;
}}
.stButton>button:hover,.stFormSubmitButton>button:hover{{
  transform:translateY(-2px) !important;
  box-shadow:0 8px 25px {t["accent"]}45 !important;
  filter:brightness(1.08) !important;
}}
.stButton>button:active{{transform:translateY(0) !important;}}
[data-testid="stSidebar"] .stButton>button{{
  background:transparent !important; box-shadow:none !important;
  color:{t["muted"]} !important;
}}
[data-testid="stSidebar"] .stButton>button:hover{{
  background:{t["badge_bg"]} !important;
  box-shadow:none !important; transform:none !important; filter:none !important;
}}

/* ── METRICS ──────────────────────────────────────────────────────────────── */
[data-testid="stMetric"]{{
  background:{t["card"]} !important; border:1px solid {t["border"]} !important;
  border-radius:14px !important; padding:1.25rem 1.5rem !important;
  transition:all .3s ease !important;
}}
[data-testid="stMetric"]:hover{{
  border-color:{t["accent"]}60 !important;
  box-shadow:0 8px 24px {t["shadow"]},0 0 0 1px {t["accent"]}20 !important;
}}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{{
  color:{t["muted"]} !important; font-size:.72rem !important;
  font-weight:700 !important; text-transform:uppercase !important;
  letter-spacing:.09em !important;
}}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{{
  color:{t["text"]} !important; font-size:1.75rem !important;
  font-weight:800 !important; letter-spacing:-.02em !important;
  font-variant-numeric:tabular-nums !important;
}}

/* ── TABS ──────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"]{{
  background:{t["bg2"]} !important; border-radius:12px !important;
  padding:4px !important; gap:3px !important;
  border:1px solid {t["border"]} !important;
}}
[data-testid="stTabs"] button[role="tab"]{{
  background:transparent !important; border:none !important;
  border-radius:8px !important; color:{t["muted"]} !important;
  font-weight:600 !important; font-size:.875rem !important;
  padding:.5rem 1.25rem !important; transition:all .2s ease !important;
}}
[data-testid="stTabs"] button[role="tab"]:hover{{
  color:{t["text"]} !important; background:{t["card"]}80 !important;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{{
  background:{t["card"]} !important; color:{t["accent"]} !important;
  box-shadow:0 1px 6px {t["shadow"]} !important;
}}

/* ── INPUTS ────────────────────────────────────────────────────────────────── */
.stTextInput label,.stSelectbox label,.stFileUploader label,
.stNumberInput label,[data-testid="stWidgetLabel"] *{{
  color:{t["text"]} !important; font-weight:600 !important;
  font-size:.875rem !important;
}}
.stTextInput input,.stNumberInput input{{
  background:{t["input_bg"]} !important; color:{t["text"]} !important;
  border:1px solid {t["input_border"]} !important; border-radius:10px !important;
  font-family:'Inter',sans-serif !important; transition:all .2s ease !important;
  padding:.6rem .875rem !important;
}}
.stTextInput input:focus,.stNumberInput input:focus{{
  border-color:{t["accent"]} !important;
  box-shadow:0 0 0 3px {t["accent"]}18 !important;
}}
.stTextInput input::placeholder{{color:{t["muted"]} !important;opacity:1 !important;}}
[data-testid="stSelectbox"] [data-baseweb="select"]>div{{
  background:{t["input_bg"]} !important; border:1px solid {t["input_border"]} !important;
  border-radius:10px !important; color:{t["text"]} !important;
}}
/* ── SELECTBOX DROPDOWN POPUP ──────────────────────────────────────────────── */
[data-baseweb="popover"],
[data-baseweb="popover"] [data-baseweb="menu"]{{
  background:{t["input_bg"]} !important;
  border:1px solid {t["input_border"]} !important;
  border-radius:10px !important;
}}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li{{
  background:{t["input_bg"]} !important;
  color:{t["text"]} !important;
}}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"]{{
  background:{t["badge_bg"]} !important;
  color:{t["accent"]} !important;
}}

/* ── FILE UPLOADER ─────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"]{{
  background:{t["card"]} !important;
  border:2px dashed {t["border"]} !important;
  border-radius:16px !important; transition:all .25s ease !important;
}}
[data-testid="stFileUploader"]:hover{{
  border-color:{t["accent"]}80 !important; background:{t["badge_bg"]} !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span{{color:{t["muted"]} !important;}}

/* ── ALERTS ────────────────────────────────────────────────────────────────── */
[data-testid="stAlert"]{{border-radius:12px !important;border:1px solid !important;}}
[data-testid="stAlert"][data-type="success"]{{
  background:{t["success"]}12 !important;border-color:{t["success"]}35 !important;
}}
[data-testid="stAlert"][data-type="error"]{{
  background:{t["error"]}12 !important;border-color:{t["error"]}35 !important;
}}
[data-testid="stAlert"][data-type="warning"]{{
  background:{t["warning"]}12 !important;border-color:{t["warning"]}35 !important;
}}
[data-testid="stAlert"][data-type="info"]{{
  background:{t["accent"]}12 !important;border-color:{t["accent"]}35 !important;
}}
[data-testid="stAlert"] p,[data-testid="stAlert"] span{{color:{t["text"]} !important;}}

/* ── EXPANDER ──────────────────────────────────────────────────────────────── */
[data-testid="stExpander"]{{
  background:{t["card"]} !important; border:1px solid {t["border"]} !important;
  border-radius:14px !important; overflow:hidden !important;
}}
[data-testid="stExpander"] summary{{
  background:{t["card"]} !important; font-weight:600 !important;
  font-size:.875rem !important; padding:.875rem 1.25rem !important;
  color:{t["text"]} !important;
  display:flex !important; align-items:center !important;
}}
[data-testid="stExpander"] summary:hover{{background:{t["bg2"]} !important;}}
[data-testid="stExpander"] summary p{{
  color:{t["text"]} !important; font-weight:600 !important;
  font-size:.875rem !important; margin:0 !important;
}}

/* ── DATA FRAME ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"]{{
  border:1px solid {t["border"]} !important; border-radius:14px !important;
  overflow:hidden !important;
}}
.dvn-scroller{{background:{t["card"]} !important;}}

/* ── CHECKBOX / TOGGLE ────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label span{{color:{t["text"]} !important;}}
[data-testid="stCheckbox"] label span:first-child{{
  border:2px solid {t["input_border"]} !important; border-radius:4px !important;
}}

/* ── SPINNER ────────────────────────────────────────────────────────────────── */
[data-testid="stStatusWidget"] p,[data-testid="stStatusWidget"] span{{
  color:{t["text"]} !important;
}}

/* ── SCROLLBAR ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:{t["bg"]};}}
::-webkit-scrollbar-thumb{{background:{t["border"]};border-radius:10px;}}
::-webkit-scrollbar-thumb:hover{{background:{t["muted2"]};}}

/* ── HR ───────────────────────────────────────────────────────────────────── */
hr{{border-color:{t["border"]} !important;opacity:1 !important;margin:1.5rem 0 !important;}}

/* ══════════════════════════════════════════════════════════════════════════
   ANIMATIONS
   ══════════════════════════════════════════════════════════════════════════ */
@keyframes fadeInUp{{
  from{{opacity:0.4;transform:translateY(18px);}}
  to  {{opacity:1;transform:translateY(0);}}
}}
@keyframes fadeIn{{from{{opacity:0.4;}}to{{opacity:1;}}}}
@keyframes float{{
  0%,100%{{transform:translateY(0);}}
  50%     {{transform:translateY(-10px);}}
}}
@keyframes pulseGlow{{
  0%,100%{{box-shadow:0 0 0 0 {t["accent"]}50;}}
  50%    {{box-shadow:0 0 0 10px transparent;}}
}}
@keyframes shimmer{{
  0%  {{background-position:-1200px 0;}}
  100%{{background-position: 1200px 0;}}
}}
@keyframes spin{{from{{transform:rotate(0deg);}}to{{transform:rotate(360deg);}}}}
@keyframes gradientShift{{
  0%  {{background-position:0% 50%;}}
  50% {{background-position:100% 50%;}}
  100%{{background-position:0% 50%;}}
}}
@keyframes borderPulse{{
  0%,100%{{border-color:{t["border"]};}}
  50%    {{border-color:{t["accent"]}80;}}
}}

/* ══════════════════════════════════════════════════════════════════════════
   GRADIENT TEXT
   ══════════════════════════════════════════════════════════════════════════ */
.gradient-text{{
  background:{t["gradient"]};
  background-size:200% 200%;
  animation:gradientShift 4s ease infinite;
  -webkit-background-clip:text !important;
  -webkit-text-fill-color:transparent !important;
  background-clip:text !important;
  color:transparent !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   HERO SECTION
   ══════════════════════════════════════════════════════════════════════════ */
.hero-section{{
  padding:1.5rem 0 1rem;
}}
.hero-eyebrow{{
  display:inline-flex;align-items:center;gap:8px;
  background:{t["badge_bg"]};
  border:1px solid {t["accent"]}35;
  border-radius:100px;padding:.35rem .875rem;
  font-size:.75rem;font-weight:700;
  color:{t["badge_text"]} !important;
  margin-bottom:1rem;letter-spacing:.04em;
  text-transform:uppercase;
}}
.pulse-dot{{
  width:7px;height:7px;
  background:{t["success"]};border-radius:50%;
  animation:pulseGlow 2s ease infinite;
}}
.hero-title{{
  font-size:clamp(1.9rem,4vw,3rem);
  font-weight:900;line-height:1.1;
  letter-spacing:-.04em;
  color:{t["text"]} !important;
  margin-bottom:0.875rem;
}}
.hero-sub{{
  font-size:.95rem;color:{t["muted"]} !important;
  line-height:1.7;max-width:620px;
  margin-bottom:1.25rem;font-weight:400;
}}
.hero-divider{{
  width:60px;height:4px;
  background:{t["gradient"]};
  border-radius:2px;margin-bottom:1.25rem;
}}
.hero-kpis{{
  display:flex;gap:2rem;flex-wrap:wrap;
  padding:1rem 0;
  border-top:1px solid {t["border"]};
}}
.kpi-item{{display:flex;flex-direction:column;gap:.25rem;}}
.kpi-val{{
  font-size:1.35rem;font-weight:800;
  color:{t["text"]} !important;
  letter-spacing:-.03em;
  font-variant-numeric:tabular-nums;
}}
.kpi-lbl{{
  font-size:.7rem;font-weight:700;
  color:{t["muted"]} !important;
  text-transform:uppercase;letter-spacing:.1em;
}}

/* ══════════════════════════════════════════════════════════════════════════
   PREMIUM CARDS
   ══════════════════════════════════════════════════════════════════════════ */
.pcard{{
  background:{t["card"]};
  border:1px solid {t["border"]};
  border-radius:16px;padding:1.25rem;
  position:relative;overflow:hidden;
  transition:box-shadow .3s cubic-bezier(.4,0,.2,1),
             border-color .3s cubic-bezier(.4,0,.2,1);
}}
.pcard::after{{
  content:'';position:absolute;
  inset:0;border-radius:16px;
  background:{t["gradient"]};
  opacity:0;transition:opacity .3s ease;
  pointer-events:none;z-index:0;
  mask:linear-gradient(transparent 60%,black);
  -webkit-mask:linear-gradient(transparent 60%,black);
}}
.pcard:hover{{
  box-shadow:0 12px 32px {t["shadow"]};
  border-color:{t["accent"]}50;
}}
.pcard>*{{position:relative;z-index:1;}}
.pcard-top-bar{{
  position:absolute;top:0;left:0;right:0;height:3px;
  background:{t["gradient"]};
  opacity:0;transition:opacity .3s ease;
}}
.pcard:hover .pcard-top-bar{{opacity:1;}}
.pcard-icon{{
  font-size:2rem;margin-bottom:.875rem;
  display:block;
  animation:float 4s ease-in-out infinite;
}}
.pcard-title{{
  font-size:1rem;font-weight:700;
  color:{t["text"]} !important;
  margin-bottom:.5rem;letter-spacing:-.01em;
}}
.pcard-desc{{
  font-size:.84rem;color:{t["muted"]} !important;
  line-height:1.65;font-weight:400;
}}
.pcard-tag{{
  display:inline-block;margin-top:.875rem;
  background:{t["badge_bg"]};
  border:1px solid {t["accent"]}30;
  border-radius:6px;padding:.2rem .6rem;
  font-size:.7rem;font-weight:700;
  color:{t["badge_text"]} !important;
  text-transform:uppercase;letter-spacing:.06em;
}}

/* Glass card */
.glass-card{{
  background:{t["glass_bg"]};
  backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px);
  border:1px solid {t["glass_border"]};
  border-radius:16px;padding:1.5rem;
}}

/* Flat card */
.flat-card{{
  background:{t["bg2"]};
  border:1px solid {t["border"]};
  border-radius:14px;padding:1.25rem;
}}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION HEADINGS
   ══════════════════════════════════════════════════════════════════════════ */
.sec-title{{
  font-size:1.35rem;font-weight:800;
  color:{t["text"]} !important;
  letter-spacing:-.025em;margin-bottom:.35rem;
}}
.sec-sub{{
  font-size:.875rem;color:{t["muted"]} !important;
  margin-bottom:1.5rem;font-weight:400;
}}
.section-label{{
  display:flex;align-items:center;gap:.625rem;
  margin-bottom:1.25rem;
}}
.section-label-line{{
  flex:1;height:1px;background:{t["border"]};
}}
.section-label-text{{
  font-size:.72rem;font-weight:700;
  color:{t["muted"]} !important;
  text-transform:uppercase;letter-spacing:.1em;
  white-space:nowrap;
}}

/* ══════════════════════════════════════════════════════════════════════════
   LOGIN
   ══════════════════════════════════════════════════════════════════════════ */
.login-wrap{{
  min-height:80vh;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  animation:fadeIn .5s ease;
}}
.login-logo{{
  font-size:4rem;margin-bottom:1rem;
  animation:float 3s ease-in-out infinite;
  display:block;text-align:center;
}}
.login-title{{
  font-size:2.2rem;font-weight:900;
  letter-spacing:-.04em;
  color:{t["text"]} !important;
  text-align:center;margin-bottom:.4rem;
}}
.login-tagline{{
  font-size:.9rem;color:{t["muted"]} !important;
  text-align:center;margin-bottom:.25rem;font-weight:400;
}}
.login-rule{{
  font-size:.78rem;
  background:{t["badge_bg"]};
  border:1px solid {t["accent"]}30;
  border-radius:8px;
  padding:.4rem .9rem;
  color:{t["badge_text"]} !important;
  font-weight:600;text-align:center;margin-bottom:1.5rem;
  display:inline-block;
}}

/* ══════════════════════════════════════════════════════════════════════════
   SIDEBAR COMPONENTS
   ══════════════════════════════════════════════════════════════════════════ */
.sb-brand{{
  display:flex;align-items:center;gap:.75rem;
  padding:.25rem .25rem 1.25rem;
  border-bottom:1px solid {t["border"]};
  margin-bottom:1.25rem;
}}
.sb-logo{{font-size:1.6rem;}}
.sb-name{{
  font-size:.9rem;font-weight:800;
  color:{t["text"]} !important;letter-spacing:-.02em;
}}
.sb-tag{{
  font-size:.65rem;color:{t["muted"]} !important;
  font-weight:500;margin-top:1px;
}}
.nav-section{{
  font-size:.65rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.12em;
  color:{t["muted2"]} !important;
  padding:0 .5rem;margin:1.25rem 0 .5rem;
  display:block;
}}
.user-pill{{
  background:{t["bg2"]};
  border:1px solid {t["border"]};
  border-radius:12px;padding:.875rem 1rem;
  margin-top:.75rem;
}}
.user-pill-name{{
  font-size:.875rem;font-weight:700;
  color:{t["text"]} !important;
}}
.user-pill-role{{
  font-size:.7rem;color:{t["muted"]} !important;margin-top:2px;
}}
.status-dot{{
  display:inline-block;width:7px;height:7px;
  background:{t["success"]};border-radius:50%;
  margin-right:5px;animation:pulseGlow 2s ease infinite;
}}

/* ══════════════════════════════════════════════════════════════════════════
   VERDICT
   ══════════════════════════════════════════════════════════════════════════ */
.verdict-ok{{
  background:{t["success"]}14;border:1.5px solid {t["success"]}45;
  border-radius:14px;padding:1.35rem;text-align:center;
}}
.verdict-bad{{
  background:{t["error"]}14;border:1.5px solid {t["error"]}45;
  border-radius:14px;padding:1.35rem;text-align:center;
}}
.verdict-warn{{
  background:{t["warning"]}14;border:1.5px solid {t["warning"]}45;
  border-radius:14px;padding:1.35rem;text-align:center;
}}
.verdict-head{{
  font-size:1.05rem;font-weight:800;
  letter-spacing:-.01em;margin-bottom:.4rem;
}}
.verdict-body{{
  font-size:.78rem;color:{t["muted"]} !important;
  line-height:1.55;font-weight:400;
}}

/* ══════════════════════════════════════════════════════════════════════════
   CHIPS / BADGES
   ══════════════════════════════════════════════════════════════════════════ */
.chip-ok{{
  display:inline-flex;align-items:center;gap:5px;
  background:{t["success"]}18;border:1px solid {t["success"]}45;
  border-radius:100px;padding:.25rem .75rem;
  font-size:.72rem;font-weight:700;
  color:{t["success"]} !important;letter-spacing:.02em;
}}
.chip-bad{{
  display:inline-flex;align-items:center;gap:5px;
  background:{t["error"]}18;border:1px solid {t["error"]}45;
  border-radius:100px;padding:.25rem .75rem;
  font-size:.72rem;font-weight:700;
  color:{t["error"]} !important;letter-spacing:.02em;
}}
.chip-warn{{
  display:inline-flex;align-items:center;gap:5px;
  background:{t["warning"]}18;border:1px solid {t["warning"]}45;
  border-radius:100px;padding:.25rem .75rem;
  font-size:.72rem;font-weight:700;
  color:{t["warning"]} !important;letter-spacing:.02em;
}}
.chip-info{{
  display:inline-flex;align-items:center;gap:5px;
  background:{t["badge_bg"]};border:1px solid {t["accent"]}35;
  border-radius:100px;padding:.25rem .75rem;
  font-size:.72rem;font-weight:700;
  color:{t["badge_text"]} !important;letter-spacing:.02em;
}}

/* ══════════════════════════════════════════════════════════════════════════
   COLUMN IMAGE LABELS
   ══════════════════════════════════════════════════════════════════════════ */
.col-lbl{{
  font-size:.68rem;font-weight:700;
  color:{t["muted"]} !important;
  text-transform:uppercase;letter-spacing:.12em;
  text-align:center;margin-bottom:8px;display:block;
}}
.img-frame{{
  background:{t["bg2"]};
  border:1px solid {t["border"]};
  border-radius:10px;overflow:hidden;
  transition:border-color .2s ease;
}}
.img-frame:hover{{border-color:{t["accent"]}50;}}

/* ══════════════════════════════════════════════════════════════════════════
   PROCESS STEP FLOW
   ══════════════════════════════════════════════════════════════════════════ */
.step-row{{
  display:flex;align-items:center;gap:0;flex-wrap:wrap;
  margin:1.25rem 0;
}}
.step-box{{
  display:flex;align-items:center;gap:.5rem;
  background:{t["card"]};border:1px solid {t["border"]};
  border-radius:10px;padding:.6rem 1rem;
  font-size:.82rem;font-weight:600;
  color:{t["text"]} !important;
  transition:all .2s ease;
}}
.step-box:hover{{
  border-color:{t["accent"]}60;
  box-shadow:0 4px 12px {t["shadow"]};
}}
.step-num{{
  width:22px;height:22px;
  background:{t["gradient"]};color:#fff !important;
  border-radius:50%;font-size:.7rem;font-weight:800;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}}
.step-arrow{{
  font-size:.75rem;color:{t["muted"]} !important;padding:0 .3rem;
}}

/* ══════════════════════════════════════════════════════════════════════════
   STAT ROW (Analytics)
   ══════════════════════════════════════════════════════════════════════════ */
.stat-card{{
  background:{t["card"]};border:1px solid {t["border"]};
  border-radius:14px;padding:1.25rem 1.5rem;
  transition:all .3s ease;
}}
.stat-card:hover{{
  box-shadow:0 8px 24px {t["shadow"]};
  border-color:{t["accent"]}40;
}}
.stat-num{{
  font-size:2rem;font-weight:900;
  color:{t["text"]} !important;
  letter-spacing:-.04em;
  font-variant-numeric:tabular-nums;
}}
.stat-tag{{
  font-size:.7rem;font-weight:700;
  color:{t["muted"]} !important;
  text-transform:uppercase;letter-spacing:.1em;margin-top:.25rem;
}}

/* ══════════════════════════════════════════════════════════════════════════
   UPLOAD ZONE
   ══════════════════════════════════════════════════════════════════════════ */
.upload-idle{{
  text-align:center;padding:3rem;
  background:{t["card"]};
  border:2px dashed {t["border"]};
  border-radius:16px;
  animation:borderPulse 3s ease infinite;
}}
.upload-icon{{font-size:3.5rem;margin-bottom:1rem;display:block;}}
.upload-label{{
  font-size:1rem;font-weight:700;
  color:{t["text"]} !important;margin-bottom:.4rem;
}}
.upload-hint{{font-size:.84rem;color:{t["muted"]} !important;}}

/* ══════════════════════════════════════════════════════════════════════════
   PROGRESS BAR
   ══════════════════════════════════════════════════════════════════════════ */
.prog-track{{height:6px;background:{t["bg2"]};border-radius:3px;overflow:hidden;}}
.prog-fill{{
  height:100%;border-radius:3px;
  background:{t["gradient"]};transition:width .6s ease;
}}

/* Divider */
.divider{{height:1px;background:{t["border"]};margin:1.5rem 0;}}

/* Animations utility */
.anim-fade{{animation:fadeIn .35s ease forwards;}}
.anim-up  {{animation:fadeInUp .45s ease forwards;}}

/* helper class only — no overrides on Streamlit internals */
.section-block{{margin-bottom:2rem;}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
U_PAT = re.compile(r"^B21\w{3,}$")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_model.pth")
LSH_HASH_SIZE = 64
FEATURE_DIM   = 64 * 64
np.random.seed(42)
LSH_W = np.random.randn(LSH_HASH_SIZE, FEATURE_DIM)
MODALITIES = ["CT Scan", "MRI", "X-Ray", "Ultrasound", "PET Scan",
              "Histopathology", "Fundus", "Dermoscopy", "Unknown"]


def _hp(pw): return hashlib.sha256(pw.encode()).hexdigest()
def _lu():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f: return json.load(f)
    return {}
def _su(u):
    with open(USERS_FILE, "w") as f: json.dump(u, f, indent=2)
def _vu(u):
    if not u.startswith("B21"): return False, "Username must start with **B21**."
    if not U_PAT.match(u):      return False, "At least 6 chars, alphanumeric/underscore."
    return True, ""
def _vp(p):
    if len(p) < 8:                          return False, "≥ 8 characters required."
    if not re.search(r"[A-Za-z]", p):       return False, "Must include a letter."
    if not re.search(r"[0-9@#$%&*!]", p):  return False, "Must include a digit or symbol."
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ic), nn.ReLU(True),
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1),
        )
        self.sc = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x):
        out = self.conv(x)
        if hasattr(self, 'sc'):
            return out + self.sc(x)       # saved model had sc (Identity or Conv)
        if out.shape[1] == x.shape[1]:
            return out + x                # channels match → identity skip
        return out                        # channels differ, no sc → plain conv


class HybridResUNetPlusPlus(nn.Module):
    def __init__(self, ic=1, oc=1):
        super().__init__()
        f = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv0_0=ResBlock(ic,  f[0]); self.conv1_0=ResBlock(f[0],f[1])
        self.conv2_0=ResBlock(f[1],f[2]); self.conv3_0=ResBlock(f[2],f[3])
        self.conv4_0=ResBlock(f[3],f[4])
        self.conv0_1=ResBlock(f[0]+f[1],f[0]); self.conv1_1=ResBlock(f[1]+f[2],f[1])
        self.conv2_1=ResBlock(f[2]+f[3],f[2]); self.conv3_1=ResBlock(f[3]+f[4],f[3])
        self.conv0_2=ResBlock(f[0]*2+f[1],f[0]); self.conv1_2=ResBlock(f[1]*2+f[2],f[1])
        self.conv2_2=ResBlock(f[2]*2+f[3],f[2])
        self.conv0_3=ResBlock(f[0]*3+f[1],f[0]); self.conv1_3=ResBlock(f[1]*3+f[2],f[1])
        self.conv0_4=ResBlock(f[0]*4+f[1],f[0])
        self.final=nn.Conv2d(f[0],oc,1); self.sig=nn.Sigmoid()

    def forward(self, x):
        import torch.nn.functional as F
        # Use pure functions for stateless ops so missing saved-model attrs never crash
        def _pool(t): return F.max_pool2d(t, 2, 2)
        def _up(t):   return F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=True)

        z00=self.conv0_0(x);          z10=self.conv1_0(_pool(z00))
        z20=self.conv2_0(_pool(z10)); z30=self.conv3_0(_pool(z20))
        z40=self.conv4_0(_pool(z30))
        z01=self.conv0_1(torch.cat([z00,_up(z10)],1))
        z11=self.conv1_1(torch.cat([z10,_up(z20)],1))
        z21=self.conv2_1(torch.cat([z20,_up(z30)],1))
        z31=self.conv3_1(torch.cat([z30,_up(z40)],1))
        z02=self.conv0_2(torch.cat([z00,z01,_up(z11)],1))
        z12=self.conv1_2(torch.cat([z10,z11,_up(z21)],1))
        z22=self.conv2_2(torch.cat([z20,z21,_up(z31)],1))
        z03=self.conv0_3(torch.cat([z00,z01,z02,_up(z12)],1))
        z13=self.conv1_3(torch.cat([z10,z11,z12,_up(z22)],1))
        z04=self.conv0_4(torch.cat([z00,z01,z02,z03,_up(z13)],1))
        return torch.sigmoid(self.final(z04))

__main__.ResBlock              = ResBlock
__main__.HybridResUNetPlusPlus = HybridResUNetPlusPlus


@st.cache_resource(show_spinner="Initialising model…")
def load_model():
    m = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    m.eval(); return m


def preprocess(img: np.ndarray) -> torch.Tensor:
    rs = cv2.resize(img, (512, 512))
    return transforms.ToTensor()(rs.astype(np.float32)).unsqueeze(0)


def lsh_hash(roi: np.ndarray) -> np.ndarray:
    r = cv2.resize(roi.astype(np.float32), (64, 64))
    b = cv2.GaussianBlur(r, (5,5), 0)
    f = (r*b).flatten(); f /= np.linalg.norm(f)+1e-8
    return (np.dot(LSH_W, f) > 0).astype(int)


def extract_hash(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    roni = (img*255).astype(np.float64) * (1-(mask>0.5).astype(float))
    LL, _ = pywt.dwt2(roni, "haar")
    return np.array([abs(int(v))%2 for v in LL.flatten()[:LSH_HASH_SIZE]])


def validate(pil: Image.Image):
    if pil.mode in ("RGB","RGBA"):
        arr = np.array(pil.convert("RGB"))
        r,g,b = arr[:,:,0].astype(int),arr[:,:,1].astype(int),arr[:,:,2].astype(int)
        d = (np.mean(np.abs(r-g))+np.mean(np.abs(r-b))+np.mean(np.abs(g-b)))/3
        if d > 8: return False, f"Colour image detected (channel divergence={d:.1f}). Upload a greyscale medical scan.", 0.0
    g = np.array(pil.convert("L")).astype(np.float32)/255.
    std,mean,dark = g.std(),g.mean(),float(np.mean(g<0.08))
    if std  < 0.04: return False, f"Contrast too low (σ={std:.3f}). Upload a scan with visible anatomy.", 0.0
    if mean > 0.88: return False, f"Image appears over-exposed (mean={mean:.2f}).", 0.0
    if dark < 0.05: return False, f"No dark background (ratio={dark:.1%}). This does not resemble a medical scan.", 0.0
    conf = min(1., dark*3)*min(1., std*6)*(1.-max(0., mean-.6))
    return True, "", float(conf)


def detect_modality(g: np.ndarray) -> str:
    std,mean = np.std(g),np.mean(g)
    dark,bright = np.mean(g<0.05),np.mean(g>0.85)
    if bright>0.10 and dark>0.40: return "X-Ray"
    if dark>0.60 and std>0.15:    return "CT Scan"
    if dark>0.50 and std<0.15:    return "MRI"
    if mean>0.45 and std<0.20:    return "Ultrasound"
    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _pc(): return T()["bg2"]   # plot background color
def _tc(): return T()["text"]  # plot text color

def _rgba(hex_col: str, opacity: float) -> str:
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"

def gauge_chart(value: float) -> go.Figure:
    t = T()
    color = t["success"] if value>=.70 else (t["warning"] if value>=.50 else t["error"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value*100,1),
        number={"suffix":"%","font":{"size":32,"color":t["text"],"family":"Inter"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":t["muted"],"tickfont":{"color":t["muted"],"size":10}},
            "bar":{"color":color,"thickness":.28},
            "bgcolor":t["bg2"],"bordercolor":t["border"],"borderwidth":1,
            "steps":[
                {"range":[0,50],  "color":_rgba(t["error"],   0.09)},
                {"range":[50,70], "color":_rgba(t["warning"], 0.09)},
                {"range":[70,100],"color":_rgba(t["success"], 0.09)},
            ],
            "threshold":{"line":{"color":color,"width":3},"value":value*100},
        },
        title={"text":"Hash Similarity","font":{"color":t["muted"],"size":12,"family":"Inter"}},
    ))
    fig.update_layout(height=220, margin=dict(t=48,b=10,l=20,r=20),
                      paper_bgcolor=t["bg2"], font_color=t["text"])
    return fig


def donut_chart(auth:int, susp:int, tamp:int) -> go.Figure:
    t = T()
    total = auth+susp+tamp or 1
    fig = go.Figure(go.Pie(
        labels=["Authentic","Suspicious","Tampered"],
        values=[auth,susp,tamp],
        hole=.65,
        marker_colors=[t["success"],t["warning"],t["error"]],
        textfont=dict(color=t["text"],family="Inter"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        height=280, margin=dict(t=20,b=20,l=20,r=20),
        paper_bgcolor=t["bg2"], plot_bgcolor=t["bg2"],
        font_color=t["text"], font_family="Inter",
        showlegend=True,
        legend=dict(font=dict(color=t["text"],size=11), bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:10px'>Total</span>",
                          x=0.5,y=0.5,font_size=18,font_color=t["text"],showarrow=False)]
    )
    return fig


def histogram_chart(values: list) -> go.Figure:
    t = T()
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=20,
        marker_color=t["accent"], opacity=0.85,
        marker_line=dict(color=t["bg"],width=.5),
    ))
    fig.update_layout(
        height=240, margin=dict(t=10,b=40,l=40,r=10),
        paper_bgcolor=t["bg2"], plot_bgcolor=t["bg2"],
        font_color=t["text"], font_family="Inter",
        xaxis=dict(title="Similarity Score",gridcolor=t["border"],color=t["muted"]),
        yaxis=dict(title="Count",gridcolor=t["border"],color=t["muted"]),
        bargap=0.05,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_login():
    t = T()
    _, c, _ = st.columns([1,2,1])
    with c:
        st.markdown("""
        <div class="login-wrap">
          <span class="login-logo">🧬</span>
          <p class="login-title">MedGuard <span class="gradient-text">AI</span></p>
          <p class="login-tagline">Medical Image Integrity Detection Platform</p>
          <p style="text-align:center;margin-bottom:1.5rem;">
            <span class="login-rule">🔐 Usernames must begin with B21</span>
          </p>
        </div>
        """, unsafe_allow_html=True)

        tab_l, tab_s = st.tabs(["  Sign In  ", "  Create Account  "])

        with tab_l:
            with st.form("lf", clear_on_submit=False):
                user = st.text_input("Username", placeholder="B21yourname")
                pw   = st.text_input("Password", type="password", placeholder="Enter password")
                st.markdown("<br>", unsafe_allow_html=True)
                sub  = st.form_submit_button("Sign In →", use_container_width=True)
            if sub:
                users = _lu()
                if user in users and users[user] == _hp(pw):
                    st.session_state.update(authenticated=True, current_user=user,
                                            active_page="Home")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

        with tab_s:
            with st.form("sf", clear_on_submit=True):
                nu  = st.text_input("Username", placeholder="B21yourname", key="su_u")
                np_ = st.text_input("Password", type="password",
                                    placeholder="Min 8 chars, include a digit/symbol", key="su_p")
                np2 = st.text_input("Confirm Password", type="password",
                                    placeholder="Re-enter password", key="su_p2")
                st.markdown("<br>", unsafe_allow_html=True)
                reg = st.form_submit_button("Create Account →", use_container_width=True)
            if reg:
                u_ok,u_e = _vu(nu); p_ok,p_e = _vp(np_); users = _lu()
                if not u_ok:           st.error(u_e)
                elif not p_ok:         st.error(p_e)
                elif np_ != np2:       st.error("Passwords do not match.")
                elif nu in users:      st.error(f"**{nu}** is already taken.")
                else:
                    users[nu]=_hp(np_); _su(users)
                    st.success(f"Account **{nu}** created. Sign in now.")

    st.markdown(f"""<p style="text-align:center;color:{t['muted2']};font-size:.72rem;
        margin-top:2rem;">Protected system · Unauthorized access is prohibited</p>""",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    t = T()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-section">
      <div class="hero-eyebrow">
        <span class="pulse-dot"></span>
        AI-Powered · Deep Learning · Medical Imaging
      </div>
      <h1 class="hero-title">
        Medical Image<br>
        <span class="gradient-text">Integrity Platform</span>
      </h1>
      <p class="hero-sub">
        An advanced deep learning system that detects tampering in medical scans
        using segmentation, locality-sensitive hashing, and wavelet-based steganography —
        preserving diagnostic authenticity in every pixel.
      </p>
      <div class="hero-divider"></div>
      <div class="hero-kpis">
        <div class="kpi-item">
          <span class="kpi-val">99.60%</span>
          <span class="kpi-lbl">Segmentation Accuracy</span>
        </div>
        <div class="kpi-item">
          <span class="kpi-val">38.79 dB</span>
          <span class="kpi-lbl">PSNR Score</span>
        </div>
        <div class="kpi-item">
          <span class="kpi-val">0.985</span>
          <span class="kpi-lbl">NCC Coefficient</span>
        </div>
        <div class="kpi-item">
          <span class="kpi-val">64-bit</span>
          <span class="kpi-lbl">LSH Hash Size</span>
        </div>
        <div class="kpi-item">
          <span class="kpi-val">0.939</span>
          <span class="kpi-lbl">LSH mAP Score</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Pipeline Steps ─────────────────────────────────────────────────────────
    st.markdown("""
    <p class="sec-title">System Pipeline</p>
    <p class="sec-sub">Four-stage processing from ingestion to verified verdict</p>
    <div class="step-row">
      <div class="step-box"><span class="step-num">1</span>Upload Scan</div>
      <span class="step-arrow">→</span>
      <div class="step-box"><span class="step-num">2</span>HResUNet++ Segmentation</div>
      <span class="step-arrow">→</span>
      <div class="step-box"><span class="step-num">3</span>LSH Hash + DWT-LSB Embed</div>
      <span class="step-arrow">→</span>
      <div class="step-box"><span class="step-num">4</span>Hash Extraction & Verdict</div>
      <span class="step-arrow">→</span>
      <div class="step-box"><span class="step-num">5</span>Grad-CAM Report</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Cards ──────────────────────────────────────────────────────────
    st.markdown("""<p class="sec-title" style="margin-top:1.5rem;">Core Capabilities</p>
    <p class="sec-sub">Each module is purpose-built for clinical-grade image verification</p>""",
    unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    features = [
        ("🧠","Hybrid Segmentation",
         "ResBlock encoder-decoder with U-Net++ nested skip connections isolates ROI with 99.60% accuracy.",
         "HResUNet++"),
        ("🔑","LSH Watermarking",
         "64-bit Locality-Sensitive Hash fingerprints the critical region with sub-millisecond compute.",
         "64-bit Hash"),
        ("📡","DWT-LSB Steganography",
         "Haar wavelet decomposes RONI into sub-bands; hash embedded invisibly via Least Significant Bit.",
         "Haar DWT"),
        ("🔍","Grad-CAM Explainability",
         "Gradient-weighted Class Activation Maps visualise the encoder's attention on each scan.",
         "XAI Ready"),
    ]
    for col,(icon,title,desc,tag) in zip([c1,c2,c3,c4],features):
        with col:
            st.markdown(f"""
            <div class="pcard">
              <div class="pcard-top-bar"></div>
              <span class="pcard-icon">{icon}</span>
              <p class="pcard-title">{title}</p>
              <p class="pcard-desc">{desc}</p>
              <span class="pcard-tag">{tag}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Second row ─────────────────────────────────────────────────────────────
    r1,r2,r3,r4 = st.columns(4)
    row2 = [
        ("⚔️","Attack Robustness",
         "Validated against JPEG compression, Gaussian noise, geometric transforms, and deletion attacks.",
         "Multi-Attack"),
        ("📄","Clinical PDF Reports",
         "One-click ReportLab-generated reports with patient info, metrics, verdict, and methodology.",
         "PDF Export"),
        ("📋","Prediction History",
         "All analyses are logged with timestamps, metrics, and verdicts for audit and compliance.",
         "Audit Trail"),
        ("🌙","Adaptive Theme",
         "Seamless dark and light mode for clinical environments and presentation settings.",
         "Dark / Light"),
    ]
    for col,(icon,title,desc,tag) in zip([r1,r2,r3,r4],row2):
        with col:
            st.markdown(f"""
            <div class="pcard">
              <div class="pcard-top-bar"></div>
              <span class="pcard-icon">{icon}</span>
              <p class="pcard-title">{title}</p>
              <p class="pcard-desc">{desc}</p>
              <span class="pcard-tag">{tag}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    # ── Accepted Modalities ────────────────────────────────────────────────────
    st.markdown("""<p class="sec-title">Accepted Imaging Modalities</p>""", unsafe_allow_html=True)
    mods = ["🫁 CT Scan","🧲 MRI","☢️ X-Ray","🔊 Ultrasound",
            "⚡ PET Scan","🔬 Histopathology","👁️ Fundus","🩺 Dermoscopy"]
    mc = st.columns(8)
    for col, mod in zip(mc, mods):
        with col:
            st.markdown(f"""
            <div style="text-align:center;background:{t['card']};border:1px solid {t['border']};
                 border-radius:12px;padding:.875rem .5rem;transition:all .2s ease;">
              <div style="font-size:1.6rem;margin-bottom:.4rem;">{mod.split()[0]}</div>
              <div style="font-size:.68rem;font-weight:700;color:{t['muted']};
                   text-transform:uppercase;letter-spacing:.06em;">
                {' '.join(mod.split()[1:])}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION LAB PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_lab():
    t = T()
    st.markdown("""
    <div>
      <p class="sec-title">🔬 Detection Lab</p>
      <p class="sec-sub">Upload a medical scan to run the full integrity verification pipeline</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Patient Info ─────────────────────────────────────────────────────────
    with st.expander("👤  Patient Information Panel", expanded=False):
        pi1,pi2,pi3 = st.columns(3)
        patient_name = pi1.text_input("Patient Name", placeholder="Anonymous")
        patient_id   = pi2.text_input("Patient ID",   placeholder="MRN-XXXX")
        scan_type    = pi3.selectbox("Scan Modality", MODALITIES)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "**Upload Medical Scan** — PNG · JPG · JPEG (greyscale only)",
        type=["png","jpg","jpeg"],
    )

    if not uploaded:
        st.markdown(f"""
        <div class="upload-idle">
          <span class="upload-icon">🫁</span>
          <p class="upload-label">Drag & Drop Medical Scan Here</p>
          <p class="upload-hint">Supports CT, MRI, X-Ray, Ultrasound — greyscale images only<br>
            PNG · JPG · JPEG &nbsp;·&nbsp; Max 50 MB</p>
        </div>""", unsafe_allow_html=True)
        return

    # ── Validate ──────────────────────────────────────────────────────────────
    img_pil = Image.open(uploaded)
    valid, err, conf = validate(img_pil)

    if not valid:
        st.markdown(f"""
        <div style="background:{t['error']}12;border:1.5px solid {t['error']}40;
             border-radius:14px;padding:1.25rem 1.5rem;margin-bottom:1rem;">
          <p style="font-size:.9rem;font-weight:700;color:{t['error']};margin-bottom:.35rem;">
            ❌ Validation Failed</p>
          <p style="font-size:.84rem;color:{t['text']};margin:0;">{err}</p>
        </div>""", unsafe_allow_html=True)
        return

    img_grey  = img_pil.convert("L")
    img_np    = np.array(img_grey)
    img_float = img_np.astype(np.float32)/255.
    auto_mod  = detect_modality(img_float)
    if scan_type == "Unknown": scan_type = auto_mod

    v1,v2,v3 = st.columns(3)
    v1.success(f"✅ Validation passed ({conf:.0%} confidence)")
    v2.info(f"🔬 Detected: **{auto_mod}**")
    v3.info(f"📐 Resolution: **{img_np.shape[1]} × {img_np.shape[0]} px**")

    # ── Load Model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        st.error("`full_model.pth` not found. Place model weights next to `app.py`.")
        return
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model load error: {e}")
        return

    # ── Segmentation ──────────────────────────────────────────────────────────
    with st.spinner("Running HResUNet++ segmentation…"):
        tensor  = preprocess(img_float)
        with torch.no_grad():
            raw = model(tensor).squeeze().numpy()
        mask    = (raw > 0.5).astype(np.float32)
        img_512 = cv2.resize(img_float,(512,512))
        roi_r   = float(np.mean(mask))

    if roi_r < 0.005 or roi_r > 0.95:
        # Fallback: circular central ROI covering ~30% of image
        h, w = mask.shape
        cy, cx = h // 2, w // 2
        r = int(min(h, w) * 0.31)
        Y, X = np.ogrid[:h, :w]
        mask = ((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2).astype(np.float32)
        roi_r = float(np.mean(mask))
        st.warning(f"⚠️ Segmentation coverage unusual — using default central ROI ({roi_r:.1%}).")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    cam_rgb = None
    with st.spinner("Generating Grad-CAM attention map…"):
        try:
            gcam    = GradCAM(model, model.conv4_0)
            cam     = gcam.generate(tensor.clone(), target_size=(512,512))
            cam_rgb = overlay_heatmap(img_512, cam)
        except Exception:
            pass

    # ── Hash Comparison ───────────────────────────────────────────────────────
    with st.spinner("Extracting & verifying watermark hash…"):
        extracted    = extract_hash(img_512, mask)
        recalculated = lsh_hash(img_512 * mask)
        similarity   = float(np.mean(extracted == recalculated))

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = all_metrics(img_512, img_512,
                          original_bits=recalculated,
                          extracted_bits=extracted)
    metrics["Hash Similarity"] = round(similarity, 4)
    metrics["ROI Coverage"]    = round(roi_r, 4)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if similarity >= 0.70:
        verdict="✅ AUTHENTIC"; vclass="verdict-ok"
        vdesc="Hash integrity confirmed. No evidence of tampering detected."
        vchip=f'<span class="chip-ok">● AUTHENTIC</span>'
    elif similarity >= 0.50:
        verdict="⚠️ SUSPICIOUS"; vclass="verdict-warn"
        vdesc="Partial hash match. Manual clinical review recommended."
        vchip=f'<span class="chip-warn">● SUSPICIOUS</span>'
    else:
        verdict="🚨 TAMPERED"; vclass="verdict-bad"
        vdesc="Hash mismatch detected. Image integrity cannot be confirmed."
        vchip=f'<span class="chip-bad">● TAMPERED</span>'

    # ── 5-Panel Display ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
      <p class="sec-title" style="margin:0;">Analysis Results</p>
      {vchip}
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    labels = ["Original","ROI Mask","RONI Region","Grad-CAM","Verdict"]
    for col,lbl in zip([c1,c2,c3,c4,c5],labels):
        with col:
            st.markdown(f'<span class="col-lbl">{lbl}</span>', unsafe_allow_html=True)

    with c1: st.image(img_grey,        use_container_width=True)
    with c2: st.image(mask,             use_container_width=True, clamp=True)
    with c3: st.image(img_512*(1-mask), use_container_width=True, clamp=True)
    with c4:
        if cam_rgb is not None: st.image(cam_rgb, use_container_width=True)
        else: st.info("Unavailable")
    with c5:
        st.markdown(f'<div class="{vclass}"><p class="verdict-head">{verdict}</p>'
                    f'<p class="verdict-body">{vdesc}</p></div>', unsafe_allow_html=True)

    # ── Gauge + Metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    gc, mc = st.columns([1, 2])

    with gc:
        st.plotly_chart(gauge_chart(similarity), use_container_width=True,
                        config={"displayModeBar":False})

    with mc:
        st.markdown('<p class="sec-title">Quality Metrics</p>', unsafe_allow_html=True)
        flat = list(metrics.items())
        r1c  = st.columns(3); r2c = st.columns(3)
        for i,(k,v) in enumerate(flat[:6]):
            (r1c[i%3] if i<3 else r2c[i%3]).metric(k, str(v))

    # ── Bit comparison ────────────────────────────────────────────────────────
    with st.expander("🔬  Detailed Diagnostics & Bit-Level Comparison"):
        d1,d2,d3,d4,d5 = st.columns(5)
        d1.metric("Image Size",       f"{img_np.shape[1]}×{img_np.shape[0]}")
        d2.metric("Mean Intensity",   f"{img_float.mean():.3f}")
        d3.metric("Std Deviation",    f"{img_float.std():.3f}")
        d4.metric("Dark Pixel Ratio", f"{np.mean(img_float<0.08):.1%}")
        d5.metric("Bits Matched",     f"{int(similarity*LSH_HASH_SIZE)}/{LSH_HASH_SIZE}")
        st.markdown("<br>**LSH Bit-Level Comparison** (first 32 bits):", unsafe_allow_html=True)
        bits_html = "".join(
            f'<span style="font-family:monospace;font-size:.9rem;font-weight:700;'
            f'color:{"#10B981" if e==r else "#EF4444"};padding:1px 2px;">{e}</span>'
            for e,r in zip(extracted[:32],recalculated[:32])
        )
        st.markdown(f'<div style="letter-spacing:.15rem;line-height:2;">{bits_html}</div>',
                    unsafe_allow_html=True)

    # ── Attack Simulation ─────────────────────────────────────────────────────
    with st.expander("⚔️  Attack Robustness Simulation"):
        st.markdown(f'<p class="sec-sub">Simulate common image attacks and measure hash degradation</p>',
                    unsafe_allow_html=True)
        ac = st.columns(4)

        def jpeg_atk(im):
            from io import BytesIO
            buf=BytesIO()
            Image.fromarray((im*255).astype(np.uint8)).save(buf,format="JPEG",quality=50)
            buf.seek(0)
            return np.array(Image.open(buf).convert("L")).astype(np.float32)/255.

        attacks=[
            ("JPEG Q=50",   jpeg_atk),
            ("Gauss Noise", lambda im: np.clip(im+np.random.normal(0,.05,im.shape).astype(np.float32),0,1)),
            ("Blur σ=2",    lambda im: cv2.GaussianBlur(im,(5,5),0)),
            ("Crop 12.5%",  lambda im: (lambda c: (c.__setitem__((slice(None,im.shape[0]//8),slice(None)),0)) or c)(im.copy())),
        ]
        for col,(name,fn) in zip(ac,attacks):
            with col:
                try:
                    ai = fn(img_512)
                    with torch.no_grad():
                        am = (model(preprocess(ai)).squeeze().numpy()>0.5).astype(np.float32)
                    ae  = extract_hash(ai,am)
                    sim = float(np.mean(ae==recalculated))
                    color = t["success"] if sim>=.7 else (t["warning"] if sim>=.5 else t["error"])
                    bar   = int(sim*100)
                    st.markdown(f"""
                    <div class="flat-card" style="text-align:center;padding:1rem;">
                      <p style="font-size:.8rem;font-weight:700;color:{t['text']};margin-bottom:.5rem;">{name}</p>
                      <p style="font-size:1.5rem;font-weight:900;color:{color};
                           letter-spacing:-.03em;margin-bottom:.5rem;">{sim:.0%}</p>
                      <div class="prog-track">
                        <div class="prog-fill" style="width:{bar}%;background:{color};"></div>
                      </div>
                      <p style="font-size:.68rem;color:{t['muted']};margin-top:.4rem;">hash similarity</p>
                    </div>""", unsafe_allow_html=True)
                except Exception as ex:
                    st.warning(f"{name}: {ex}")

    # ── Save to history ───────────────────────────────────────────────────────
    add_record(st.session_state.get("current_user","unknown"),
               uploaded.name, verdict, similarity, roi_r, metrics)

    # ── PDF Download ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="sec-title">Export Report</p>', unsafe_allow_html=True)
    if REPORTLAB_OK:
        pdf = generate_pdf(patient_name, patient_id, scan_type, uploaded.name,
                           verdict, similarity, roi_r, metrics,
                           st.session_state.get("current_user","analyst"))
        if pdf:
            st.download_button("📄  Download Clinical PDF Report", data=pdf,
                               file_name=f"MedGuard_{uploaded.name}.pdf",
                               mime="application/pdf", use_container_width=True)
    else:
        st.info("Install `reportlab` (`pip install reportlab`) to enable PDF export.")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    t = T()
    st.markdown("""
    <div>
      <p class="sec-title">📊 Analytics</p>
      <p class="sec-sub">Aggregated insights across all detection runs</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    user     = st.session_state.get("current_user")
    show_all = st.checkbox("Show all users' records", value=False)
    records  = get_records(username=None if show_all else user)

    if not records:
        st.markdown(f"""
        <div class="flat-card" style="text-align:center;padding:3rem;">
          <span style="font-size:3rem;">📭</span><br><br>
          <p style="color:{t['muted']};">No analysis records found. Run a detection first.</p>
        </div>""", unsafe_allow_html=True)
        return

    auth  = sum(1 for r in records if "AUTHENTIC"  in r["verdict"])
    tamp  = sum(1 for r in records if "TAMPERED"   in r["verdict"])
    susp  = len(records)-auth-tamp
    sims  = [r["similarity"] for r in records]
    avg_s = sum(sims)/len(sims) if sims else 0

    # Stats row
    s1,s2,s3,s4,s5 = st.columns(5)
    for col,(lbl,val,delta) in zip([s1,s2,s3,s4,s5],[
        ("Total Analyses",     len(records), None),
        ("Authentic",          auth,         None),
        ("Tampered",           tamp,         None),
        ("Suspicious",         susp,         None),
        ("Avg Similarity",     f"{avg_s:.1%}", None),
    ]):
        col.metric(lbl, val)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 2])

    with left:
        st.markdown('<p class="sec-title">Verdict Distribution</p>', unsafe_allow_html=True)
        st.plotly_chart(donut_chart(auth,susp,tamp), use_container_width=True,
                        config={"displayModeBar":False})

    with right:
        st.markdown('<p class="sec-title">Similarity Score Distribution</p>', unsafe_allow_html=True)
        st.plotly_chart(histogram_chart(sims), use_container_width=True,
                        config={"displayModeBar":False})

    st.markdown("---")
    st.markdown('<p class="sec-title">Prediction History</p>', unsafe_allow_html=True)

    for rec in records[:50]:
        v = rec["verdict"]
        chip = (f'<span class="chip-ok">● AUTHENTIC</span>'   if "AUTHENTIC"  in v else
                f'<span class="chip-bad">● TAMPERED</span>'   if "TAMPERED"   in v else
                f'<span class="chip-warn">● SUSPICIOUS</span>')
        with st.expander(f"{rec['timestamp']}  ·  {rec['filename']}"):
            ec1,ec2,ec3,ec4 = st.columns(4)
            ec1.metric("Analyst",    rec["username"])
            ec2.metric("Similarity", f"{rec['similarity']:.1%}")
            ec3.metric("ROI",        f"{rec['roi_ratio']:.1%}")
            ec4.markdown(f"<br>{chip}", unsafe_allow_html=True)
            if rec.get("metrics"):
                mx = list(rec["metrics"].items())
                mc2 = st.columns(min(len(mx),5))
                for i,(k,v) in enumerate(mx[:5]):
                    mc2[i].metric(k,str(v))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️  Clear My Records", type="secondary"):
        clear_records(username=user)
        st.success("History cleared."); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH HUB PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_research():
    t = T()
    st.markdown("""
    <div>
      <p class="sec-title">📚 Research Hub</p>
      <p class="sec-sub">Technical benchmarks, model comparisons, and architecture specifications</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # Performance KPIs
    st.markdown('<p class="sec-title">System Performance Benchmarks</p>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    benchmarks = [("38.79 dB","PSNR"),("99.60%","Seg. Accuracy"),
                  ("0.985","NCC"),("0.9107","BER"),
                  ("0.939","LSH mAP @ 32-bit"),("99.65%","Specificity")]
    for col,(val,lbl) in zip([k1,k2,k3,k4,k5,k6],benchmarks):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="text-align:center;">
              <div class="stat-num">{val}</div>
              <div class="stat-tag">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "  Segmentation  ","  Embedding  ","  Hashing  ","  Architecture  "
    ])

    with tab1:
        st.markdown('<p class="sec-sub">Segmentation model comparison on medical CT dataset</p>',
                    unsafe_allow_html=True)
        df1 = pd.DataFrame({
            "Model":        ["FCN","PSPNet","DeepLab V3+","HResUNet++ (Ours)"],
            "Accuracy (%)": [91.59, 94.51,  96.36,         99.60],
            "Precision (%)": [91.20, 94.00, 95.82,         99.63],
            "F1-Score (%)":  [91.30, 94.30, 96.10,         99.59],
            "Recall (%)":    [91.80, 95.00, 96.79,         99.54],
            "Specificity (%)": [90.80,93.50,95.60,         99.65],
        })
        st.dataframe(df1.style.highlight_max(subset=df1.columns[1:],
                     color=f"{t['accent']}30"), use_container_width=True, hide_index=True)

        fig_seg = go.Figure()
        metrics_seg = ["Accuracy (%)","Precision (%)","F1-Score (%)","Recall (%)","Specificity (%)"]
        colors = [t["muted"]]*3+[t["accent"]]
        for i,(model,color) in enumerate(zip(df1["Model"],colors)):
            fig_seg.add_trace(go.Bar(
                name=model, x=metrics_seg,
                y=df1.iloc[i][1:].values, marker_color=color,
                hovertemplate=f"<b>{model}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>"
            ))
        fig_seg.update_layout(
            barmode="group", height=320, margin=dict(t=20,b=40,l=40,r=10),
            paper_bgcolor=t["bg2"], plot_bgcolor=t["bg2"],
            font_color=t["text"], font_family="Inter",
            xaxis=dict(gridcolor=t["border"]),
            yaxis=dict(range=[88,101],gridcolor=t["border"]),
            legend=dict(font=dict(color=t["text"]),bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_seg, use_container_width=True, config={"displayModeBar":False})

    with tab2:
        st.markdown('<p class="sec-sub">Embedding technique comparison across image quality metrics</p>',
                    unsafe_allow_html=True)
        df2 = pd.DataFrame({
            "Method":   ["LWT","PVD","LSBMR","Proposed (DWT-LSB)"],
            "PSNR (dB)":[29.32, 30.49, 33.52, 38.79],
            "NC":       [0.0006,0.2325,0.6392, 0.9858],
            "BER":      [0.2128,0.3188,0.8008, 0.9107],
            "NAE":      [0.5335,0.3788,0.1775, 0.1183],
            "TAF":      [0.1183,0.3788,0.1775, 0.1183],
        })
        st.dataframe(df2.style.highlight_max(
            subset=["PSNR (dB)","NC","BER"], color=f"{t['accent']}30"
        ).highlight_min(
            subset=["NAE"], color=f"{t['success']}30"
        ), use_container_width=True, hide_index=True)

        fig_emb = go.Figure()
        fig_emb.add_trace(go.Bar(
            name="PSNR (dB)", x=df2["Method"], y=df2["PSNR (dB)"],
            marker_color=[t["muted"],t["muted"],t["muted"],t["accent"]],
            hovertemplate="<b>%{x}</b><br>PSNR: %{y:.2f} dB<extra></extra>"
        ))
        fig_emb.update_layout(
            height=280, margin=dict(t=10,b=40,l=40,r=10),
            paper_bgcolor=t["bg2"], plot_bgcolor=t["bg2"],
            font_color=t["text"], font_family="Inter",
            xaxis=dict(gridcolor=t["border"]),
            yaxis=dict(title="PSNR (dB)",gridcolor=t["border"]),
            showlegend=False,
        )
        st.plotly_chart(fig_emb, use_container_width=True, config={"displayModeBar":False})

    with tab3:
        st.markdown('<p class="sec-sub">Hashing method mAP at different bit lengths</p>',
                    unsafe_allow_html=True)
        df3 = pd.DataFrame({
            "Method":         ["LCMH","Spectral Hashing","Proposed LSH"],
            "8-bit mAP":      [0.787,  0.885,            0.921],
            "16-bit mAP":     [0.827,  0.876,            0.936],
            "32-bit mAP":     [0.814,  0.888,            0.939],
            "64-bit mAP":     [0.819,  0.869,            0.928],
            "Compute @ 64-bit (s)":[0.175, 0.063,        0.058],
        })
        st.dataframe(df3.style.highlight_max(
            subset=[c for c in df3.columns if "mAP" in c], color=f"{t['accent']}30"
        ).highlight_min(
            subset=["Compute @ 64-bit (s)"], color=f"{t['success']}30"
        ), use_container_width=True, hide_index=True)

        bits = ["8-bit","16-bit","32-bit","64-bit"]
        fig_lsh = go.Figure()
        clrs = [t["muted2"],t["muted"],t["accent"]]
        for (_, row), col in zip(df3.iterrows(), clrs):
            fig_lsh.add_trace(go.Scatter(
                x=bits, y=[row["8-bit mAP"],row["16-bit mAP"],row["32-bit mAP"],row["64-bit mAP"]],
                name=row["Method"], line=dict(color=col,width=2.5),
                marker=dict(size=8,color=col),
                hovertemplate=f"<b>{row['Method']}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>"
            ))
        fig_lsh.update_layout(
            height=260, margin=dict(t=10,b=40,l=40,r=10),
            paper_bgcolor=t["bg2"], plot_bgcolor=t["bg2"],
            font_color=t["text"], font_family="Inter",
            xaxis=dict(gridcolor=t["border"]),
            yaxis=dict(title="mAP",gridcolor=t["border"],range=[0.75,0.96]),
            legend=dict(font=dict(color=t["text"]),bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_lsh, use_container_width=True, config={"displayModeBar":False})

    with tab4:
        st.markdown('<p class="sec-sub">Technical architecture and component specifications</p>',
                    unsafe_allow_html=True)
        ac1, ac2 = st.columns(2)
        with ac1:
            specs = [
                ("Segmentation Network","Hybrid Residual U-Net++"),
                ("Input Resolution",    "512 × 512 px (grayscale)"),
                ("Encoder Depth",       "5 levels (f=[32,64,128,256,512])"),
                ("Skip Connections",    "Dense nested (U-Net++ style)"),
                ("Activation",          "ReLU + Sigmoid (output)"),
                ("Hash Algorithm",      "Locality-Sensitive Hashing"),
                ("Hash Dimensions",     "64-bit (k=64 planes)"),
                ("LSH Features",        "64×64 Gaussian-weighted"),
                ("Wavelet",             "Haar DWT (2-level)"),
                ("Embedding",           "LSB in LL sub-band"),
                ("Framework",           "PyTorch 2.x"),
                ("Explainability",      "Grad-CAM on conv4_0"),
            ]
            for k,v in specs:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                     padding:.625rem .875rem;border-bottom:1px solid {t['border']};">
                  <span style="font-size:.82rem;font-weight:600;color:{t['muted']};">{k}</span>
                  <span style="font-size:.82rem;font-weight:700;color:{t['text']};">{v}</span>
                </div>""", unsafe_allow_html=True)
        with ac2:
            st.markdown("""
            <div class="pcard">
              <p class="pcard-title">🔬 Pipeline Overview</p>
              <div style="line-height:2.2;font-size:.875rem;">
                <div>📥 <b>Input:</b> Greyscale DICOM/PNG/JPG medical scan</div>
                <div>🧠 <b>Segmentation:</b> HResUNet++ → ROI + RONI masks</div>
                <div>🔑 <b>Hashing:</b> LSH fingerprint of ROI (64-bit)</div>
                <div>🌊 <b>Decomposition:</b> 2-level Haar DWT on RONI</div>
                <div>📦 <b>Embedding:</b> LSB insertion into LL sub-band</div>
                <div>📤 <b>Transmission:</b> Watermarked stego-image</div>
                <div>🔓 <b>Extraction:</b> Hash retrieved from RONI</div>
                <div>✅ <b>Verification:</b> WEC threshold comparison</div>
                <div>📊 <b>Report:</b> Verdict + Grad-CAM + PDF</div>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    t = T()
    with st.sidebar:
        st.markdown(f"""
        <div class="sb-brand">
          <span class="sb-logo">🧬</span>
          <div>
            <div class="sb-name">MedGuard AI</div>
            <div class="sb-tag">Integrity Detection Platform</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Theme toggle
        dark = st.checkbox("🌙  Dark Mode", value=st.session_state.get("dark_mode", True))
        if dark != st.session_state.get("dark_mode", True):
            st.session_state["dark_mode"] = dark
            st.rerun()

        st.markdown("---")
        st.markdown('<span class="nav-section">Navigation</span>', unsafe_allow_html=True)

        nav_items = [
            ("🏠","Home"),
            ("🔬","Detection Lab"),
            ("📊","Analytics"),
            ("📚","Research Hub"),
        ]
        for icon,page in nav_items:
            active = st.session_state.get("active_page","Home") == page
            label  = f"{'▶  ' if active else '     '}{icon}  {page}"
            if st.button(label, key=f"nav_{page}", use_container_width=True):
                st.session_state["active_page"] = page
                st.rerun()

        st.markdown("---")
        st.markdown('<span class="nav-section">Session</span>', unsafe_allow_html=True)

        # Quick stats
        user = st.session_state.get("current_user","unknown")
        recs = get_records(username=user)
        auth = sum(1 for r in recs if "AUTHENTIC" in r["verdict"])
        st.markdown(f"""
        <div class="user-pill">
          <div class="user-pill-name">
            <span class="status-dot"></span>{user}
          </div>
          <div class="user-pill-role">
            {len(recs)} analyses · {auth} authentic
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬡  Sign Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        st.markdown(f"""<div style="font-size:.65rem;color:{t['muted2']};text-align:center;
            line-height:1.8;">HResUNet++ · LSH · DWT-LSB · Grad-CAM<br>
            Final Year Engineering Project · v3.0</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("active_page",   "Home")
    st.session_state.setdefault("dark_mode",     True)

    inject_css()

    if not st.session_state["authenticated"]:
        page_login()
        return

    render_sidebar()

    page = st.session_state.get("active_page","Home")
    dispatch = {
        "Home":          page_home,
        "Detection Lab": page_lab,
        "Analytics":     page_analytics,
        "Research Hub":  page_research,
    }
    dispatch.get(page, page_home)()


if __name__ == "__main__":
    main()
