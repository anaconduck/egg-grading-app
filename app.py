import io
import base64
import time
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import torchvision
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from PIL import Image
from collections import Counter
from torchvision import transforms
from matplotlib import pyplot as plt

# ==============================
# 🎨 PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Cek Kualitas Telur 🥚",
    layout="wide",
    page_icon="🥚",
    initial_sidebar_state="collapsed"
)

# ==============================
# 🖼️ LOAD BACKGROUND AS BASE64
# ==============================
def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

bg_path = Path(__file__).parent / "Assets" / "farm_bg.png"
bg_b64 = get_base64_image(str(bg_path))
bg_css = f"url('data:image/png;base64,{bg_b64}')" if bg_b64 else "linear-gradient(135deg, #78c843 0%, #4a9b30 100%)"

# ==============================
# 🎨 MASTER CSS STYLING
# ==============================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

/* ── BASE ── */
*, *::before, *::after {{ box-sizing: border-box; }}

html, body {{
    font-family: 'Nunito', sans-serif;
    margin: 0; padding: 0;
    overflow-x: hidden;
}}

/* ── FULL PAGE BACKGROUND ── */
[data-testid="stAppViewContainer"] {{
    background-color: #f5f0e8;
    font-family: 'Nunito', sans-serif;
    overflow-x: hidden;
}}

/* ═══════════════════════════════════════════
   STREAMLIT LAYOUT RESET — ALL LAYERS
   ═══════════════════════════════════════════ */

/* L1 — outermost stApp wrapper: prevents blank space below footer */
[data-testid="stApp"] {{
    min-height: unset !important;
    overflow-x: hidden;
}}

/* L2 — stAppViewContainer > .main */
[data-testid="stAppViewContainer"] > .main {{
    background: transparent;
    padding: 0 !important;
    margin: 0 !important;
}}

/* L3 — stMain <section> */
[data-testid="stMain"] {{
    padding: 0 !important;
    margin: 0 !important;
    overflow-x: hidden;
}}

/* L4 — block-container & stMainBlockContainer */
.block-container,
[data-testid="stMainBlockContainer"] {{
    padding: 0 !important;
    max-width: 100% !important;
    margin: 0 !important;
    min-height: 0 !important;
}}

/* L5 — section and its direct div child */
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewContainer"] > section > div {{
    padding: 0 !important;
    margin: 0 !important;
}}

/* L6 — top-level stVerticalBlock only (preserves inner widget spacing) */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{
    gap: 0 !important;
    padding: 0 !important;
    /* Flex sticky footer implementation */
    min-height: 100vh !important;
    min-height: 100dvh !important;
    display: flex;
    flex-direction: column;
}}

/* Push the Streamlit element containing the footer to the bottom of the screen */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > div:has(.custom-footer) {{
    margin-top: auto;
    width: 100%;
}}

/* L7 — stBottom: remove the dead space Streamlit renders after last element */
[data-testid="stBottom"] {{
    display: none !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}}

/* ── HIDE STREAMLIT TOP HEADER BAR ── */
[data-testid="stHeader"],
[data-testid="stDecoration"] {{
    display: none !important;
    height: 0 !important;
}}

/* ── HERO HEADER BANNER ── */
.hero-banner {{
    background-image: {bg_css};
    background-size: cover;
    background-position: center;
    padding: 70px 20px 60px;
    text-align: center;
    overflow: hidden;
    border-bottom: 5px solid #e8a020;
    /* Full-width escape from any container */
    position: relative;
    width: 110vw; /* Slightly wider to prevent scrollbar/sub-pixel gaps on the right */
    left: 50%;
    margin-left: -55vw;
    margin-right: -55vw;
    box-sizing: border-box;
}}

.hero-banner::before {{
    content: '';
    position: absolute;
    inset: 0;
    /* Dark gradient overlay — increased opacity to fade/wash out background image */
    background: linear-gradient(180deg, rgba(0,0,0,0.45) 0%, rgba(0,0,0,0.72) 100%);
    z-index: 1;
}}

/* Second overlay: semi-transparent white to make bg image feel muted/transparent */
.hero-banner::after {{
    content: '';
    position: absolute;
    inset: 0;
    background: rgba(255,255,255,0.18);
    z-index: 1;
    pointer-events: none;
}}

.hero-content {{
    position: relative;
    z-index: 2;
}}

.hero-title {{
    font-size: 1.5rem !important;
    font-weight: 900;
    color: #ffffff;
    text-shadow: 2px 3px 10px rgba(0,0,0,0.7);
    margin: 0 0 10px 0;
    line-height: 1.15;
}}

@media (max-width: 768px) {{
    .hero-title {{ font-size: 2.4rem !important; }}
}}

@media (max-width: 480px) {{
    .hero-title {{ font-size: 1.9rem !important; }}
}}

.hero-subtitle {{
    font-size: clamp(0.9rem, 3vw, 1.2rem);
    color: #ffefc0;
    text-shadow: 1px 2px 4px rgba(0,0,0,0.5);
    margin: 0;
    font-weight: 600;
}}

.hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 2px solid rgba(255,255,255,0.5);
    backdrop-filter: blur(4px);
    color: #fff;
    padding: 6px 20px;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: 14px;
    letter-spacing: 1px;
}}

/* ── CONTENT WRAPPER ── */
.content-wrapper {{
    max-width: 700px;
    margin: 0 auto;
    padding: 0 16px;
}}

/* ── CATEGORY CARD ── */
.category-section {{
    background: rgba(255,255,255,0.92);
    border-radius: 20px;
    padding: 16px;
    /* top margin creates visual gap after hero banner */
    margin-top: 24px;
    margin-bottom: 32px;
    box-shadow: 0 6px 28px rgba(0,0,0,0.18);
    border: 2px solid #e8f5e1;
    /* card height fits its content exactly */
    height: fit-content;
}}

/* Beri jarak ekstra di atas khusus untuk judul Panduan Kategori Telur */
.category-section .section-title {{
    margin-top: 12px !important;
}}

/* category image: centered, no distortion */
.category-section img {{
    display: block;
    width: 100%;
    max-width: 700px;
    height: auto;
    object-fit: contain;
    border-radius: 10px;
    margin: 0 auto;
}}

/* ── CENTERED IMAGE PREVIEW ── */
.img-center-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    margin: 8px 0 16px;
}}

.img-center-wrap img {{
    max-width: 300px;
    width: 100%;
    height: auto;
    border-radius: 14px;
    object-fit: contain;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}}

.img-center-wrap .img-caption {{
    margin-top: 8px;
    font-size: 0.9rem;
    color: #888;
    font-weight: 600;
    text-align: center;
}}

/* ── INPUT CARD (Tab Panels) ── */
.stTabs [data-baseweb="tab-panel"],
.stTabs [role="tabpanel"] {{
    background: rgba(255,255,255,0.92);
    border-radius: 20px;
    padding: 24px !important;
    margin-top: 8px !important; /* Jarak antara tombol tab dan kotak putih dikurangi */
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    border: 2px solid rgba(255,255,255,0.7);
}}

.section-title {{
    font-size: 1.1rem !important; /* Diperbesar sesuai permintaan */
    font-weight: 800 !important;
    color: #3d7a1b !important;
    margin: 0 0 16px 0 !important;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* ── GRADE BADGES ── */
.grade-row {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 6px;
}}

.grade-badge {{
    flex: 1;
    min-width: 80px;
    text-align: center;
    padding: 10px 6px;
    border-radius: 14px;
    font-weight: 800;
    font-size: 0.95rem;
}}

.grade-a {{
    background: #d4edda;
    color: #1a5c2a;
    border: 2px solid #28a745;
}}

.grade-b {{
    background: #fff3cd;
    color: #7d5a00;
    border: 2px solid #ffc107;
}}

.grade-c {{
    background: #f8d7da;
    color: #721c24;
    border: 2px solid #dc3545;
}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"],
.stTabs [role="tablist"] {{
    gap: 8px;
    background: rgba(255,255,255,0.92);
    border-radius: 16px;
    padding: 6px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.15);
}}

.stTabs [data-baseweb="tab"],
.stTabs button[role="tab"],
.stTabs [role="tab"] {{
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 700;
    font-size: 1rem;
    color: #666;
    border: none;
    background: transparent;
}}

.stTabs [aria-selected="true"] {{
    background: #e8a020 !important;
    color: white !important;
}}

/* ── HIDE DEFAULT STREAMLIT RED INDICATOR LINE ── */
.stTabs [data-baseweb="tab-highlight"],
[data-testid="stTabIndicator"] {{
    display: none !important;
}}

/* ── CARD SECTIONS inside tabs ── */
.input-card {{
    background: rgba(255,255,255,0.92);
    border-radius: 20px;
    padding: 20px;
    margin: 28px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    border: 2px solid rgba(255,255,255,0.7);
}}

/* Target Streamlit's own container element when we use st.container() with a key */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 20px !important;
    overflow: hidden;
}}

/* ── RADIO BUTTONS custom style ── */
.stRadio [data-testid="stMarkdownContainer"] p {{
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #3d5a1b !important;
}}

.stRadio label {{
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 8px 0 !important;
}}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {{
    border: 3px dashed #e8a020 !important;
    border-radius: 16px !important;
    background: #fffdf5 !important;
    padding: 10px !important;
}}

[data-testid="stFileUploader"] label {{
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #5a3e00 !important;
}}

/* ── BIG ACTION BUTTON ── */
.stButton > button {{
    background: linear-gradient(135deg, #e8a020, #c97d10) !important;
    color: white !important;
    border-radius: 16px !important;
    height: 62px !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    width: 100% !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(232,160,32,0.4) !important;
    letter-spacing: 0.5px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}}

.stButton > button:hover {{
    background: linear-gradient(135deg, #c97d10, #a86200) !important;
    box-shadow: 0 8px 26px rgba(200,120,0,0.45) !important;
    transform: translateY(-2px) !important;
}}

/* ── NUMBER INPUT ── */
.stNumberInput input {{
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    border-radius: 12px !important;
    border: 2px solid #e0d5c0 !important;
    padding: 10px !important;
}}

/* ── RESULT CARDS ── */
.result-card {{
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    margin: 16px 0 8px;
    animation: fadeInUp 0.5s ease;
}}

.result-card-a {{
    background: linear-gradient(135deg, #d4edda, #a8d5b5);
    border: 3px solid #28a745;
}}

.result-card-b {{
    background: linear-gradient(135deg, #fff3cd, #ffe69c);
    border: 3px solid #ffc107;
}}

.result-card-c {{
    background: linear-gradient(135deg, #f8d7da, #f5aaae);
    border: 3px solid #dc3545;
}}

.result-grade-label {{
    font-size: clamp(2rem, 8vw, 3.5rem);
    font-weight: 900;
    margin: 0;
    line-height: 1;
}}

.result-grade-a {{ color: #1a5c2a; }}
.result-grade-b {{ color: #7d5a00; }}
.result-grade-c {{ color: #721c24; }}

.result-desc {{
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 8px;
    color: #444;
}}

.result-accuracy {{
    font-size: 0.95rem;
    color: #666;
    margin-top: 10px;
    font-weight: 600;
}}

/* ── ACCURACY BAR ── */
.stProgress > div > div {{
    border-radius: 10px !important;
    height: 14px !important;
}}

.stProgress > div > div > div {{
    background: linear-gradient(90deg, #4caf50, #8bc34a) !important;
    border-radius: 10px !important;
}}

/* ── SPINNER ── */
[data-testid="stSpinner"] p {{
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #5a3e00 !important;
}}

/* ── STEP LABEL ── */
.step-label {{
    background: #3d7a1b;
    color: white;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 0.9rem;
    margin-right: 8px;
    flex-shrink: 0;
}}

.step-row {{
    display: flex;
    align-items: center;
    margin: 0 0 12px 0;
}}

.step-text {{
    font-size: 1.1rem;
    font-weight: 800;
    color: #3d5a1b;
    margin: 0;
}}

/* ── TRAY RESULT SUMMARY ── */
.tray-result-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    border-radius: 14px;
    margin: 8px 0;
    font-weight: 700;
    font-size: 1.1rem;
}}

/* ── FOOTER ── */
.custom-footer {{
    text-align: center;
    padding: 20px 16px 16px; /* Reduced bottom padding */
    background: #2d5a0e;
    color: #c5e8a0;
    font-size: 0.85rem;
    font-family: 'Nunito', sans-serif;
    margin-top: 32px;
    /* Full-width escape — same technique as hero banner */
    position: relative;
    width: 110vw; /* Slightly wider to prevent gaps on the right */
    left: 50%;
    margin-left: -55vw;
    margin-right: -55vw;
    box-sizing: border-box;
    /* Visual bleed to cover any small gaps at the very bottom */
    box-shadow: 0 40px 0 0 #2d5a0e;
}}

.custom-footer a {{
    color: #ffd966;
    text-decoration: none;
}}

/* ── HIDE STREAMLIT DEFAULT ELEMENTS ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── FADE IN ANIMATION ── */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.fade-in {{
    animation: fadeInUp 0.5s ease;
}}

/* ── MOBILE RESPONSIVE ── */
@media (max-width: 600px) {{
    .hero-banner {{ padding: 40px 16px 32px; }}
    .content-wrapper {{ padding: 16px 12px 0; }}
    .category-section {{ padding: 12px; margin-bottom: 16px; }}
    .grade-row {{ gap: 6px; }}
    .grade-badge {{ font-size: 0.85rem; padding: 8px 4px; min-width: 60px; }}
    .stTabs [data-baseweb="tab"], .stTabs button[role="tab"], .stTabs [role="tab"] {{ padding: 10px 12px; font-size: 0.9rem; }}
    .result-grade-label {{ font-size: 2.4rem; }}
    .step-row {{ margin-bottom: 10px; }}
    .custom-footer {{ margin-top: 24px; padding: 20px 16px 12px; }}
}}

/* ── SAFE AREA for iPhone notch / home indicator ── */
@supports (padding-bottom: env(safe-area-inset-bottom)) {{
    .custom-footer {{
        padding-bottom: calc(16px + env(safe-area-inset-bottom));
    }}
}}
</style>
""", unsafe_allow_html=True)


# ==============================
# 🧠 MODEL
# ==============================
class VisionModel(torch.nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 3),
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.model(x))


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionModel().to(device)

    model_path = Path(__file__).parent / "model-final-cadangan.pth"
    state_dict = torch.load(model_path, map_location=device)

    if 'module' in state_dict:
        state_dict = state_dict['module']

    model.load_state_dict(state_dict)
    model.eval()

    return model, device


model, device = load_model()

# ==============================
# 🔄 TRANSFORM
# ==============================
mean = torch.tensor([0.6750, 0.6106, 0.5683])
std  = torch.tensor([0.1580, 0.1903, 0.2257])

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class_labels = ["Grade A", "Grade B", "Grade C"]

# ==============================
# 🔍 FUNCTIONS
# ==============================
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)

    probs = output[0].tolist()
    idx = probs.index(max(probs))
    return class_labels[idx], max(probs) * 100


def crop_image(img, rows, cols):
    width, height = img.size
    crop_w = width // cols
    crop_h = height // rows

    crops = []
    for i in range(rows):
        for j in range(cols):
            left   = j * crop_w
            top    = i * crop_h
            right  = (j + 1) * crop_w
            bottom = (i + 1) * crop_h
            crops.append(img.crop((left, top, right, bottom)))

    return crops


def render_result_card(label, confidence):
    """Render a beautiful result card based on grade label."""
    if label == "Grade A":
        card_class = "result-card-a"
        grade_class = "result-grade-a"
        emoji = "🥇"
        desc = "Telur Kualitas Sangat Bagus! Aman untuk dijual & dikonsumsi."
    elif label == "Grade B":
        card_class = "result-card-b"
        grade_class = "result-grade-b"
        emoji = "🥈"
        desc = "Kualitas Cukup Bagus. Segera dijual, masa simpan lebih pendek."
    else:
        card_class = "result-card-c"
        grade_class = "result-grade-c"
        emoji = "🥉"
        desc = "Kualitas Kurang Bagus. Sebaiknya dipisahkan dari yang lain."

    st.markdown(f"""
    <div class="result-card {card_class}">
        <p class="result-grade-label {grade_class}">{emoji} {label}</p>
        <p class="result-desc">{desc}</p>
        <p class="result-accuracy">📊 Akurasi AI: <strong>{confidence:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence / 100)


# ==============================
# 🥚 HERO BANNER
# ==============================
st.markdown("""
<div class="hero-banner">
    <div class="hero-content">
        <p class="hero-title">Cek Kualitas Telur Ayam</p>
        <p class="hero-subtitle">Teknologi AI untuk Peternak &amp; Pedagang Pasar</p>
        <span class="hero-badge">🤖 Didukung Kecerdasan Buatan</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================
# 📌 CONTENT WRAPPER START
# ==============================
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

# ── Kategori Telur Section ──
img_path = Path(__file__).parent / "Assets" / "jenis-telur.png"
if img_path.exists():
    ref_img = Image.open(img_path)
    buf = io.BytesIO()
    ref_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f"""
    <div class="category-section">
        <p class="section-title">📌 Panduan Kategori Kualitas Telur</p>
        <img src="data:image/png;base64,{img_b64}"
             alt="Grade A, B, C telur ayam" />
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="category-section">
        <p class="section-title">📌 Panduan Kategori Kualitas Telur</p>
    </div>
    """, unsafe_allow_html=True)



# ==============================
# 🔘 TABS NAVIGATION
# ==============================
tab_single, tab_tray = st.tabs(["🥚 Telur Tunggal", "📦 Sepapan Telur (Tray)"])


# ──────────────────────────────
# 🟢 SINGLE EGG MODE
# ──────────────────────────────
with tab_single:

    st.markdown("""
    <div class="step-row" style="margin-top:0px;">
        <span class="step-label">1</span>
        <p class="step-text">Pilih cara ambil gambar telur</p>
    </div>
    """, unsafe_allow_html=True)

    input_type = st.radio(
        "Metode input:",
        ["🖼️ Pilih dari Galeri", "📷 Buka Kamera HP"],
        horizontal=True,
        label_visibility="collapsed"
    )

    image = None

    if input_type == "🖼️ Pilih dari Galeri":
        st.markdown("""
        <div class="step-row" style="margin-top:12px;">
            <span class="step-label">2</span>
            <p class="step-text">Pilih foto telur dari HP Anda</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Pastikan telur terlihat jelas & berada di tengah",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible"
        )
        if uploaded:
            image = Image.open(uploaded)

    elif input_type == "📷 Buka Kamera HP":
        st.markdown("""
        <div class="step-row" style="margin-top:12px;">
            <span class="step-label">2</span>
            <p class="step-text">Arahkan kamera ke telur, lalu jepret!</p>
        </div>
        """, unsafe_allow_html=True)
        camera_image = st.camera_input("Foto Telur Sekarang", label_visibility="collapsed")
        if camera_image:
            image = Image.open(camera_image)

    # ── Processing & Result ──
    if image is not None:
        st.markdown("""
        <div class="step-row" style="margin-top:16px;">
            <span class="step-label">3</span>
            <p class="step-text">Foto yang Anda pilih</p>
        </div>
        """, unsafe_allow_html=True)

        # Center image using pure HTML/CSS — avoids st.columns() empty-column gaps
        buf_preview = io.BytesIO()
        image.save(buf_preview, format="PNG")
        img_preview_b64 = base64.b64encode(buf_preview.getvalue()).decode()
        st.markdown(f"""
        <div class="img-center-wrap">
            <img src="data:image/png;base64,{img_preview_b64}" alt="Foto Telur Anda" />
            <span class="img-caption">Foto Telur Anda</span>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🔍 AI sedang menganalisis telur... sebentar ya!"):
            time.sleep(2.0)
            label, confidence = predict(image)

        st.markdown("""
        <div class="step-row" style="margin-top:20px;">
            <span class="step-label">4</span>
            <p class="step-text">Hasil Pengecekan AI</p>
        </div>
        """, unsafe_allow_html=True)

        render_result_card(label, confidence)


# ──────────────────────────────
# 🟡 TRAY MODE
# ──────────────────────────────
with tab_tray:

    # ── Upload (no card box — plain step row + uploader) ──
    st.markdown("""
    <div class="step-row" style="margin-top:0px;">
        <span class="step-label">1</span>
        <p class="step-text">Upload foto sepapan telur (tray)</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_tray = st.file_uploader(
        "Pastikan semua telur terlihat jelas dari atas",
        type=["jpg", "jpeg", "png"],
        key="tray_uploader"
    )

    if uploaded_tray:
        image_tray = Image.open(uploaded_tray)

        # ── Card 2: Preview tray image (centered) ──
        st.markdown("""
        <div class="step-row" style="margin-top:16px;">
            <span class="step-label">2</span>
            <p class="step-text">Foto tray Anda</p>
        </div>
        """, unsafe_allow_html=True)
        buf_tray = io.BytesIO()
        image_tray.save(buf_tray, format="PNG")
        img_tray_b64 = base64.b64encode(buf_tray.getvalue()).decode()
        st.markdown(f"""
        <div class="img-center-wrap" style="margin-bottom:8px;">
            <img src="data:image/png;base64,{img_tray_b64}"
                 alt="Foto Tray Anda"
                 style="max-width:480px; border-radius:14px;" />
            <span class="img-caption">Foto Tray Anda</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Card 3: Grid config ──
        st.markdown("""
        <div class="step-row" style="margin-top:16px;">
            <span class="step-label">3</span>
            <p class="step-text">Berapa susunan telur di tray?</p>
        </div>
        """, unsafe_allow_html=True)
        col_row, col_col = st.columns(2)
        with col_row:
            row = st.number_input("Jumlah Baris ↕", min_value=1, max_value=20, step=1, value=3)
        with col_col:
            col = st.number_input("Jumlah Kolom ↔", min_value=1, max_value=20, step=1, value=6)

        if st.button("🔍 Mulai Cek Semua Telur!", key="btn_analyze_tray"):
            crops = crop_image(image_tray, row, col)

            progress_bar = st.progress(0)
            results = []

            fig, axes = plt.subplots(row, col, figsize=(max(10, col * 2), max(6, row * 2)))
            fig.patch.set_facecolor('#f5f0e8')

            for i in range(row * col):
                lbl, conf = predict(crops[i])
                results.append(lbl)

                r_idx = i // col
                c_idx = i % col

                color_map = {"Grade A": "#28a745", "Grade B": "#ffc107", "Grade C": "#dc3545"}
                title_color = color_map.get(lbl, "#333")

                if row > 1:
                    ax = axes[r_idx, c_idx]
                else:
                    ax = axes[c_idx]

                ax.imshow(crops[i])
                ax.set_title(lbl, fontsize=9, fontweight='bold', color=title_color, pad=3)
                ax.axis("off")

                progress_bar.progress((i + 1) / (row * col))

            plt.tight_layout(pad=0.5)
            st.pyplot(fig)

            # ── Tray Summary ──
            counter = Counter(results)
            total = sum(counter.values())

            st.markdown("### 📊 Rangkuman Hasil:")

            color_info = {
                "Grade A": ("result-card-a", "🥇"),
                "Grade B": ("result-card-b", "🥈"),
                "Grade C": ("result-card-c", "🥉"),
            }

            for grade in ["Grade A", "Grade B", "Grade C"]:
                if grade in counter:
                    cnt = counter[grade]
                    pct = cnt / total * 100
                    c_class, emoji = color_info[grade]
                    st.markdown(f"""
                    <div class="tray-result-item result-card {c_class}">
                        <span>{emoji} {grade}</span>
                        <span>{cnt} telur &nbsp;({pct:.0f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align:center; font-size:0.95rem; color:#666; margin-top:10px; font-weight:600;">
                Total: <strong>{total} telur</strong> diperiksa
            </div>
            """, unsafe_allow_html=True)


# end content-wrapper (no closing </div> needed — content-wrapper is pure CSS class on a markdown div that Streamlit doesn't nest widgets into)

# ==============================
# 🌿 FOOTER
# ==============================
st.markdown("""
<div class="custom-footer">
    🥚 <strong>Sistem Klasifikasi Kualitas Telur Ayam</strong> &copy; 2026<br>
    <span style="font-size:0.78rem; opacity:0.8;">
        Referensi: <a href="https://github.com/putrinahampun/final-project-scAI5" target="_blank">github.com/putrinahampun/final-project-scAI5</a>
    </span>
</div>
""", unsafe_allow_html=True)