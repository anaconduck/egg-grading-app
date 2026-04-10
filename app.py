import io
import time
import torch
import streamlit as st
import torchvision
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from PIL import Image
from collections import Counter
from torchvision import transforms
from matplotlib import pyplot as plt

# ==============================
# 🎨 CUSTOM UI STYLE
# ==============================
st.set_page_config(page_title="Egg Grading AI", layout="wide")

st.markdown("""
<style>
/* Top & bottom spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 0rem;
}

/* Full height layout */
html, body, [data-testid="stAppViewContainer"] {
    height: 100%;
}

/* Flex layout */
[data-testid="stAppViewContainer"] > .main {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

/* Push footer ke bawah */
footer {
    margin-top: auto;
}

/* Button style */
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-weight: bold;
}

/* Upload box */
.stFileUploader {
    border: 2px dashed #444;
    padding: 20px;
    border-radius: 10px;
}

/* Text color */
h1, h2, h3, h4 {
    color: white;
}

/* Background */
body {
    background-color: #0E1117;
}
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
std = torch.tensor([0.1580, 0.1903, 0.2257])

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
            left = j * crop_w
            top = i * crop_h
            right = (j + 1) * crop_w
            bottom = (i + 1) * crop_h
            crops.append(img.crop((left, top, right, bottom)))

    return crops


# ==============================
# ⚙️ SIDEBAR
# ==============================
st.sidebar.title("⚙️ Pengaturan")

mode = st.sidebar.radio(
    "Pilih Mode Analisis:",
    ["Telur Tunggal", "Banyak Telur (Tray)"]
)

# ==============================
# 🥚 MAIN TITLE
# ==============================
st.title("🥚 Sistem Klasifikasi Kualitas Telur")
st.markdown("Sistem Berbasis AI Untuk Klasifikasi Kualitas Telur Ayam Berdasarkan Warna.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Kategori Klasifikasi:")

st.image("jenis-telur.png",width=515)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================
# 🟢 SINGLE MODE (UPLOAD + CAMERA)
# ==============================
if mode == "Telur Tunggal":

    st.subheader("📤 Masukkan Gambar Telur")

    input_type = st.radio(
        "Pilih Metode Input:",
        ["Unggah Gambar", "Gunakan Kamera"],
        horizontal=True
    )

    image = None

    # ==============================
    # 📁 UPLOAD MODE
    # ==============================
    if input_type == "Unggah Gambar":
        uploaded = st.file_uploader(
            "Pastikan telur terlihat jelas dan berada di tengah gambar.",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            image = Image.open(uploaded)

    # ==============================
    # 📸 CAMERA MODE
    # ==============================
    elif input_type == "Gunakan Kamera":
        camera_image = st.camera_input("Ambil Foto")

        if camera_image:
            image = Image.open(camera_image)

    # ==============================
    # 🔍 PROCESSING
    # ==============================
    if image is not None:

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, caption="Unggah Gambar", width=250)

        with st.spinner("🧠 Memuat Analisis..."):
            time.sleep(2.5)
            label, confidence = predict(image)

        st.markdown("## 🧠 Hasil Klasifikasi")

        # Badge warna
        if label == "Grade A":
            st.success(f"🥇 {label}")
            st.info("""✔ Kualitas tinggi, Cocok untuk konsumsi dan penjualan""")

        elif label == "Grade B":
            st.warning(f"🥈 {label}")
            st.warning("""⚠ Kualitas sedang, Masa simpan terbatas""")

        else:
            st.error(f"🥉 {label}")
            st.error("""❌ Kualitas rendah, Tidak direkomendasikan""")

        st.progress(confidence / 100)
        st.write(f"Tingkat Akurasi: {confidence:.2f}%")


# ==============================
# 🟡 MULTIPLE MODE
# ==============================
elif mode == "Banyak Telur (Tray)":
    st.subheader("📊 Analisis Tray Telur")

    uploaded = st.file_uploader("Pastikan telur terlihat jelas dan berada di tengah gambar.", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, caption="Uploaded Tray", width=300)

        st.markdown("### Tray Configuration")
        row = st.number_input("Jumlah Baris", min_value=1, step=1)
        col = st.number_input("Jumlah Kolom", min_value=1, step=1)

        if st.button("🔍 Analyze Tray"):

            if row == 0 or col == 0:
                st.warning("Please input valid values")
            else:
                crops = crop_image(image, row, col)

                progress = st.progress(0)
                results = []

                fig, axes = plt.subplots(row, col, figsize=(10,10))

                for i in range(row * col):
                    label, conf = predict(crops[i])
                    results.append(label)

                    r = i // col
                    c = i % col

                    if row > 1:
                        axes[r, c].imshow(crops[i])
                        axes[r, c].set_title(label)
                        axes[r, c].axis("off")
                    else:
                        axes[c].imshow(crops[i])
                        axes[c].set_title(label)
                        axes[c].axis("off")

                    progress.progress((i+1)/(row*col))

                st.pyplot(fig)

                counter = Counter(results)

                st.markdown("## 📊 Hasil Analisis")
                for k, v in counter.items():
                    st.write(f"{k}: {v} telur")

st.divider()

st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px; padding-bottom: 10px;">
© 2026 Sistem Klasifikasi Kualitas Telur <br>
Referensi: <i>Egg Grading Quality Classification Based on its External Property (Shell Color) Using ResNet-18</i> <br>
<a href="https://github.com/putrinahampun/final-project-scAI5" target="_blank" style="color: #888;">
github.com/putrinahampun/final-project-scAI5
</a>
</div>
""", unsafe_allow_html=True)
            