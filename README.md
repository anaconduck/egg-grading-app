# 🥚 Sistem Klasifikasi Kualitas Telur Ayam (Egg Grading App)

Aplikasi pintar berbasis **Kecerdasan Buatan (AI)** yang dirancang untuk mengklasifikasikan kualitas telur ayam ke dalam **Grade A, Grade B, atau Grade C** secara otomatis melalui unggahan gambar. Aplikasi ini sangat berguna bagi peternak, pedagang pasar, maupun konsumen untuk memastikan kualitas telur secara objektif dan cepat.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://egg-grading-app29.streamlit.app/)

## 🚀 Live Demo
Cobalah aplikasi ini secara langsung tanpa perlu instalasi!
🔗 **[Buka Aplikasi di Streamlit Cloud](https://egg-grading-app29.streamlit.app/)**

---

## 🛠️ Cara Menjalankan Secara Lokal (Local Installation)

Jika Anda ingin menjalankan aplikasi ini di komputer Anda sendiri, ikuti langkah-langkah berikut:

### 1. Clone Repositori
Clone (unduh) repositori ini ke komputer lokal Anda:
```bash
git clone <link-repositori-anda>
cd <nama-folder-repositori>
```

### 2. Siapkan Virtual Environment (Opsional namun Disarankan)
Sangat disarankan untuk membuat *virtual environment* Python agar dependensi tidak berantakan:
```bash
python -m venv .venv

# Untuk Windows:
.venv\Scripts\activate

# Untuk Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependensi
Install seluruh *library* yang dibutuhkan (terutama PyTorch, torchvision, dan Streamlit) melalui perintah:
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
Setelah instalasi selesai, jalankan aplikasi Streamlit dengan perintah:
```bash
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka pada browser Anda di alamat `http://localhost:8501`.

---

## ⚠️ Hak Cipta, Atribusi, & Referensi Dataset

**Penafian (Disclaimer):**
Proyek ini merupakan bentuk pengembangan modifikasi (*UI/UX, arsitektur tata letak, dan optimasi web*) untuk keperluan studi/pengembangan aplikasi antarmuka. 

Dataset model *machine learning* dan referensi konseptual utama dari AI pengolahan gambar yang digunakan di dalam aplikasi ini **Bukanlah milik maupun buatan saya pribadi**. 

Saya dengan tegas memberikan kredit penuh dan hak cipta dataset kepada pemilik aslinya. Karya aplikasi ini dibangun dan mengambil referensi dari repositori berikut:
👉 **[github.com/putrinahampun/final-project-scAI5](https://github.com/putrinahampun/final-project-scAI5)**

Penggunaan dataset/sumber dari repositori di atas dilakukan semata-mata sebagai referensi pendidikan/pembelajaran. Tidak ada klaim kepemilikan pribadi atas dataset tersebut.
