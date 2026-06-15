# Real-Time Driver DrowsinessDetection System

Sistem pemantauan keselamatan aktif pengemudi (*driver safety monitoring*) berbasis *Computer Vision* secara *real-time*. Sistem dirancang untuk menekan angka kecelakaan lalu lintas akibat kelelahan pengemudi dengan mendeteksi tanda-tanda krusial seperti kantuk, aktivitas menguap, hingga hilangnya konsentrasi akibat berpaling dari arah jalan.

## 🎯 Fitur & Logika Pendeteksian

Aplikasi ini melacak 68 titik koordinat wajah (*facial landmarks*) menggunakan pustaka Dlib untuk mengekstrak dan mengalkulasi matriks geometris wajah berikut:

* **Deteksi Tingkat Kantuk (Eye Aspect Ratio - EAR):** Jika nilai EAR berada di bawah `0.25` selama lebih dari `20 frame` berturut-turut, sistem memunculkan peringatan visual dan mengaktifkan alarm audio.
* **Deteksi Aktivitas Menguap (Mouth Aspect Ratio - MAR):** Jika nilai MAR melebihi `0.65` selama lebih dari `35 frame` berturut-turut, sistem mendeteksi pengemudi sedang menguap dan memicu alarm.
* **Deteksi Fokus Berkendara (Head Pose Estimation):** Mengklasifikasikan arah hadap wajah. Jika terdeteksi selain menghadap "Depan" selama lebih dari `40 frame`, sistem akan memperingatkan pengemudi untuk kembali fokus.

## 🛠️ Arsitektur & Teknologi

* **Language Support:** Python 3.9 (Direkomendasikan untuk stabilitas kompilasi dlib)
* **Computer Vision Framework:** OpenCV
* **Landmark Extraction:** Dlib
* **Mathematical Calculations:** SciPy, NumPy
* **Audio & Alert System:** Pygame Mixer

## ⚙️ Panduan Instalasi Lokal

Sistem ini bergantung pada **Python 3.9** untuk menjaga stabilitas kompilasi pustaka `dlib`. Sangat disarankan untuk menggunakan *virtual environment*.

**1. Clone Repositori**
```bash
git clone [https://github.com/USERNAME/driver-drowsiness-detection.git](https://github.com/USERNAME/driver-drowsiness-detection.git)
cd driver-drowsiness-detection
```

**2. Unduh Model Dlib (Wajib)**
- Proyek ini membutuhkan file bobot model dari Dlib yang berukuran cukup besar.
- Unduh file shape_predictor_68_face_landmarks.dat.bz2 dari tautan resmi [Dlib ini](https://drive.google.com/file/d/1giSlhSaeBGNhsMXK1sScfCqt3YhKPg0f/view?usp=drive_link).
- Ekstrak file tersebut hingga Anda mendapatkan file shape_predictor_68_face_landmarks.dat.
- Letakkan file .dat tersebut tepat di dalam folder utama proyek ini.\

**3. Siapkan Virtual Environment & Instal Dependensi**
```bash
py -3.9 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
**4. Jalankan Aplikasi**
Pastikan Anda telah memiliki file music.wav di dalam direktori proyek sebagai alarm audio.
```bash
python "Driver Drowsiness Detection.py"
```
