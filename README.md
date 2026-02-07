# 🔥 Yangın Tespit Sistemi (Fire Detection System)

YOLOv8 tabanlı gerçek zamanlı yangın algılama sistemi - Derin öğrenme ile güvenlik tehdidi tespiti.

---

## 🎯 Özellikler

- ✅ **YOLOv8s** derin öğrenme modeli
- ✅ **Web Arayüzü** - Modern, kullanıcı dostu
- ✅ **Görüntü Yükleme** - Sürükle-bırak desteği
- ✅ **Canlı Kamera** - Gerçek zamanlı tespit
- ✅ **Yüksek Doğruluk** - mAP50: 0.785, Precision: 0.793

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Sanal ortam oluştur
python -m venv .venv
.venv\Scripts\activate

# Gerekli paketleri yükle
pip install -r requirements.txt
pip install -r requirements_web.txt
```

### 2. Web Uygulamasını Başlat

```bash
python web_app.py
```

Tarayıcınızda açın: **http://127.0.0.1:5000**

### 3. Kullanım

- **Sol Panel:** Görüntü yükle (JPG, PNG, BMP)
- **Sağ Panel:** Canlı kamera ile test et

---

## 📁 Proje Yapısı

```
Bitirme/
├── web_app.py              # Ana web uygulaması
├── test_images.py          # Toplu test scripti
├── config.yaml             # Yapılandırma dosyası
│
├── src/
│   ├── fire_detection/     # Yangın tespit modülü
│   │   ├── fire_detector.py
│   │   ├── train.py
│   │   └── demo.py
│   └── utils/              # Yardımcı araçlar
│       ├── config_loader.py
│       ├── check_data_leakage.py
│       └── prepare_dataset.py
│
├── data/                   # Veri seti
│   └── fire_detection/
│       ├── train/
│       ├── valid/
│       └── test/
│
├── results/                # Eğitim sonuçları
│   └── fire_detection/
│       └── yolov8_fire8/
│           └── weights/
│               └── best.pt
│
└── templates/              # Web arayüzü
    └── index.html
```

---

## 💻 Kullanım Örnekleri

### Web Uygulaması (Önerilen)
```bash
python web_app.py
# Tarayıcıda aç: http://127.0.0.1:5000
```

### Komut Satırı - Tek Resim
```bash
python src/fire_detection/demo.py --model "results/fire_detection/yolov8_fire8/weights/best.pt" --source "resim.jpg" --output "sonuc.jpg"
```

### Komut Satırı - Canlı Kamera
```bash
python src/fire_detection/demo.py --model "results/fire_detection/yolov8_fire8/weights/best.pt" --source 0
```

### Toplu Test
```bash
python test_images.py --model "results/fire_detection/yolov8_fire8/weights/best.pt" --images "test_klasoru/" --output "sonuclar/"
```

---

## 🎓 Model Eğitimi

### 1. Dataset Hazırlama
```bash
python src/utils/prepare_dataset.py --source "ham_veri/" --output "data/fire_detection"
```

### 2. Data Leakage Kontrolü
```bash
python src/utils/check_data_leakage.py --dataset "data/fire_detection"
```

### 3. Model Eğitimi
```bash
python src/fire_detection/train.py --config config.yaml
```

---

## 📊 Performans Metrikleri

| Metrik | Değer |
|--------|-------|
| **Model** | YOLOv8s |
| **mAP50** | 0.785 |
| **mAP50-95** | 0.389 |
| **Precision** | 0.793 |
| **Recall** | 0.726 |
| **FPS (GPU)** | ~25-30 |
| **Inference Time** | ~30ms |

---

## ⚙️ Yapılandırma

`config.yaml` dosyasını düzenleyerek parametreleri ayarlayın:

```yaml
fire_detection:
  model_name: "yolov8s"
  img_size: 640
  batch_size: 16
  epochs: 150
  confidence_threshold: 0.35      # Training
  inference_threshold: 0.5        # Inference
  iou_threshold: 0.45
```

---

## 📚 Dokümantasyon

- **[WEB_APP_KULLANIM.md](WEB_APP_KULLANIM.md)** - Web uygulaması detaylı kullanım
- **[INSTALLATION.md](INSTALLATION.md)** - Kurulum rehberi
- **[QUICKSTART.md](QUICKSTART.md)** - Hızlı başlangıç kılavuzu
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Proje genel bakış

---

## 🔧 Sorun Giderme

### Model Bulunamıyor
```bash
# Doğru yolu kullanın:
python web_app.py --model "results/fire_detection/yolov8_fire8/weights/best.pt"
```

### Kamera Açılmıyor
- Başka uygulama kamerayı kullanıyor olabilir
- Tarayıcıya kamera izni verin
- Kamera device ID'sini kontrol edin (0, 1, 2...)

### Yavaş Çalışıyor
- GPU kullanıldığından emin olun
- CUDA kurulu mu kontrol edin: `torch.cuda.is_available()`
- Görüntü boyutunu küçültün

---

## 🛠️ Gereksinimler

- Python 3.8+
- CUDA 11.8+ (GPU için)
- 8GB+ RAM
- Webcam (canlı tespit için)

---

## 📦 Bağımlılıklar

```bash
# Core
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0

# Web App
flask>=2.3.0
pillow>=10.0.0

# Utils
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 📞 Destek

Sorun yaşarsanız:
1. Dokümantasyonu kontrol edin
2. Console çıktısını inceleyin
3. Model path'ini doğrulayın

---

**⚡ Hızlı Başlat:**
```bash
python web_app.py
```
**Tarayıcıda açın:** http://127.0.0.1:5000

---

© 2026 Yangın Tespit Sistemi - Bitirme Projesi
