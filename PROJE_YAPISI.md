# 📂 Temiz Proje Yapısı

## 🎯 Ana Dosyalar

### **Çalıştırılabilir Programlar:**
```
web_app.py              # 🌐 WEB UYGULAMASI (ANA PROGRAM)
test_images.py          # 📸 Toplu resim test aracı
```

### **Yapılandırma:**
```
config.yaml             # ⚙️ Ana yapılandırma dosyası
requirements.txt        # 📦 Python bağımlılıkları
requirements_web.txt    # 📦 Web uygulaması bağımlılıkları
```

### **Dokümantasyon:**
```
README.md               # 📖 Ana dokümantasyon
WEB_APP_KULLANIM.md     # 🌐 Web uygulaması kullanım kılavuzu
INSTALLATION.md         # 💿 Kurulum rehberi
QUICKSTART.md           # ⚡ Hızlı başlangıç
PROJECT_OVERVIEW.md     # 📊 Proje genel bakış
```

---

## 📁 Klasör Yapısı

```
Bitirme/
│
├── 🌐 WEB UYGULAMASI
│   ├── web_app.py                 # Flask backend
│   └── templates/
│       └── index.html             # Modern web arayüzü
│
├── 🔥 YANGIN TESPİT MODÜLLERİ
│   └── src/
│       ├── fire_detection/
│       │   ├── fire_detector.py   # Ana tespit sınıfı
│       │   ├── train.py           # Model eğitimi
│       │   └── demo.py            # Komut satırı demo
│       │
│       └── utils/
│           ├── config_loader.py   # Yapılandırma yükleyici
│           ├── check_data_leakage.py  # Veri kontrolü
│           └── prepare_dataset.py     # Dataset hazırlama
│
├── 📊 VERİ VE MODELLER
│   ├── data/
│   │   └── fire_detection/
│   │       ├── train/             # Eğitim seti
│   │       ├── valid/             # Validasyon seti
│   │       └── test/              # Test seti
│   │
│   └── results/
│       └── fire_detection/
│           └── yolov8_fire8/
│               └── weights/
│                   └── best.pt    # Eğitilmiş model
│
└── 📚 DOKÜMANTASYON
    ├── README.md
    ├── WEB_APP_KULLANIM.md
    ├── INSTALLATION.md
    ├── QUICKSTART.md
    └── PROJECT_OVERVIEW.md
```

---

## 🎮 Kullanım Senaryoları

### **Senaryo 1: Web Üzerinden Test (EN KOLAY)**
```bash
# 1. Uygulamayı başlat
python web_app.py

# 2. Tarayıcıda aç
http://127.0.0.1:5000

# 3. Resim yükle veya kamera kullan
```

### **Senaryo 2: Komut Satırından Tek Resim**
```bash
python src/fire_detection/demo.py \
  --model "results/fire_detection/yolov8_fire8/weights/best.pt" \
  --source "resim.jpg"
```

### **Senaryo 3: Toplu Test**
```bash
python test_images.py \
  --model "results/fire_detection/yolov8_fire8/weights/best.pt" \
  --images "klasor/" \
  --output "sonuclar/"
```

### **Senaryo 4: Yeni Model Eğitimi**
```bash
# 1. Dataset kontrol
python src/utils/check_data_leakage.py

# 2. Model eğit
python src/fire_detection/train.py --config config.yaml
```

---

## 🗑️ Silinen Dosyalar

Aşağıdaki dosyalar **TEMİZLENDİ** (artık kullanılmıyor):

### Silinen Python Scriptleri:
- ❌ `main.py` - Eski menü sistemi (web app ile değiştirildi)
- ❌ `clear_light_labels.py` - Tek kullanımlık script
- ❌ `collect_hard_negatives.py` - Tek kullanımlık script
- ❌ `merge_datasets.py` - Tek kullanımlık script
- ❌ `visualize_training.py` - Gereksiz (results klasöründe zaten var)

### Silinen Dokümantasyon:
- ❌ `ALARM_VE_RECALL_DUZELTMELERI.md` - Geliştirme notu
- ❌ `DETECTION_RATE_IYILESTIRMELERI.md` - Geliştirme notu
- ❌ `improve_dataset.md` - Geliştirme notu
- ❌ `NEXT_STEPS.md` - Geliştirme notu

**Sonuç:** Proje daha temiz ve anlaşılır! ✨

---

## 🎯 Ne Kullanmalısınız?

### **Normal Kullanım İçin:**
→ **`web_app.py`** - Web arayüzü ile kolay kullanım

### **Geliştirme/Test İçin:**
→ **`test_images.py`** - Toplu test
→ **`src/fire_detection/demo.py`** - Komut satırı test

### **Model Eğitimi İçin:**
→ **`src/fire_detection/train.py`** - Yeni model eğit
→ **`src/utils/check_data_leakage.py`** - Dataset kontrol

---

## ⚡ Hızlı Komutlar

```bash
# Web uygulamasını başlat
python web_app.py

# Tek resim test et
python src/fire_detection/demo.py --model "results/fire_detection/yolov8_fire8/weights/best.pt" --source "resim.jpg"

# Canlı kamera
python src/fire_detection/demo.py --model "results/fire_detection/yolov8_fire8/weights/best.pt" --source 0

# Dataset kontrol
python src/utils/check_data_leakage.py --dataset "data/fire_detection"

# Model eğit
python src/fire_detection/train.py --config config.yaml
```

---

© 2026 - Temiz ve Organize Proje Yapısı
