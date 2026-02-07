# 🔥 Yangın Tespit Web Uygulaması - Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### 1. Web Uygulamasını Başlatın
```powershell
python web_app.py
```

**Alternatif (özel model ile):**
```powershell
python web_app.py --model "results/fire_detection/yolov8_fire8/weights/best.pt"
```

### 2. Tarayıcınızı Açın
```
http://127.0.0.1:5000
```

### 3. Kullanım

#### 📤 **Görüntü Yükleme Modu:**
1. Sol taraftaki "Görüntü Yükle" bölümüne tıklayın
2. Bilgisayarınızdan bir resim seçin VEYA sürükle-bırak yapın
3. Sistem otomatik olarak analiz eder
4. Sonuçlar gösterilir:
   - ✅ Yangın tespit edildi/edilmedi
   - 📊 Güven skoru (%)
   - 🖼️ İşaretlenmiş görüntü

#### 📹 **Canlı Kamera Modu:**
1. Sağ taraftaki "Canlı Kamera" bölümünde "Kamerayı Başlat" butonuna tıklayın
2. Tarayıcı kamera izni isteyecek - izin verin
3. Gerçek zamanlı yangın tespiti başlar
4. Durdurmak için "Kamerayı Durdur" butonuna tıklayın

---

## ⚙️ Gelişmiş Ayarlar

### Farklı Port Kullanma
```powershell
python web_app.py --port 8080
```

### Tüm Cihazlardan Erişim (LAN)
```powershell
python web_app.py --host 0.0.0.0
```
Sonra diğer cihazlardan şu adrese girin:
```
http://[BILGISAYARIN_IP_ADRESI]:5000
```

### Kendi Modelinizi Kullanma
```powershell
python web_app.py --model "path/to/your/model.pt"
```

---

## 📱 Özellikler

### ✅ Web Arayüzü
- Modern, responsive tasarım
- Masaüstü ve mobil uyumlu
- Sürükle-bırak dosya yükleme
- Gerçek zamanlı sonuçlar

### ✅ Görüntü Yükleme
- JPG, PNG, BMP formatları desteklenir
- Maksimum 16MB dosya boyutu
- Otomatik analiz
- Detaylı sonuç gösterimi

### ✅ Canlı Kamera
- Gerçek zamanlı tespit
- ~30 FPS performans
- Güven skoru gösterimi
- Yangın durumunda kırmızı uyarı

### ✅ Sonuçlar
- Yangın tespit durumu (EVET/HAYIR)
- Güven skoru (0-100%)
- Tespit edilen yangın sayısı
- İşaretlenmiş görüntü

---

## 🛠️ Sorun Giderme

### "Model not found" Hatası
```powershell
# Model yolunu kontrol edin:
python web_app.py --model "results/fire_detection/yolov8_fire8/weights/best.pt"
```

### Kamera Açılmıyor
- Başka bir uygulama kamerayı kullanıyor olabilir
- Kamera izinlerini kontrol edin
- Tarayıcıya kamera erişimi verin

### Port Zaten Kullanımda
```powershell
# Farklı port kullanın:
python web_app.py --port 5001
```

### Yavaş Çalışıyor
- GPU kullanıldığından emin olun
- TTA kapalı (varsayılan)
- Görüntü çözünürlüğünü düşürün

---

## 📊 Performans

| Özellik | Değer |
|---------|-------|
| Görüntü İşleme | ~0.3 saniye |
| Canlı Kamera FPS | ~25-30 FPS |
| Model mAP50 | 0.785 |
| Precision | 0.793 |
| Recall | 0.726 |

---

## 🔒 Güvenlik Notları

- Uygulamayi sadece güvendiğiniz ağlarda çalıştırın
- `--host 0.0.0.0` kullanırken dikkatli olun
- Üretim için ek güvenlik önlemleri ekleyin
- HTTPS kullanımı önerilir

---

## 💡 İpuçları

1. **En İyi Sonuçlar İçin:**
   - Iyi aydınlatılmış görüntüler kullanın
   - Kamera odaklı ve sabit olsun
   - Net, bulanık olmayan görüntüler tercih edin

2. **Performans İçin:**
   - GPU kullanımını etkinleştirin
   - Gereksiz arka plan programlarını kapatın
   - Görüntü boyutunu makul tutun

3. **Test İçin:**
   - Test klasöründeki örnek görüntüleri kullanın
   - Farklı yangın senaryolarını deneyin
   - False positive kontrolü yapın

---

## 🎯 Örnek Kullanımlar

### Senaryo 1: Ofis Güvenliği
```powershell
# Sabit IP ile LAN üzerinden erişim
python web_app.py --host 0.0.0.0 --port 5000
```

### Senaryo 2: Test ve Geliştirme
```powershell
# Lokal test
python web_app.py
```

### Senaryo 3: Demo Sunumu
```powershell
# Büyük ekranda gösterim
python web_app.py --port 8080
```

---

## 📞 Destek

Sorun yaşarsanız:
1. Konsol çıktısını kontrol edin
2. Model dosyasının var olduğundan emin olun
3. Gerekli paketlerin yüklü olduğunu doğrulayın
4. Kamera ve dosya izinlerini kontrol edin

---

**Tüm hakları saklıdır © 2026 - Yangın Tespit Sistemi**
