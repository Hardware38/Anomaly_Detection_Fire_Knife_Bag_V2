import os

# Resimlerin ve etiketlerin yolu
resim_klasoru = os.path.join("data", "fire_detection", "train", "images")
etiket_klasoru = os.path.join("data", "fire_detection", "train", "labels")

print("İşlem başlıyor...")

# Klasör kontrolü
if not os.path.exists(resim_klasoru):
    print("HATA: Resim klasörü bulunamadı!")
else:
    sayac = 0
    # Klasördeki tüm dosyaları gez
    for dosya in os.listdir(resim_klasoru):
        if dosya.lower().endswith((".jpg", ".png", ".jpeg")):
            dosya_adi = os.path.splitext(dosya)[0]
            txt_dosyasi = os.path.join(etiket_klasoru, dosya_adi + ".txt")
            
            # Eğer txt dosyası YOKSA, boş bir tane oluştur
            if not os.path.exists(txt_dosyasi):
                with open(txt_dosyasi, "w") as f:
                    pass # Boş dosya
                print(f"Boş etiket oluşturuldu: {dosya_adi}.txt")
                sayac += 1
    
    print(f"Bitti! Toplam {sayac} yeni boş etiket oluşturuldu.")