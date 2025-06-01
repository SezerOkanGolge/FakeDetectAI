# 🧠 FAKE DETECT AI

**Gerçek mi, sahte mi?**  
Görüntü ve metin tabanlı içeriklerin doğruluğunu yapay zeka ile analiz eden web uygulaması.

![fake-detect-banner](https://via.placeholder.com/800x250?text=Fake+Detect+AI)

---

## 🚀 Özellikler

- 🔍 Görsel içerikler için deepfake ve forensik sahtecilik tespiti (9 model birden çalışır)
- ✍️ Metin içerikler için sahte haber tahmini (NLP + Wikipedia + Google destekli)
- 🧪 Ayrıntılı model analizi ve güven oranı
- 🌐 Modern kullanıcı arayüzü
- 🧰 Flask + PyTorch + Scikit-learn + Transformers destekli

---

## 📁 Proje Yapısı

```
FakeDetectAI/
├── backend_img/           → Görsel analiz için Flask API (app.py)
│   └── model/             → Eğitilmiş görsel modeller (.pth / .pkl)
├── backend_text/          → Metin analiz için Flask API (predict.py)
│   └── distilBert_model/  → Eğitilmiş DistilBERT metin modeli
├── pages/                 → Ortak HTML, CSS, JS dosyaları
│   ├── imgD.html          → Görsel yükleme arayüzü
│   └── textD.html         → Metin analizi arayüzü
├── assets/                → Logo, stil, ikon vb.
├── requirements_img.txt   → Görsel analiz ortamı paketleri
├── requirements_text.txt  → Metin analiz ortamı paketleri
└── README.md              → Bu dosya
```

---

## ⚙️ Kurulum

### ️1. Ortamları oluştur:

```bash
# Görsel için
conda create -n venv_img python=3.9
conda activate venv_img
pip install -r requirements_img.txt

# Metin için
conda create -n venv_text python=3.9
conda activate venv_text
pip install -r requirements_text.txt
```

### ️2. Flask sunucularını başlat:

```bash
# Görsel analiz servisi
cd backend_img
python app.py

# Metin analiz servisi
cd backend_text
python predict.py
```

### ️3. HTML arayüzünü çalıştır:

```bash
cd pages
python -m http.server 8080
```

Tarayıcıda aç:
```
http://localhost:8080/imgD.html
http://localhost:8080/textD.html
```

---

## 💡 Örnek Çıktılar

**Görsel analiz sonucu:**
```
📷 Analiz edilen görselde 6/9 model sahtecilik bulgusu tespit etti.
Güven Oranı: %68.3 → Sonuç: SAHTE
```

**Metin analiz sonucu:**
```
🧠 Girdiğiniz metin istatistiksel olarak SAHTE olarak sınıflandırıldı.
Güven Oranı: %74.5
Kaynak: Wikipedia, Google Search
```

---

## 📦 Kullanılan Teknolojiler

- Python 3.9
- Flask / Flask-CORS
- PyTorch
- Scikit-learn
- HuggingFace Transformers
- BeautifulSoup, Wikipedia API, Googlesearch

---

## 🧠 Eğitilmiş Modeller

### Görsel Analiz Modelleri:
- **Forensik İşleme Modelleri:**
  - `Copy_Move_FIM_fixed.pkl` - Copy-Move saldırı tespiti
  - `Inpainting_FIM_fixed.pkl` - Inpainting manipülasyon tespiti
  - `Splicing_FIM_fixed.pkl` - Splicing sahtecilik tespiti

- **Deepfake Tespit Modelleri:**
  - `Deepfake_Model_1.pth` - Deepfake tespit modeli v1
  - `Deepfake_Model_2.pth` - Deepfake tespit modeli v2
  - `model_2.pth` - Genel deepfake modeli

- **Sahte Yüz Tespit Modelleri:**
  - `Fake_Face_Detection_Model.pth` - Ana sahte yüz tespit modeli
  - `Fake_Face_Detection_Model_1.pth` - Sahte yüz modeli v1
  - `Fake_Face_Detection_Model_2.pth` - Sahte yüz modeli v2

### Metin Analiz Modeli:
- **DistilBERT** tabanlı metin sınıflandırıcı (bert-base-uncased)

---

## ⚠️ Uyarı

> Bu sistem %100 doğruluk garantisi vermez. Yalnızca istatistiksel ve içerik temelli bir tahmin sunar. Nihai değerlendirme için insan kontrolü tavsiye edilir.

---

## 👥 Geliştirici Ekibi

Bu proje **Uludağ Üniversitesi Bilgisayar Mühendisliği** bölümü öğrencileri tarafından geliştirilmiştir.

### 🎓 Ekip Üyeleri:
- **Sezer Okan Gölge** - Kullanıcı Arayüzü Tasarımı (UI/UX), Yapay Zeka Model Entegrasyonu, Flask Backend Geliştirme, Model Onarımı & Yükleme Süreci, Web Yayınına Hazırlık
- **Selsabil Aya Belkabla** - Metin Analiz Modelini Geliştirme, Model Onarımı
- **Sarah Alayi** - Metin Analiz Açıklama Sistemini Geliştirme

---

*Yapay zeka destekli sahtecilik tespiti için geliştirilmiş açık kaynak projesi.*