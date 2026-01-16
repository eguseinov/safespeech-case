# 🎙️ SafeSpeech Case: Real-Time Gender Recognition System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **TIMIT Veri Seti** ile eğitilmiş, uçtan uca Derin Öğrenme (CNN) tabanlı ve **gerçek zamanlı (real-time)** çalışan cinsiyet tespit sistemi.

---

## 📱 Uygulama Önizlemesi

Sistem, canlı mikrofon verisini veya yüklediğiniz ses dosyasını işleyerek anlık cinsiyet tahmini yapar.

![Uygulama Ekran Görüntüsü](assets/app_screenshot.png)

---

## 📂 Proje Yapısı

Repoyu bilgisayarınıza indirdiğinizde klasör yapısı aşağıdaki gibi görünmelidir:

```
Safespeech-Case/
│
├── README.md                
├── assets/                  
│   └── app_screenshot.png
│    ...
│
└── case_app/                # Uygulama ana klasörü
    ├── app.py               # Streamlit uygulama dosyası
    ├── requirements.txt     # Gerekli kütüphane listesi
    ├── models/              # Eğitilmiş model klasörü
    │   └── best_model_v2.keras
```

---

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde (Localhost) çalıştırmak için aşağıdaki adımları izleyin.

### 1. Repoyu Klonlayın

Projeyi bilgisayarınıza indirin ve proje dizinine girin:

```bash
git clone https://github.com/EGuseinov/Safespeech-Case.git
cd Safespeech-Case/case_app
```

### 2. Sanal Ortam (Virtual Environment) Oluşturun

Bağımlılıkların sistem geneline yayılmaması için izole bir ortam kurun:

**Windows için:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux için:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Gerekli Kütüphaneleri Yükleyin

requirements.txt dosyasındaki paketleri kurun:

```bash
pip install -r requirements.txt
```

> **Not:** Sisteminizde ffmpeg kurulu değilse MP3/OGG dosyalarını okurken hata alabilirsiniz. Windows kullanıcıları için genellikle pip kurulumu yeterlidir, ancak Linux kullanıcılarının aşağıdaki komutu çalıştırması gerekebilir:

```bash
sudo apt-get install ffmpeg
```

### 4. Uygulamayı Başlatın

Streamlit arayüzünü ayağa kaldırın:

```bash
streamlit run app.py
```

Komutu çalıştırdıktan sonra tarayıcınızda otomatik olarak şu adres açılacaktır:

```
http://localhost:8501
```

---

## 📋 Gereksinimler

Tüm gereksinimler `requirements.txt` dosyasında listelenmiştir.

---
## 🎯 Kullanım Örneği

### Canlı Mikrofon Kullanarak

1. Uygulamayı başlatın
2. "Başlat" butonuna basın
3. Mikrofona konuşun
4. Sistem otomatik olarak cinsiyet tahmini yapar

### Ses Dosyası Yükleyerek

1. "Dosya Yükle" seçeneğini seçin
2. Bilgisayarınızdan belirtilen formatlarda ses dosyası seçin
3. Sistem dosyayı işleyip sonucu gösterir

---
