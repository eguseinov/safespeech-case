import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os
import time
import io 
import soundfile as sf

# --- AYARLAR ---
MODEL_PATH = "model/best_model_v2.keras" 
SAMPLE_RATE = 16000
IMG_SIZE = (128, 128)
DURATION = 5 

# --- 1. MODELİ YÜKLEME (Cache ile hızlandırılmış) ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model dosyası bulunamadı: {MODEL_PATH}. Lütfen modeli bu klasöre ekleyin.")
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- 2. ÖNİŞLEME FONKSİYONU (Eğitimdekiyle AYNI) ---
def preprocess_audio(audio, sr):
    # Eğer örnekleme hızı farklıysa 16000'e çevir
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # 5 Saniyeye Sabitle (Padding veya Kesme)
    max_len = SAMPLE_RATE * DURATION
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        padding = max_len - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')

    # Tensor Dönüşümü
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    
    # STFT -> Spectrogram -> Log
    spectrogram = tf.signal.stft(audio_tensor, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    
    # Resize (128x128) ve Boyut Ekleme
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.image.resize(spectrogram, IMG_SIZE)
    spectrogram = tf.expand_dims(spectrogram, axis=0) # Batch boyutu (1, 128, 128, 1)
    
    return spectrogram

# --- 3. TAHMİN FONKSİYONU ---
def predict_segment(audio_chunk):
    # Sessizlik Kontrolü (Max Genlik Eşiği - Simülasyondaki aynı ayarlar)
    max_amp = np.max(np.abs(audio_chunk))
    if max_amp < 0.02:
        return "SESSİZLİK", 0.0, max_amp
    
    # Tahmin
    spec = preprocess_audio(audio_chunk, SAMPLE_RATE)
    prob = model.predict(spec, verbose=0)[0][0]
    
    label = "ERKEK" if prob > 0.5 else "KADIN"
    return label, prob, max_amp

# --- ARAYÜZ TASARIMI ---
st.title("Gerçek Zamanlı Cinsiyet Tespiti")
st.markdown("Bu uygulama **TIMIT Veri Seti** ile eğitilmiş CNN modelini kullanır.")

# Sekmeler
tab1, tab2 = st.tabs(["🔴 Senaryo 1: Canlı Mikrofon", "📂 Senaryo 2: Dosya Analizi"])

# --- SENARYO 1: CANLI MİKROFON ---
with tab1:
    st.header("Mikrofon ile Canlı Analiz")
    st.write("Sistem kesintisiz (streaming) modda çalışır.")
    
    # Başlat/Durdur Butonu
    if st.button("🔴 Yayını Başlat / Durdur", key="stream_btn"):
        if 'streaming' not in st.session_state:
            st.session_state.streaming = True
        else:
            st.session_state.streaming = not st.session_state.streaming

    # --- AKIŞ MANTIĞI ---
    if st.session_state.get('streaming'):
        placeholder = st.empty()
        
        # Buffer Hazırlığı
        full_buffer_len = int(SAMPLE_RATE * DURATION)
        if 'audio_buffer' not in st.session_state:
            st.session_state.audio_buffer = np.zeros(full_buffer_len, dtype=np.float32)

        # Blok Ayarları (0.5 sn)
        block_duration = 0.5 
        block_size = int(SAMPLE_RATE * block_duration)
        
        st.success("Mikrofon Açık! Dinleniyor... 🎤")

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=block_size, channels=1) as stream:
                
                while st.session_state.streaming:
                    # 1. Veri Okuma
                    new_data, overflowed = stream.read(block_size)
                    new_data = new_data.flatten()
                    
                    if overflowed: st.warning("⚠️ Veri atlandı!")

                    # 2. Buffer Kaydırma
                    st.session_state.audio_buffer = np.roll(st.session_state.audio_buffer, -block_size)
                    st.session_state.audio_buffer[-block_size:] = new_data
                    
                    # 3. Analiz Hazırlığı
                    current_audio = st.session_state.audio_buffer
                    
                    # --- DÜZELTME 1: ANLIK SESSİZLİK KONTROLÜ ---
                    # Yeni gelen 0.5 saniyeye bakıyoruz , böylece sustuğun an sistem de susar.
                    instant_amp = np.max(np.abs(new_data))
                    
                    with placeholder.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Eşik Değeri 0.02 olarak belirlendi.
                            if instant_amp < 0.02: 
                                st.warning("### 😶 SESSİZLİK")
                                st.progress(0, text="Sinyal Bekleniyor...")
                            else:
                                label, prob, _ = predict_segment(current_audio)
                                prob_val = float(prob)
                                prob_val = max(0.0, min(1.0, prob_val))
                                
                                # --- DÜZELTME 2: GÜVEN EŞİĞİ (GÜRÜLTÜ FİLTRESİ) ---
                                # Model %40 ile %60 arasında kaldıysa "Emin değilim" desin.
                                if 0.40 < prob_val < 0.60:
                                    st.warning("### ⚠️ BELİRSİZ SES / GÜRÜLTÜ")
                                    st.progress(0.5, text="Model Kararsız (%50)")
                                
                                elif prob_val <= 0.40: # Kesin Kadın
                                    # Güven skorunu 0-1 arasına yaymak için ters işlem
                                    confidence = (1.0 - prob_val)
                                    st.success(f"### 👩 KADIN (Güven: %{confidence*100:.1f})")
                                    st.progress(confidence, text="Kadın Sesi Olasılığı")
                                    
                                else: # Kesin Erkek (prob_val >= 0.60)
                                    st.info(f"### 👨 ERKEK (Güven: %{prob_val*100:.1f})")
                                    st.progress(prob_val, text="Erkek Sesi Olasılığı")
                        
                        with col2:
                            st.metric("Anlık Şiddet", f"{float(instant_amp):.4f}")
                    
                    time.sleep(0.01)

        except Exception as e:
            st.error(f"Hata: {e}")
            st.session_state.streaming = False

# --- SENARYO 2: DOSYA ANALİZİ ---
with tab2:
    st.header("Gelişmiş Ses Dosyası Analizi")
    uploaded_file = st.file_uploader(
        "Bir ses dosyası yükleyin (WAV, MP3, OGG, FLAC)", 
        type=["wav", "mp3", "ogg", "flac"]
    )
    
    if uploaded_file is not None:
        try:
            # --- 2. Geliştirme: EVRENSEL OKUYUCU ---
            # Dosya ne formatta olursa olsun (mp3, nist wav, flac...) Librosa bunu okur ve standart 16000 Hz array'e çevirir.
            y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
            
            # --- 3. Geliştirme: EVRENSEL OYNATICI ---
            # Tarayıcının kafası karışmasın diye, elimizdeki temiz array'i sanal bir WAV dosyasına yazıp oynatıcıya veriyoruz.
            virtual_wav = io.BytesIO()
            sf.write(virtual_wav, y, SAMPLE_RATE, format='WAV', subtype='PCM_16')
            
            # Oynatıcıyı göster
            st.audio(virtual_wav, format='audio/wav')
            
            # --- 4. ANALİZ BUTONU ---
            if st.button("Dosyayı Analiz Et"):
                with st.spinner('Ses işleniyor...'):
                    # 'y' değişkeni zaten yukarıda yüklendiği için tekrar yüklemeye gerek yok!
                    # Direkt 'y' üzerinden analize devam ediyoruz.
                    
                    duration_sec = len(y) / SAMPLE_RATE
                    results = []
                    progress_bar = st.progress(0)
                    
                    # KISA DOSYA (Tek Parça)
                    if duration_sec < DURATION:
                        label, prob, _ = predict_segment(y)
                        results.append({
                            "Zaman": f"0.0sn - {duration_sec:.1f}sn",
                            "Tahmin": label,
                            "Olasılık (Erkek)": f"{prob:.4f}"
                        })
                        progress_bar.progress(100)
                        
                    # UZUN DOSYA (Kayar Pencere)
                    else:
                        step = 1.0 
                        max_range = len(y) - int(SAMPLE_RATE * DURATION) + 1
                        if max_range <= 0: max_range = 1 

                        for i in range(0, max_range, int(SAMPLE_RATE * step)):
                            chunk = y[i : i + int(SAMPLE_RATE * DURATION)]
                            if len(chunk) < int(SAMPLE_RATE * DURATION): break

                            label, prob, _ = predict_segment(chunk)
                            current_time = i / SAMPLE_RATE
                            results.append({
                                "Zaman": f"{current_time:.1f}sn - {current_time+5:.1f}sn",
                                "Tahmin": label,
                                "Olasılık (Erkek)": f"{prob:.4f}"
                            })
                            progress_bar.progress(min(i / len(y), 1.0))
                    
                    progress_bar.progress(100)
                    st.success("Analiz Tamamlandı!")
                    st.table(results)

        except Exception as e:
            st.error(f"Dosya işlenirken hata oluştu: {e}")
            st.warning("İpucu: Eğer MP3 hatası alıyorsanız sisteminizde 'ffmpeg' kurulu olmayabilir. WAV dosyası deneyin.")