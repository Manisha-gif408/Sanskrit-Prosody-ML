import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from gtts import gTTS
from transformers import pipeline
from scipy.signal import lfilter
import re
import os
import io

# --- 1. UI SETUP ---
st.set_page_config(page_title="Veda-Vox Melodic", layout="wide")
st.markdown("<style>.stApp { background-color: #0b0e14; color: #ff9933; }</style>", unsafe_allow_html=True)

# --- 2. ADVANCED MELODIC ENGINE ---
def synthesize_vedic_chant(text, base_speed):
    sr = 22050
    
    # Check if "Om" is present to handle it separately
    has_om = "ॐ" in text or "ओम्" in text or "Om" in text.lower()
    clean_text = text.replace("ॐ", "").strip()

    # A. SYNTHESIZE "OM" (Deep, Long, Resonant)
    if has_om:
        tts_om = gTTS(text="ओम्", lang='hi')
        tts_om.save("om.mp3")
        y_om, _ = librosa.load("om.mp3", sr=sr)
        # Deepen Om significantly (-4 steps) and stretch it (0.6x speed = 40% slower)
        y_om = librosa.effects.pitch_shift(y_om, sr=sr, n_steps=-4.5)
        y_om = librosa.effects.time_stretch(y_om, rate=0.55) 
        # Bass Boost for Om
        y_om = np.tanh(y_om * 2.0) 
        os.remove("om.mp3")
    else:
        y_om = np.array([])

    # B. SYNTHESIZE SHLOKA (Melodic, Clear, Authoritative)
    tts_main = gTTS(text=clean_text, lang='hi')
    tts_main.save("main.mp3")
    y_main, _ = librosa.load("main.mp3", sr=sr)
    
    # Melodic shift (-2.5 is deep but musical)
    y_main = librosa.effects.pitch_shift(y_main, sr=sr, n_steps=-2.8)
    
    # Spectral Sharpening (Letter Clarity)
    # This makes consonants "pop" so pronunciation is crystal clear
    b, a = [1.3, -1.3], [1, -0.96]
    y_main = lfilter(b, a, y_main)
    
    # Natural Laya (Speed)
    y_main = librosa.effects.time_stretch(y_main, rate=base_speed)
    
    # Harmonic Warmth (Melodic Texture)
    y_main = np.tanh(y_main * 1.3)
    os.remove("main.mp3")

    # C. COMBINE (Natural Flow)
    # Add a very tiny melodic transition gap
    gap = np.zeros(int(sr * 0.1))
    y_final = np.concatenate([y_om, gap, y_main]) if has_om else y_main
    
    return librosa.util.normalize(y_final), sr

# --- 3. VALIDATOR ---
@st.cache_resource
def load_stt():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", 
                    generate_kwargs={"language": "sanskrit", "task": "transcribe"})

asr_pipe = load_stt()

# --- 4. UI DISPLAY ---
st.title("🕉️ Veda-Vox: Deep Om & Melodic Shloka")
st.write("Specialized for **resonant Pranava (Om)** and **musical phonetic clarity**.")

input_text = st.text_input("Enter Verse (Start with ॐ for best effect)", "ॐ नमः शिवाय")
laya = st.slider("Chant Pace", 0.7, 1.5, 1.0)

if st.button("Generate Sacred Chant"):
    with st.spinner("Refining Vocal Formants..."):
        y_audio, sr_audio = synthesize_vedic_chant(input_text, laya)
        
        buffer = io.BytesIO()
        sf.write(buffer, y_audio, sr_audio, format='WAV')
        buffer.seek(0)
        
        st.subheader("🔊 Audio Output")
        st.audio(buffer)
        st.download_button("📥 Save Recitation (.wav)", buffer, "vedic_chant.wav", "audio/wav")

    # --- 5. STT VERIFICATION ---
    st.divider()
    st.subheader("🧐 Phonetic Accuracy Verification")
    y_16k = librosa.resample(y_audio, orig_sr=sr_audio, target_sr=16000)
    stt_result = asr_pipe(y_16k)["text"]
    
    col1, col2 = st.columns(2)
    col1.metric("Target Input", input_text)
    col2.metric("AI Transcription", stt_result)

    # --- 6. VISUAL ANALYSIS ---
    st.divider()
    st.subheader("📊 Melodic Analysis Dashboard")
    
    # WAVEFORM
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(y=y_audio[::10], line=dict(color='#ff9933', width=1)))
    fig_w.update_layout(title="Vocal Energy (See the deep Om peaks at the start)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_w, use_container_width=True)
    

    # SPECTROGRAM
    st.write("**Frequency Harmonics (Showing Melodic Resonance)**")
    S = librosa.feature.melspectrogram(y=y_audio, sr=sr_audio, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig_s, ax_s = plt.subplots(figsize=(12, 4))
    fig_s.patch.set_facecolor('#0b0e14')
    img = librosa.display.specshow(S_dB, sr=sr_audio, x_axis='time', y_axis='mel', ax=ax_s, cmap='magma')
    ax_s.tick_params(colors='#ffcc66')
    st.pyplot(fig_s)
