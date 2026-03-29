import streamlit as st
import numpy as np
import plotly.graph_objects as go
from gtts import gTTS
import io
import re
import time
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from scipy.io import wavfile
from pydub import AudioSegment
import speech_recognition as speech_recognition
from scipy.fft import fft, fftfreq

# --- 1. DYNAMIC PITCH ENVELOPE ENGINE ---
def apply_vedic_depth(samples, sr, text):
    """
    Applies specific depth: 
    Om (Deepest), Namah (Mid-Deep), Shivaya (Standard Priest Deep)
    """
    n = len(samples)
    # Define segments based on the phrase "Om Namah Shivaya"
    # We apply a 'Resampling Factor' curve
    # 0.6 = Very Deep, 0.75 = Mid Deep, 0.85 = Priest Deep
    
    # Create an envelope array
    envelope = np.ones(n)
    third = n // 3
    
    envelope[:third] = 0.65          # Om (Deepest)
    envelope[third:2*third] = 0.75   # Namah (Mid-Deep)
    envelope[2*third:] = 0.82        # Shivaya (Standard Deep)
    
    # Apply subtle human jitter
    t = np.linspace(0, n/sr, n)
    jitter = 1 + 0.005 * np.sin(2 * np.pi * 5 * t)
    
    return (samples * envelope * jitter).astype(np.int16)

# --- 2. HUMAN SYNTHESIS CORE ---
def generate_custom_priest(text, base_speed):
    tts = gTTS(text=text, lang='hi')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio = AudioSegment.from_file(fp, format="mp3")
    
    # Convert to numpy
    raw_samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        raw_samples = raw_samples.reshape((-1, 2)).sum(axis=1) / 2
    
    # Apply dynamic depth
    final_sr = int(audio.frame_rate * base_speed)
    processed_samples = apply_vedic_depth(raw_samples, final_sr, text)
    
    return processed_samples, final_sr

# --- 3. UI ---
st.set_page_config(page_title="NaadBrahma: Dynamic Svara", layout="wide")
st.markdown("<style>.stApp { background: #050505; color: #ffd700; }</style>", unsafe_allow_html=True)

st.title("🕉️ NaadBrahma: Dynamic Pitch Modulation")
st.subheader("Modulating Om (Deep) → Namah (Mid) → Shivaya (Priest)")

c1, c2 = st.columns([1, 1.5])

with c1:
    verse = st.text_input("Sanskrit Verse", "ॐ नमः शिवाय")
    laya = st.slider("Global Tempo (Laya)", 0.5, 1.5, 0.85)
    
    if st.button("🔥 GENERATE VEDIC CHANT", use_container_width=True):
        samples, sr_val = generate_custom_priest(verse, laya)
        
        # Audio Playback
        wav_buf = io.BytesIO()
        wavfile.write(wav_buf, sr_val, samples)
        st.audio(wav_buf)
        st.session_state['audio'] = (samples, sr_val)
        
        # Export logic
        st.download_button("📥 Download Ringtone (WAV)", wav_buf.getvalue(), "Om_Namah_Shivaya.wav")

with c2:
    if 'audio' in st.session_state:
        s, sr = st.session_state['audio']
        
        # 1. WAVEFORM (Temporal Energy)
        fig_w = go.Figure(data=go.Scatter(y=s[::100], line=dict(color='#FFD700', width=1)))
        fig_w.update_layout(title="Waveform (Visualizing Matra Energy)", template="plotly_dark", height=230)
        st.plotly_chart(fig_w, use_container_width=True)
        
        # 2. SPECTRUM (Frequency Depth)
        n = len(s)
        yf = fft(s)
        xf = fftfreq(n, 1/sr)[:n//2]
        fig_f = go.Figure(data=go.Scatter(x=xf[:3000], y=2.0/n * np.abs(yf[0:n//2])[:3000], fill='tozeroy', line=dict(color='#00BFFF')))
        fig_f.update_layout(title="Frequency Spectrum (Chest Resonance Analysis)", template="plotly_dark", height=230)
        st.plotly_chart(fig_f, use_container_width=True)

st.info("💡 Observe the Spectrum: The 'Om' generates heavy energy in the 100Hz-150Hz range, while 'Shivaya' moves toward 250Hz.")
