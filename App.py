import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from gtts import gTTS
from transformers import pipeline
from scipy.signal import convolve, lfilter
from difflib import SequenceMatcher
import re
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Vedic Chanda-Vox Pro", layout="wide", page_icon="📿")

# --- CUSTOM TEMPLE THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #FF9933; }
    h1, h2, h3 { color: #FFCC66 !important; font-family: 'Georgia', serif; }
    .stButton>button { background: linear-gradient(45deg, #FF9933, #FF4B4B); color: white; border-radius: 8px; border: none; height: 3em; font-weight: bold; }
    .stMetric { background-color: #1a1a1a; padding: 15px; border-radius: 10px; border: 1px solid #FF9933; }
    </style>
    """, unsafe_allow_html=True)

# --- AUDIO PROCESSING ENGINE (PRIEST MODE) ---
def create_priest_voice(input_path, speed):
    y, sr = librosa.load(input_path, sr=22050)
    
    # 1. DEEP RESONANCE (Pitch + Formant Shift)
    # We shift pitch down significantly for that 'Vedic Baritone'
    y_deep = librosa.effects.pitch_shift(y, sr=sr, n_steps=-5.0)
    
    # 2. LETTER CLARITY (High-Shelf Filter)
    # Boosts frequencies above 3kHz so consonants are 'loud' and 'clear'
    b, a = [0.5, -0.5], [1, -0.95] # Simple high-pass to sharpen 's' and 't' sounds
    y_sharp = lfilter(b, a, y_deep)
    
    # 3. RHYTHMIC STRETCH (Laya)
    y_rhythmic = librosa.effects.time_stretch(y_sharp, rate=speed)

    # 4. TEMPLE ACOUSTICS (Dense Reverb)
    # This creates the 'Garbhagriha' feel without washing out the letters
    ir = np.zeros(int(sr * 0.4))
    ir[0] = 1.0 # Direct clarity
    ir[int(sr * 0.04)] = 0.5 # Early reflection for body
    ir[int(sr * 0.12)] = 0.2 # Distant wall
    y_final = convolve(y_rhythmic, ir, mode='full')[:len(y_rhythmic)]
    
    # Normalize to prevent clipping
    y_final = librosa.util.normalize(y_final)
    
    output_fn = "priest_final_pro.wav"
    sf.write(output_fn, y_final, sr)
    return output_fn

# --- AI VALIDATOR ---
@st.cache_resource
def load_validator():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", 
                    generate_kwargs={"language": "sanskrit", "task": "transcribe"})

asr_pipe = load_validator()

# --- APP UI ---
st.title("📿 Veda-Vox: Authentic Priest Voice Synthesis")
st.write("Generating deep-resonant Vedic recitation with high-clarity phonetic output.")

col_main, col_side = st.columns([3, 1])

with col_side:
    st.subheader("Vocal Settings")
    laya = st.slider("Laya (Tempo)", 0.5, 1.0, 0.75)
    st.info("Lower Laya provides more 'Bhāva' and depth to each syllable.")

with col_main:
    verse = st.text_area("Sanskrit Shloka", "शुक्लाम्वरधरं विष्णुं शशिवर्णं चतुर्भुजम्")
    
    if st.button("Generate Intense Priest Recitation"):
        with st.spinner("Synthesizing Authentic Resonance..."):
            
            # 1. Generate Base & Process
            tts = gTTS(text=verse, lang='hi')
            tts.save("base.mp3")
            final_path = create_priest_voice("base.mp3", laya)
            
            # 2. Audio Player
            st.audio(final_path)
            
            # 3. Accuracy Check
            audio_raw, _ = librosa.load(final_path, sr=16000)
            stt_text = asr_pipe(audio_raw)["text"]
            score = SequenceMatcher(None, verse, stt_text).ratio() * 100
            
            st.metric("Phonetic Integrity Score", f"{score:.1f}%")
            
            # --- 4. THE VISUAL STACK ---
            st.divider()
            st.subheader("📊 Signal Analysis Dashboard")
            
            y, sr = librosa.load(final_path)
            
            # ROW 1: WAVEFORM & PITCH
            v1, v2 = st.columns(2)
            with v1:
                st.write("**High-Fidelity Waveform**")
                fig_wave = go.Figure()
                fig_wave.add_trace(go.Scatter(y=y[::10], line=dict(color='#FF9933')))
                fig_wave.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_wave, use_container_width=True)
            
            with v2:
                st.write("**Pitch Modulation (Melody)**")
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                p_vals = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if np.any(pitches[:, t] > 10)]
                fig_p = go.Figure(data=go.Scatter(y=p_vals, line=dict(color='#FF4B4B')))
                fig_p.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_p, use_container_width=True)

            # ROW 2: HIGH-RES SPECTROGRAM
            st.write("**High-Resolution Power Spectrogram**")
            # Using a larger window for clearer frequency resolution
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            
            
            fig_s, ax_s = plt.subplots(figsize=(12, 4))
            fig_s.patch.set_facecolor('#050505')
            ax_s.set_facecolor('#050505')
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax_s, cmap='plasma')
            ax_s.tick_params(colors='#FFCC66')
            st.pyplot(fig_s)
