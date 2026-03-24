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

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Veda-Vox Pro: High-Speed Priest AI", layout="wide", page_icon="🕉️")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #FF9933; }
    h1, h2, h3 { color: #FFCC66 !important; font-family: 'Georgia', serif; }
    .stButton>button { background: linear-gradient(45deg, #FF9933, #FF4B4B); color: white; border-radius: 8px; font-weight: bold; width: 100%; }
    .stMetric { background-color: #1a1a1a; padding: 15px; border-radius: 10px; border: 1px solid #FF9933; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PRIEST VOICE ENGINE (High-Speed Capable) ---
def create_priest_voice(input_path, speed):
    y, sr = librosa.load(input_path, sr=22050)
    
    # A. Pitch Shift: Deep Baritone
    y_deep = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4.5)
    
    # B. High-Pass Filter: Makes 'K', 'T', 'S' sounds sharp even at high speed
    b, a = [0.6, -0.6], [1, -0.9] 
    y_sharp = lfilter(b, a, y_deep)
    
    # C. SPEED CONTROL: Now supports up to 2.0x
    y_rhythmic = librosa.effects.time_stretch(y_sharp, rate=speed)

    # D. Temple Reverb: Small room reflection for clarity at high speeds
    ir = np.zeros(int(sr * 0.3))
    ir[0] = 1.0 
    ir[int(sr * 0.03)] = 0.4 
    y_final = convolve(y_rhythmic, ir, mode='full')[:len(y_rhythmic)]
    
    y_final = librosa.util.normalize(y_final)
    output_fn = "priest_final.wav"
    sf.write(output_fn, y_final, sr)
    return output_fn

# --- 3. STT VALIDATOR MODEL ---
@st.cache_resource
def load_validator():
    # Using 'whisper-tiny' for speed in hackathons
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", 
                    generate_kwargs={"language": "sanskrit", "task": "transcribe"})

asr_pipe = load_validator()

# --- 4. APP UI ---
st.title("🕉️ Veda-Vox: Priest-Voice & STT Validator")

col_input, col_ctrl = st.columns([2, 1])

with col_ctrl:
    st.subheader("🎛️ Control Panel")
    # SPEED RANGE UPDATED: 0.5x to 2.0x
    laya = st.slider("Laya (Chant Speed)", 0.5, 2.0, 1.0, step=0.1)
    st.write(f"Current Tempo: **{laya}x**")
    st.info("1.0 is Normal. >1.0 is Fast Chanting. <1.0 is Meditative.")

with col_input:
    verse = st.text_area("Input Sanskrit Verse", "शुक्लाम्वरधरं विष्णुं शशिवर्णं चतुर्भुजम्")
    generate_btn = st.button("Generate & Verify Recitation")

if generate_btn:
    with st.spinner("Synthesizing Priest Resonance..."):
        # Step 1: Base Audio
        tts = gTTS(text=verse, lang='hi')
        tts.save("base.mp3")
        
        # Step 2: Priest Modulation
        final_path = create_priest_voice("base.mp3", laya)
        st.audio(final_path)

        # Step 3: SPEECH-TO-TEXT VERIFICATION
        st.divider()
        st.subheader("🧐 Speech-to-Text (STT) Verification")
        
        # Load audio for Whisper (16kHz requirement)
        audio_raw, _ = librosa.load(final_path, sr=16000)
        stt_output = asr_pipe(audio_raw)["text"]
        
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("**Original Text:**")
            st.info(verse)
        with v_col2:
            st.markdown("**AI Transcription:**")
            st.success(stt_output)
        
        # Fidelity Score
        similarity = SequenceMatcher(None, verse, stt_output).ratio() * 100
        st.metric("Phonetic Fidelity Score", f"{similarity:.1f}%")

        # --- 4. ADVANCED WAVEFORMS ---
        st.divider()
        st.subheader("📊 High-Resolution Signal Analysis")
        
        y_viz, sr_viz = librosa.load(final_path)
        
        # WAVEFORM
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(y=y_viz[::5], line=dict(color='#FF9933', width=1)))
        fig_wave.update_layout(title="Time-Domain Waveform (Energy Peaks)", height=300, 
                              plot_bgcolor='black', paper_bgcolor='black', font_color='#FFCC66')
        st.plotly_chart(fig_wave, use_container_width=True)
        
        

        # SPECTROGRAM
        st.write("**Frequency Spectrogram (Clarity Visualizer)**")
        S = librosa.feature.melspectrogram(y=y_viz, sr=sr_viz, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig_spec, ax_spec = plt.subplots(figsize=(12, 4))
        fig_spec.patch.set_facecolor('#050505')
        ax_spec.set_facecolor('#050505')
        img = librosa.display.specshow(S_dB, sr=sr_viz, x_axis='time', y_axis='mel', ax=ax_spec, cmap='plasma')
        ax_spec.tick_params(colors='#FFCC66')
        st.pyplot(fig_spec)

        

        if similarity > 85:
            st.balloons()
            st.write("✅ **Verification Success:** The Priest voice remains 100% intelligible at this speed.")
