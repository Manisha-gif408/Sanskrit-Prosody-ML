import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from gtts import gTTS
from transformers import pipeline
from scipy.signal import convolve
from difflib import SequenceMatcher
import re
import os

# 1. PAGE CONFIG & CUSTOM THEME (Temple Aesthetic)
st.set_page_config(page_title="Chanda-Vox Priest AI", layout="wide", page_icon="🕉️")

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #FF9933;
    }
    h1, h2, h3 {
        color: #FFCC66 !important;
    }
    .stButton>button {
        background-color: #FF9933;
        color: black;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
    }
    .stTextArea textarea {
        background-color: #1c1c1c !important;
        color: #FFCC66 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. PROSODY LOGIC (Syllabic Weighting)
def get_syllable_metrics(text):
    guru_indicators = "आईऊएऐओऔाीूेैोौंः"
    tokens = re.findall(r'[क-ह]्?[अ-औा-ौ]?[ंः]?', text)
    weights = []
    for i in range(len(tokens)):
        char = tokens[i]
        next_char = tokens[i+1] if i+1 < len(tokens) else ""
        if any(c in char for c in guru_indicators) or "्" in next_char:
            weights.append(2)
        else:
            weights.append(1)
    return weights, tokens

# 3. PRIEST VOICE MODULATION ENGINE
def apply_priest_fx(input_path, laya_speed):
    y, sr = librosa.load(input_path, sr=22050)
    
    # A. Pitch Shift: Move to a deep, resonant Baritone
    y_deep = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4.5)
    
    # B. Speed: Apply rhythmic Laya
    y_rhythmic = librosa.effects.time_stretch(y_deep, rate=laya_speed)

    # C. Temple Reverb: Synthetic Impulse Response (Garbhagriha Acoustics)
    ir = np.zeros(int(sr * 0.6))
    ir[0] = 1.0 # Direct Sound
    ir[int(sr * 0.08)] = 0.5 # Stone wall reflection 1
    ir[int(sr * 0.18)] = 0.3 # Reflection 2
    y_final = convolve(y_rhythmic, ir, mode='full')[:len(y_rhythmic)]
    
    output_fn = "priest_final.wav"
    sf.write(output_fn, y_final, sr)
    return output_fn

# 4. LOAD STT MODEL (For Proof of Clarity)
@st.cache_resource
def load_validator():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", 
                    generate_kwargs={"language": "sanskrit", "task": "transcribe"})

asr_pipe = load_validator()

# 5. UI LAYOUT
st.title("🕉️ Chanda-Vox: High-Fidelity Priest AI")
st.write("Generating melodic, rhythmically constrained Sanskrit recitation with phonetic validation.")

# Sidebar Controls
st.sidebar.header("🎛️ Modulation Controls")
laya = st.sidebar.slider("Laya (Chant Speed)", 0.5, 1.2, 0.75)
st.sidebar.info("Tip: Lower speed mimics slow Vedic chanting.")

# Input Section
input_verse = st.text_area("Enter Sanskrit Verse (Devanagari)", "शुक्लाम्वरधरं विष्णुं शशिवर्णं चतुर्भुजम्")

if st.button("Generate Melodic Priest Recitation"):
    with st.spinner("Invoking the Digital Guru..."):
        
        # Step A: Base AI Synthesis
        tts = gTTS(text=input_verse, lang='hi')
        tts.save("base_ai.mp3")
        
        # Step B: Humanize/Priest Modulation
        weights, tokens = get_syllable_metrics(input_verse)
        final_audio_path = apply_priest_fx("base_ai.mp3", laya)
        
        # Step C: Before and After Comparison
        st.subheader("🔊 Audio Comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.write("🤖 Standard AI (Robotic)")
            st.audio("base_ai.mp3")
        with c2:
            st.write("🕉️ Priest-Modulated (Human)")
            st.audio(final_audio_path)

        # Step D: STT Back-Conversion (The Proof)
        st.divider()
        st.subheader("🧐 Phonetic Accuracy Check")
        audio_check, _ = librosa.load(final_audio_path, sr=16000)
        stt_result = asr_pipe(audio_check)["text"]
        
        val_c1, val_c2 = st.columns(2)
        val_c1.info(f"**Original Input:**\n{input_verse}")
        val_c2.success(f"**AI Transcribed:**\n{stt_result}")
        
        score = SequenceMatcher(None, input_verse, stt_result).ratio() * 100
        st.metric("Transcription Accuracy", f"{score:.1f}%")

        # Step E: Scientific Visualizations
        st.divider()
        st.subheader("📈 Signal Visualizations")
        v1, v2 = st.columns(2)

        with v1:
            st.write("**Melodic Pitch Contour**")
            y_v, sr_v = librosa.load(final_audio_path)
            pitches, magnitudes = librosa.piptrack(y=y_v, sr=sr_v)
            pitch_vals = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if np.any(pitches[:, t] > 0)]
            fig_p = go.Figure(data=go.Scatter(y=pitch_vals, line=dict(color='#FF9933', width=2)))
            st.plotly_chart(fig_p, use_container_width=True)

        with v2:
            st.write("**Power Spectrogram (Harmonics)**")
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y_v)), ref=np.max)
            fig_s, ax_s = plt.subplots(figsize=(10, 5))
            fig_s.patch.set_facecolor('#0e1117')
            ax_s.set_facecolor('#0e1117')
            img = librosa.display.specshow(D, sr=sr_v, x_axis='time', y_axis='log', ax=ax_s, cmap='magma')
            st.pyplot(fig_s)

        # Cleanup
        if os.path.exists("base_ai.mp3"): os.remove("base_ai.mp3")
