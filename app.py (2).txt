import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gtts import gTTS
import os
from transformers import pipeline
import soundfile as sf
from chandas_engine import get_syllable_weights, METRE_RULES, validate_recitation

st.set_page_config(page_title="IKS Chanda-Vox AI", layout="wide")

# Load AI for STT Verification
@st.cache_resource
def load_stt():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

asr_model = load_stt()

st.title("🕉️ Chanda-Vox: Rhythmic Sanskrit Synthesis & Validation")

# --- STEP 1: TEXT INPUT & METRE SELECTION ---
col1, col2 = st.columns([2, 1])
with col1:
    input_text = st.text_area("Enter Sanskrit Verse (Devanagari)", "शुक्लाम्वरधरं विष्णुं")
with col2:
    selected_metre = st.selectbox("Select Metrical Framework", list(METRE_RULES.keys()))
    laya_speed = st.slider("Adjust Laya (Speed)", 0.5, 2.0, 1.0)

if st.button("Generate Rhythmic Recitation"):
    # 1. Metrical Analysis
    weights, tokens = get_syllable_weights(input_text)
    st.write(f"**Syllable Analysis:** {'-'.join(tokens)}")
    st.write(f"**Weight Pattern:** `{weights}`")

    # 2. Text-to-Speech (TTS)
    tts = gTTS(text=input_text, lang='hi') # Using Hindi engine for Sanskrit phonetics
    tts.save("raw_audio.mp3")

    # 3. Speed & Rhythm Management (Librosa)
    y, sr = librosa.load("raw_audio.mp3")
    # Apply Speed Management (Laya)
    y_stretched = librosa.effects.time_stretch(y, rate=laya_speed)
    sf.write("final_output.wav", y_stretched, sr)

    # 4. Speech-to-Text (STT) Verification
    stt_result = asr_model("final_output.wav")["text"]
    
    # --- STEP 2: VIRTUALIZATION & DASHBOARD ---
    st.divider()
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.subheader("🔊 Audio Output")
        st.audio("final_output.wav")
        is_perfect = validate_recitation(stt_result, input_text)
        if is_perfect:
            st.success(f"✅ Phonetic Accuracy Verified: {stt_result}")
        else:
            st.warning(f"⚠️ STT Transcription: {stt_result} (Check pronunciation)")

    with v_col2:
        st.subheader("📈 Waveform Virtualization")
        # Creating a waveform plot
        fig_wave = px.line(y_stretched[::100], title="Time-Domain Waveform")
        st.plotly_chart(fig_wave, use_container_width=True)

    # 5. Pitch & Modulation Analysis
    st.subheader("🎵 Melodic Pitch Contour")
    pitches, magnitudes = librosa.piptrack(y=y_stretched, sr=sr)
    pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1])]
    pitch_values = [p for p in pitch_values if p > 50 and p < 500] # Filter human voice range
    
    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(y=pitch_values, mode='lines', line=dict(color='orange')))
    fig_pitch.update_layout(xaxis_title="Time", yaxis_title="Frequency (Hz)")
    st.plotly_chart(fig_pitch, use_container_width=True)

st.sidebar.markdown("### IKS Framework Documentation")
st.sidebar.info("This system uses the Akṣara-gaṇachandas rules where 1=Laghu (1 matra) and 2=Guru (2 matras). Speed management is handled via DSP time-stretching without pitch distortion.")