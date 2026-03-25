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
from difflib import SequenceMatcher
import re
import os
import io

# --- 1. UI & THEME ---
st.set_page_config(page_title="Veda-Vox Continuous", layout="wide")
st.markdown("<style>.stApp { background-color: #0b0e14; color: #ff9933; }</style>", unsafe_allow_html=True)

# --- 2. CHANDAS MAPPING ---
def get_rhythm_map(text):
    # Rule-based weighting for Guru (Long) vs Laghu (Short)
    guru_marks = "आईऊएऐओऔाीूेैोौंः"
    # Identify positions of characters
    weights = []
    for i, char in enumerate(text):
        if char in guru_marks or char == "ॐ":
            weights.append(1.8) # Slow down for melody
        else:
            weights.append(1.0) # Normal speed
    return weights

# --- 3. CONTINUOUS MELODIC SYNTHESIS ---
def synthesize_continuous_priest(text, base_speed):
    sr = 22050
    # Generate the WHOLE sentence for natural flow
    tts = gTTS(text=text, lang='hi')
    tts.save("full_sentence.mp3")
    y, _ = librosa.load("full_sentence.mp3", sr=sr)

    # A. MELODIC TRANSFORMATION
    # -2.5 Pitch shift gives 'Authority' and 'Depth' without losing melody
    y_melodic = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2.5)
    
    # B. SPECTRAL ENHANCER (Letter Clarity)
    # This ensures "S", "T", "P" are loud and clear over the deep voice
    b, a = [1.2, -1.2], [1, -0.95]
    y_clear = lfilter(b, a, y_melodic)

    # C. CONTINUOUS TIME STRETCH (The Rhythm Pulse)
    # We apply a single stretch based on the average weight of the verse
    # to maintain sentence flow while honoring the Chandas
    weights = get_rhythm_map(text)
    avg_weight = np.mean(weights) if weights else 1.0
    effective_speed = base_speed / avg_weight
    
    y_final = librosa.effects.time_stretch(y_clear, rate=effective_speed)
    
    # D. HARMONIC WARMTH (Saturation)
    y_final = np.tanh(y_final * 1.5) 
    
    if os.path.exists("full_sentence.mp3"): os.remove("full_sentence.mp3")
    return librosa.util.normalize(y_final), sr

# --- 4. VALIDATOR ---
@st.cache_resource
def load_stt():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", 
                    generate_kwargs={"language": "sanskrit", "task": "transcribe"})

asr_pipe = load_stt()

# --- 5. UI DISPLAY ---
st.title("🕉️ Veda-Vox: Continuous Melodic Priest")
st.write("Synthesizing natural, deep-resonant Sanskrit speech with high-definition letter clarity.")

input_text = st.text_input("Enter Sanskrit Shloka", "ॐ नमः शिवाय")
laya = st.slider("Chant Speed (Laya)", 0.6, 1.8, 1.0)

if st.button("Synthesize Full Recitation"):
    with st.spinner("Processing Melodic Harmonics..."):
        y_audio, sr_audio = synthesize_continuous_priest(input_text, laya)
        
        # Audio Storage
        buffer = io.BytesIO()
        sf.write(buffer, y_audio, sr_audio, format='WAV')
        buffer.seek(0)
        
        st.subheader("🔊 Priest Recitation")
        st.audio(buffer)
        st.download_button("📥 Download WAV", buffer, "priest_recitation.wav", "audio/wav")

    # --- 6. PHONETIC VALIDATION ---
    st.divider()
    st.subheader("🧐 Speech-to-Text (STT) Verification")
    y_16k = librosa.resample(y_audio, orig_sr=sr_audio, target_sr=16000)
    stt_result = asr_pipe(y_16k)["text"]
    
    col1, col2 = st.columns(2)
    col1.metric("Input Verse", input_text)
    col2.metric("AI Transcription", stt_result)
    
    # --- 7. SIGNAL VISUALS ---
    st.divider()
    st.subheader("📊 Signal Analysis Dashboard")
    
    # WAVEFORM
    
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(y=y_audio[::10], line=dict(color='#ff9933', width=1.5)))
    fig_w.update_layout(title="Continuous Vocal Waveform", height=300, 
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_w, use_container_width=True)

    # SPECTROGRAM
    
    st.write("**Letter-Frequency Articulation (Spectrogram)**")
    S = librosa.feature.melspectrogram(y=y_audio, sr=sr_audio, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig_s, ax_s = plt.subplots(figsize=(12, 4))
    fig_s.patch.set_facecolor('#0b0e14')
    img = librosa.display.specshow(S_dB, sr=sr_audio, x_axis='time', y_axis='mel', ax=ax_s, cmap='magma')
    ax_s.tick_params(colors='#ffcc66')
    st.pyplot(fig_s)
