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

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Chanda-Vox Precision", layout="wide", page_icon="📿")
st.markdown("<style>.stApp { background-color: #050505; color: #FF9933; }</style>", unsafe_allow_html=True)

# --- 2. CHANDAS LOGIC (Letter by Letter) ---
def analyze_aksharas(text):
    # Matches individual Sanskrit syllables (Aksharas)
    tokens = re.findall(r'[क-ह]्?[अ-औा-ौ]?[ंः]?|ॐ', text)
    weights = []
    guru_marks = "आईऊएऐओऔाीूेैोौंः"
    
    for i, t in enumerate(tokens):
        next_t = tokens[i+1] if i+1 < len(tokens) else ""
        if any(m in t for m in guru_marks) or "्" in next_t or t == "ॐ":
            weights.append(2.0) # Guru: 2x duration
        else:
            weights.append(1.0) # Laghu: 1x duration
    return tokens, weights

# --- 3. THE PRIEST SYNTHESIZER ---
def synthesize_priest_chanda(tokens, weights, base_speed):
    combined_audio = []
    sr = 22050
    
    for token, weight in zip(tokens, weights):
        tts = gTTS(text=token, lang='hi')
        temp_fn = f"temp_{token}.mp3"
        tts.save(temp_fn)
        
        y, _ = librosa.load(temp_fn, sr=sr)
        
        # A. Priest Transformation (Deep Baritone)
        y_priest = librosa.effects.pitch_shift(y, sr=sr, n_steps=-5.0)
        
        # B. Letter Clarity Filter (Exciter)
        b, a = [0.8, -0.8], [1, -0.85]
        y_priest = lfilter(b, a, y_priest)
        
        # C. Rhythmic Duration (Chanda Speed)
        # Slower for Guru, Faster for Laghu
        stretch_rate = base_speed / weight
        y_timed = librosa.effects.time_stretch(y_priest, rate=stretch_rate)
        
        combined_audio.extend(y_timed)
        # Minimal gap for clear articulation
        combined_audio.extend(np.zeros(int(sr * 0.04)))
        
        if os.path.exists(temp_fn): os.remove(temp_fn)
        
    final_y = np.array(combined_audio)
    final_y = librosa.util.normalize(final_y)
    return final_y, sr

# --- 4. AI VALIDATOR ---
@st.cache_resource
def load_stt():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", 
                    generate_kwargs={"language": "sanskrit", "task": "transcribe"})

asr_pipe = load_stt()

# --- 5. UI ---
st.title("📿 Veda-Vox: Precision Priest Synth")
st.write("Processing: **ॐ नमः शिवाय** | Analyzing metrical weights for each letter.")

input_text = st.text_input("Enter Shloka", "ॐ नमः शिवाय")
speed_slider = st.slider("Base Laya (Tempo)", 0.5, 2.5, 1.2)

if st.button("Generate & Analyze"):
    tokens, weights = analyze_aksharas(input_text)
    
    # Visualizing the Chanda Weights
    st.write("**Metrical Rhythm Breakdown:**")
    cols = st.columns(len(tokens))
    for i, (t, w) in enumerate(zip(tokens, weights)):
        cols[i].metric(t, "Guru (Long)" if w==2 else "Laghu (Short)")

    with st.spinner("Synthesizing Aksharas..."):
        y_final, sr_final = synthesize_priest_chanda(tokens, weights, speed_slider)
        
        # Save to Buffer for Download
        buffer = io.BytesIO()
        sf.write(buffer, y_final, sr_final, format='WAV')
        buffer.seek(0)
        
        # Audio Player & Download
        st.audio(buffer, format='audio/wav')
        st.download_button(label="📥 Download Priest Recitation (.wav)", 
                          data=buffer, 
                          file_name="priest_chanda.wav", 
                          mime="audio/wav")

    # --- 6. STT VERIFICATION ---
    st.divider()
    st.subheader("🧐 Speech-to-Text (STT) Accuracy")
    # Whisper needs 16kHz
    y_16k = librosa.resample(y_final, orig_sr=sr_final, target_sr=16000)
    stt_result = asr_pipe(y_16k)["text"]
    
    c1, c2 = st.columns(2)
    c1.info(f"**Target:** {input_text}")
    c2.success(f"**AI Heard:** {stt_result}")
    
    # --- 7. SIGNAL ANALYSIS ---
    st.divider()
    st.subheader("📊 High-Definition Signal Dashboard")
    
    # WAVEFORM
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(y=y_final[::5], line=dict(color='#FF9933')))
    fig_wave.update_layout(title="Time-Domain Waveform", paper_bgcolor='black', plot_bgcolor='black', font_color='#FFCC66')
    st.plotly_chart(fig_wave, use_container_width=True)

    # SPECTROGRAM
    st.write("**Letter-Frequency Spectrogram**")
    S = librosa.feature.melspectrogram(y=y_final, sr=sr_final, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig_spec, ax_spec = plt.subplots(figsize=(12, 4))
    fig_spec.patch.set_facecolor('#050505')
    img = librosa.display.specshow(S_dB, sr=sr_final, x_axis='time', y_axis='mel', ax=ax_spec, cmap='plasma')
    ax_spec.tick_params(colors='#FFCC66')
    st.pyplot(fig_spec)
