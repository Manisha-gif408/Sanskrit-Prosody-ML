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

# --- 1. THEME & UI ---
st.set_page_config(page_title="Veda-Vox: Melodic Prosody", layout="wide", page_icon="🕉️")
st.markdown("<style>.stApp { background-color: #0b0e14; color: #ff9933; }</style>", unsafe_allow_html=True)

# --- 2. CHANDAS ENGINE ---
def analyze_metrics(text):
    # Matches syllables (Aksharas) like 'ॐ', 'न', 'मः', 'शि'
    tokens = re.findall(r'[क-ह]्?[अ-औा-ौ]?[ंः]?|ॐ', text)
    weights = []
    guru_marks = "आईऊएऐओऔाीूेैोौंः"
    
    for i, t in enumerate(tokens):
        next_t = tokens[i+1] if i+1 < len(tokens) else ""
        if any(m in t for m in guru_marks) or "्" in next_t or t == "ॐ":
            weights.append(1.8) # Guru: Long/Melodic
        else:
            weights.append(1.0) # Laghu: Short/Quick
    return tokens, weights

# --- 3. MELODIC SYNTHESIS ---
def generate_melodic_voice(tokens, weights, base_speed):
    full_audio = []
    sr = 22050
    
    for token, weight in zip(tokens, weights):
        tts = gTTS(text=token, lang='hi')
        temp_fn = f"syl_{token}.mp3"
        tts.save(temp_fn)
        
        y, _ = librosa.load(temp_fn, sr=sr)
        
        # A. Melodic Pitch: -2.8 is the "Sweet Spot" for a resonant Priest voice
        y_mod = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2.8)
        
        # B. Rhythmic Laya: Stretching Guru syllables for melody
        stretch = base_speed / weight
        y_timed = librosa.effects.time_stretch(y_mod, rate=stretch)
        
        # C. Harmonic Saturation: Adds human 'warmth' to the digital signal
        y_warm = np.tanh(y_timed * 1.4) 
        
        full_audio.extend(y_warm)
        # 15ms micro-pause for distinct letter articulation
        full_audio.extend(np.zeros(int(sr * 0.015)))
        
        if os.path.exists(temp_fn): os.remove(temp_fn)
        
    final_y = np.array(full_audio)
    return librosa.util.normalize(final_y), sr

# --- 4. VALIDATOR ---
@st.cache_resource
def load_stt():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", 
                    generate_kwargs={"language": "sanskrit", "task": "transcribe"})

asr_pipe = load_stt()

# --- 5. UI LAYOUT ---
st.title("🕉️ Veda-Vox: Syllabic Melodic Synthesis")
st.write("Generating **Deep Melodic** recitation with **Letter-by-Letter** prosody weighting.")

input_text = st.text_input("Enter Sanskrit Text", "ॐ नमः शिवाय")
laya = st.slider("Chant Pace (Laya)", 0.5, 2.0, 1.1)

if st.button("Synthesize Priest Melody"):
    tokens, weights = analyze_metrics(input_text)
    
    # Show Syllable Analysis
    st.write("**Syllable Weight Analysis:**")
    cols = st.columns(len(tokens))
    for i, (t, w) in enumerate(zip(tokens, weights)):
        cols[i].info(f"{t}\n({'Guru' if w > 1 else 'Laghu'})")

    with st.spinner("Modulating Vocal Formants..."):
        y_res, sr_res = generate_melodic_voice(tokens, weights, laya)
        
        # Buffer for Playback/Download
        buf = io.BytesIO()
        sf.write(buf, y_res, sr_res, format='WAV')
        buf.seek(0)
        
        st.audio(buf)
        st.download_button("📥 Download Master Recitation", buf, "priest_melody.wav", "audio/wav")

    # --- 6. PHONETIC SCORE HIGHLIGHTS ---
    st.divider()
    st.subheader("🧐 Phonetic Score & Validation")
    y_16 = librosa.resample(y_res, orig_sr=sr_res, target_sr=16000)
    stt_out = asr_pipe(y_16)["text"]
    
    # Highlight Logic
    score = SequenceMatcher(None, input_text.replace(" ", ""), stt_out.replace(" ", "")).ratio() * 100
    st.metric("Overall Phonetic Fidelity", f"{score:.1f}%")
    
    st.write("**Transcription Accuracy per Letter:**")
    # Visually mapping the STT result
    st.success(f"Synthesized: {input_text}  ➡️  AI Detected: {stt_out}")

    # --- 7. SIGNAL ANALYSIS ---
    st.divider()
    st.subheader("📊 High-Definition Signal Analysis")
    
    # WAVEFORM
    
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(y=y_res[::10], line=dict(color='#ff9933', width=1)))
    fig_w.update_layout(title="Vocal Energy Waveform", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_w, use_container_width=True)

    # SPECTROGRAM
    
    st.write("**High-Res Harmonics (Spectrogram)**")
    S = librosa.feature.melspectrogram(y=y_res, sr=sr_res, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig_s, ax_s = plt.subplots(figsize=(12, 4))
    fig_s.patch.set_facecolor('#0b0e14')
    img = librosa.display.specshow(S_dB, sr=sr_res, x_axis='time', y_axis='mel', ax=ax_s, cmap='magma')
    ax_s.tick_params(colors='#ffcc66')
    st.pyplot(fig_s)
