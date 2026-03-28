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
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import time
import re

# --- CONFIG ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide", page_icon="🕉️")

# --- PRIEST-VOICE MAPPINGS ---
RAGA_MAP = {
    "Bhairav (Early Morning)": [1.0, 1.06, 1.25, 1.33, 1.5, 1.58, 1.87],
    "Bhairavi (Devotional)": [1.0, 1.06, 1.2, 1.33, 1.5, 1.58, 1.77],
    "Yaman (Evening/Peace)": [1.0, 1.12, 1.26, 1.42, 1.5, 1.68, 1.89],
    "Rigveda Monotone (Udatta)": [1.0, 1.0, 1.0, 1.12, 1.0, 1.0, 0.9] # Traditional 3-tone
}

RASA_MAP = {
    "Gambhira (Deep/Priestly)": {"attack": 0.3, "vibrato": 1.5, "harmonics": [1, 2, 3, 0.5], "breath": 0.1},
    "Karuna (Compassionate)": {"attack": 0.5, "vibrato": 4.0, "harmonics": [1, 1.5, 2], "breath": 0.2},
    "Shanti (Peaceful)": {"attack": 0.4, "vibrato": 1.0, "harmonics": [1, 2], "breath": 0.05},
    "Veera (Powerful/Rigid)": {"attack": 0.02, "vibrato": 0.5, "harmonics": [1, 2, 3, 4], "breath": 0.0}
}

# --- CORE LOGIC ---
def analyze_syllables(text):
    hk_text = transliterate(text, sanscript.DEVANAGARI, sanscript.HK)
    clean = re.sub(r'[\s।॥]', '', hk_text)
    pattern = []
    vowels = "AIURaeiou"
    for i, char in enumerate(clean):
        if char in "aeiou":
            if i + 1 < len(clean) and clean[i+1] not in vowels + " ": pattern.append("G")
            else: pattern.append("L")
        elif char in "AIUReo": pattern.append("G")
    return pattern if pattern else ["L", "G"]

def generate_recitation(pattern, base_freq, raga_name, rasa_name, speed):
    sr = 22050
    audio_full = np.array([])
    intervals = RAGA_MAP[raga_name]
    rasa = RASA_MAP[rasa_name]
    
    for i, weight in enumerate(pattern):
        # 2:1 Rhythmic Ratio
        duration = 0.45 / speed if weight == "L" else 0.9 / speed
        t = np.linspace(0, duration, int(sr * duration))
        
        # Calculate Frequency from Raga
        note_idx = i % len(intervals)
        freq = base_freq * intervals[note_idx]
        
        # Apply Priest-like "Deep" Harmonics (Additive Synthesis)
        wave = np.zeros_like(t)
        for h_idx, h_mult in enumerate(rasa['harmonics']):
            # Amplitude drops as harmonic increases (natural voice physics)
            amp = 1.0 / (h_idx + 1)
            wave += amp * np.sin(2 * np.pi * (freq * h_mult) * t)
        
        # Add "Vibrato" for human-like chanting
        vibrato = 1 + (0.01 * np.sin(2 * np.pi * rasa['vibrato'] * t))
        wave = wave * vibrato
        
        # Add "Breath/Noise" for organic texture
        noise = (np.random.randn(len(t)) * rasa['breath'] * 0.1)
        wave += noise

        # Priest Envelope: Smooth fade in/out
        attack_len = int(len(t) * rasa['attack'])
        envelope = np.ones_like(t)
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
        envelope[-attack_len:] = np.square(np.linspace(1, 0, attack_len)) # Exponential decay
        
        note = (wave * envelope * 0.5)
        audio_full = np.concatenate([audio_full, note, np.zeros(int(sr * 0.03))])
        
    return audio_full, sr

# --- UI ---
st.title("🕉️ NaadBrahma AI")
st.subheader("Authentic Sanskrit Priest-Voice Synthesizer")

c1, c2 = st.columns([1, 1.2])

with c1:
    st.markdown("### 🖋️ Sacred Text")
    verse = st.text_area("Verse", "ॐ त्र्यम्बकं यजामहे सुगन्धिं पुष्टिवर्धनम् ।", height=100)
    
    st.markdown("### 🎚️ Priest Voice Configuration")
    col_a, col_b = st.columns(2)
    with col_a:
        raga_sel = st.selectbox("Melodic Framework", list(RAGA_MAP.keys()))
        rasa_sel = st.selectbox("Vocal Intensity (Rasa)", list(RASA_MAP.keys()))
    with col_b:
        base_hz = st.slider("Voice Depth (Pitch)", 80, 200, 110) # Lower is more 'Priestly'
        tempo = st.slider("Chant Speed", 0.5, 1.5, 0.8)

with c2:
    if st.button("✨ GENERATE CHANT", use_container_width=True):
        pattern = analyze_syllables(verse)
        audio, sr = generate_recitation(pattern, base_hz, raga_sel, rasa_sel, tempo)
        
        st.success(f"Reciting in {rasa_sel} Style...")
        st.audio(audio, sample_rate=sr)
        
        # Show the "Voice Print"
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=audio[:2000], line=dict(color='#FF4B4B')))
        fig.update_layout(title="Acoustic Waveform (Vocal Print)", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.info("💡 **Pro-Tip:** For the most authentic 'Priest' sound, set the **Voice Depth** to **100Hz - 120Hz** and use **Gambhira** style.")

