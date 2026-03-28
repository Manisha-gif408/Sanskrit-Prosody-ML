import streamlit as st
import numpy as np
import plotly.graph_objects as go
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import re
from scipy.signal import butter, lfilter

# --- AUTHENTIC SANSKRIT PROSODY ENGINE ---

def get_prosody_weights(text):
    """
    Analyzes Sanskrit text to identify Laghu (1 Matra) and Guru (2 Matras).
    Rules: Long vowels, vowels before conjuncts, and ending with Anusvara/Visarga are Guru.
    """
    # Transliterate to IAST for precise vowel/consonant weight checking
    iast = transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
    words = iast.split()
    weights = []
    
    long_vowels = ['ā', 'ī', 'ū', 'ṝ', 'e', 'ai', 'o', 'au']
    
    for word in words:
        # Clean word of punctuation
        clean_word = re.sub(r'[।॥,]', '', word)
        for i, char in enumerate(clean_word):
            if char in ['a', 'i', 'u', 'ṛ', 'l']:
                # Check for weight by position (conjunct consonant following)
                if i + 1 < len(clean_word) and clean_word[i+1] not in 'aeiouāīūṝaiou':
                    weights.append("G")
                else:
                    weights.append("L")
            elif char in long_vowels or char in ['ṁ', 'ḥ']:
                weights.append("G")
        weights.append("P") # Pause between words
    return weights

def apply_vocal_filter(data, cutoff, fs, order=5):
    """Simulates the resonant chamber of a human throat (Low pass)."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def generate_priest_chant(weights, base_freq, raga_intervals, speed):
    fs = 22050
    audio_stream = np.array([])
    
    # Fundamental Matra Timing (Standard: Laghu = 0.3s)
    unit_time = 0.35 / speed 
    
    for i, w in enumerate(weights):
        if w == "P": # Word Pause
            audio_stream = np.concatenate([audio_stream, np.zeros(int(fs * 0.15))])
            continue
            
        # 1:2 Timing Ratio (Matra logic)
        duration = unit_time if w == "L" else unit_time * 2
        t = np.linspace(0, duration, int(fs * duration))
        
        # Frequency selection based on Raga sequence
        freq = base_freq * raga_intervals[i % len(raga_intervals)]
        
        # PRIEST VOICE PHYSICS:
        # 1. Additive Harmonics (Fundamental + Chest + Nasal)
        # S (1.0) + Octave (2.0) + Fifth (1.5) + Sub-harmonic (0.5)
        wave = (1.0 * np.sin(2 * np.pi * freq * t)) + \
               (0.4 * np.sin(2 * np.pi * freq * 2.0 * t)) + \
               (0.2 * np.sin(2 * np.pi * freq * 1.5 * t)) + \
               (0.15 * np.sin(2 * np.pi * freq * 0.5 * t))
        
        # 2. Subtle Frequency Jitter (Humanization)
        jitter = 1 + (0.005 * np.sin(2 * np.pi * 5 * t)) 
        wave *= jitter
        
        # 3. Envelope: 'Soft' Priest Attack (fade in) and 'Mantra' Decay
        env = np.ones_like(t)
        fade_in = int(len(t) * 0.2)
        fade_out = int(len(t) * 0.3)
        env[:fade_in] = np.linspace(0, 1, fade_in)
        env[-fade_out:] = np.cos(np.linspace(0, np.pi/2, fade_out))
        
        chunk = wave * env
        audio_stream = np.concatenate([audio_stream, chunk])

    # Apply Vocal Throat Resonance (Formant Simulation)
    audio_stream = apply_vocal_filter(audio_stream, 1200, fs)
    
    # Normalized Volume
    audio_stream = audio_stream / np.max(np.abs(audio_stream))
    return audio_stream, fs

# --- STREAMLIT UI ---

st.title("🕉️ NaadBrahma AI")
st.markdown("### Advanced Chanda-Aware Priest Voice Synthesis")

# Sidebar for Raga & Style
with st.sidebar:
    st.header("🎵 Melodic Framework")
    raga = st.selectbox("Raga", ["Bhairav", "Bhairavi", "Yaman", "Shanti Monotone"])
    rasa = st.select_slider("Rasa (Intensity)", ["Shanti", "Karuna", "Veera"])
    base_hz = st.slider("Priest Pitch (Hz)", 80, 180, 105) # 105Hz is deep 'Pandit' range
    pace = st.slider("Chant Speed", 0.5, 2.0, 1.0)

# Main UI
verse = st.text_area("Enter Sanskrit Verse (Devanagari)", "धर्माक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।", height=150)

if st.button("🔥 GENERATE VEDIC RECITATION", use_container_width=True):
    # Mapping Raga intervals
    scales = {
        "Bhairav": [1.0, 1.06, 1.25, 1.5, 1.58], 
        "Bhairavi": [1.0, 1.06, 1.2, 1.5, 1.58],
        "Yaman": [1.0, 1.12, 1.26, 1.5, 1.89],
        "Shanti Monotone": [1.0, 1.0, 1.0, 1.05, 1.0]
    }
    
    with st.spinner("Analyzing Chandas and Matras..."):
        weights = get_prosody_weights(verse)
        audio, fs = generate_priest_chant(weights, base_hz, scales[raga], pace)
        
        st.audio(audio, sample_rate=fs)
        
        # Visualizer
        st.markdown("#### Matra Breakdown (Laghu ᑌ / Guru —)")
        display_str = " ".join(["ᑌ" if w == "L" else "—" if w == "G" else " | " for w in weights])
        st.code(display_str, language="text")
        
        # Pitch Tracking Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=audio[1000:5000], line=dict(color='#FFD700')))
        fig.update_layout(title="Acoustic Resonance Pattern", template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

st.success("Analysis Complete: The engine has applied the 1:2 Matra ratio for rhythmic perfection.")
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

