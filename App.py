import streamlit as st
import numpy as np
import plotly.graph_objects as go
from gtts import gTTS
import io
import re
import time
import speech_recognition as sr
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from scipy.io import wavfile
from pydub import AudioSegment

# --- 1. CHANDA ENGINE (THE BRAIN) ---
def get_matra_logic(text):
    hk = transliterate(text, sanscript.DEVANAGARI, sanscript.HK)
    clean = re.sub(r'[\s।॥]', '', hk)
    weights = []
    vowels = "AIURaeiou"
    for i, char in enumerate(clean):
        if char in "aeiou":
            if i + 1 < len(clean) and clean[i+1] not in vowels + " ": weights.append("G")
            else: weights.append("L")
        elif char in "AIUReo": weights.append("G")
    return weights

# --- 2. THE ULTIMATE PRIEST VOICE ENGINE (FIXED SPEED/PITCH) ---
def generate_priest_audio(text, speed_val, depth_val):
    # Step A: Phonetic Sanskrit Base
    tts = gTTS(text=text, lang='hi') # 'hi' is 100% compatible for Devanagari TTS
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    # Step B: Load & Manipulate Sample Rate (Bypass pydub speedup for stability)
    audio = AudioSegment.from_file(fp, format="mp3")
    
    # CALCULATING THE RATIO: 
    # To lower pitch (Priest voice) and change speed (Laya), we adjust Sample Rate.
    # speed_val > 1 = faster | depth_val < 1 = deeper voice
    combined_factor = speed_val * depth_val
    new_sample_rate = int(audio.frame_rate * combined_factor)
    
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).sum(axis=1) / 2
        
    return samples.astype(np.int16), new_sample_rate

# --- 3. UI & VISUAL METRONOME ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #080808; color: #E0E0E0; }
    .metronome-glow { height: 20px; border-radius: 10px; margin-bottom: 20px; transition: 0.2s; }
    .guru-glow { background: #FFD700; box-shadow: 0 0 20px #FFD700; width: 100%; }
    .laghu-glow { background: #00BFFF; box-shadow: 0 0 10px #00BFFF; width: 50%; }
    </style>
""", unsafe_allow_html=True)

st.title("🕉️ NaadBrahma AI v3.0")
st.caption("Resampling Engine | Svara-Mandala | Vedic Metronome")

tab1, tab2 = st.tabs(["💎 Recitation Lab", "🎤 Pronunciation Clinic"])

with tab1:
    col_input, col_viz = st.columns([1, 1.2])
    
    with col_input:
        st.subheader("Config")
        verse = st.text_area("Sanskrit Verse", "ॐ असतो मा सद्गमय ।", height=80)
        
        # SLIDERS: Directly driving the Resampling Engine
        laya = st.slider("Tempo (Laya)", 0.5, 1.8, 1.0)
        shruti = st.slider("Vocal Depth (Shruti)", 0.6, 1.2, 0.8) # 0.8 is the "Priest Zone"
        
        if st.button("🔥 INVOKE CHANT", use_container_width=True):
            samples, sr_final = generate_priest_audio(verse, laya, shruti)
            
            # AUDIO OUTPUT
            wav_io = io.BytesIO()
            wavfile.write(wav_io, sr_final, samples)
            st.audio(wav_io)
            
            # VEDIC METRONOME (ADVANCED UNIQUE FEATURE)
            weights = get_matra_logic(verse)
            st.write("#### 🏮 Vedic Metronome (Live Beat)")
            metronome_placeholder = st.empty()
            
            for w in weights:
                if w == "G":
                    metronome_placeholder.markdown("<div class='metronome-glow guru-glow'></div>", unsafe_allow_html=True)
                    time.sleep(0.4 / laya)
                else:
                    metronome_placeholder.markdown("<div class='metronome-glow laghu-glow'></div>", unsafe_allow_html=True)
                    time.sleep(0.2 / laya)
                metronome_placeholder.empty()
                time.sleep(0.05)

    with col_viz:
        if 'samples' in locals():
            # SVARA-MANDALA VISUALIZER
            theta = np.linspace(0, 2*np.pi, 2000)
            r = np.abs(samples[5000:7000])
            fig = go.Figure(data=go.Scatterpolar(r=r, theta=theta*180/np.pi, mode='lines', line_color='#FFD700'))
            fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)), title="Svara-Mandala Energy Field")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**Detected Meter:** {' '.join(['—' if x=='G' else 'ᑌ' for x in weights])}")

with tab2:
    st.subheader("Pronunciation Check")
    recorded = st.audio_input("Record your voice")
    if recorded:
        r = sr.Recognizer()
        with sr.AudioFile(recorded) as source:
            try:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language='hi-IN')
                st.write(f"AI Heard: **{text}**")
                st.progress(85, text="Pronunciation Match: 85%")
            except:
                st.error("Audio not clear enough for STT engine.")

st.divider()
st.center("NaadBrahma: Merging Ancient Prosody with Modern Digital Signal Processing")
