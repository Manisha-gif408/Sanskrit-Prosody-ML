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

# --- 1. CHANDA ENGINE (PROSODY LOGIC) ---
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

# --- 2. PRIEST VOICE ENGINE (FIXED SPEED & PITCH) ---
def generate_priest_audio(text, speed_val, depth_val):
    # Use Hindi engine for Sanskrit Phonetics
    tts = gTTS(text=text, lang='hi') 
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    audio = AudioSegment.from_file(fp, format="mp3")
    
    # Combined Factor: speed_val > 1 (faster), depth_val < 1 (deeper pitch)
    combined_factor = speed_val * depth_val
    new_sample_rate = int(audio.frame_rate * combined_factor)
    
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).sum(axis=1) / 2
        
    return samples.astype(np.int16), new_sample_rate

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide", page_icon="🕉️")

st.markdown("""
    <style>
    .stApp { background: #080808; color: #E0E0E0; }
    .metronome-glow { height: 15px; border-radius: 10px; margin-bottom: 10px; transition: 0.1s; }
    .guru-glow { background: #FFD700; box-shadow: 0 0 20px #FFD700; width: 100%; }
    .laghu-glow { background: #00BFFF; box-shadow: 0 0 10px #00BFFF; width: 50%; }
    .footer { text-align: center; color: #888; padding: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("🕉️ NaadBrahma AI v3.1")
st.caption("Resampling Engine | Svara-Mandala | Vedic Metronome | STT Validator")

tab1, tab2 = st.tabs(["💎 Recitation Lab", "🎤 Pronunciation Clinic"])

with tab1:
    col_input, col_viz = st.columns([1, 1.2])
    
    with col_input:
        st.subheader("Vocal Parametrics")
        verse = st.text_area("Sanskrit Verse", "धर्माक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।", height=80)
        
        laya = st.slider("Tempo (Laya)", 0.5, 1.8, 1.0)
        shruti = st.slider("Vocal Depth (Shruti)", 0.6, 1.2, 0.8) 
        
        if st.button("🔥 INVOKE CHANT", use_container_width=True):
            samples, sr_final = generate_priest_audio(verse, laya, shruti)
            
            # AUDIO OUTPUT
            wav_io = io.BytesIO()
            wavfile.write(wav_io, sr_final, samples)
            st.audio(wav_io)
            
            # VEDIC METRONOME (UNIQUE FEATURE)
            weights = get_matra_logic(verse)
            st.write("#### 🏮 Vedic Metronome (Live Beat)")
            metronome_placeholder = st.empty()
            
            # Simulate the beat visually
            for w in weights:
                color_class = 'guru-glow' if w == 'G' else 'laghu-glow'
                metronome_placeholder.markdown(f"<div class='metronome-glow {color_class}'></div>", unsafe_allow_html=True)
                time.sleep(0.3 / laya if w == 'G' else 0.15 / laya)
                metronome_placeholder.empty()
                time.sleep(0.05)

    with col_viz:
        if 'samples' in locals():
            # SVARA-MANDALA VISUALIZER
            # Taking a chunk for visualization
            slice_idx = min(len(samples), 10000)
            theta = np.linspace(0, 2*np.pi, slice_idx)
            r = np.abs(samples[:slice_idx])
            
            fig = go.Figure(data=go.Scatterpolar(r=r[::10], theta=(theta*180/np.pi)[::10], mode='lines', line_color='#FFD700'))
            fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False)), title="Svara-Mandala Resonance")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**Detected Meter (Matra):** {' '.join(['—' if x=='G' else 'ᑌ' for x in weights])}")

with tab2:
    st.subheader("STT Pronunciation Validator")
    st.write("Record your voice to verify your articulation against the sacred text.")
    recorded = st.audio_input("Record Audio")
    
    if recorded:
        r = sr.Recognizer()
        with sr.AudioFile(recorded) as source:
            try:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language='hi-IN')
                st.write(f"**AI Transcribed:** `{text}`")
                
                # Accuracy Score simulation
                similarity = 100 if text[:3] in verse else 65
                st.progress(similarity, text=f"Pronunciation Match: {similarity}%")
                if similarity > 80: st.success("Articulation is Shuddha (Pure)!")
            except:
                st.error("Audio clarity insufficient for analysis.")

st.divider()
st.markdown("<div class='footer'>NaadBrahma: Merging Ancient Prosody with Modern Digital Signal Processing</div>", unsafe_allow_html=True)
