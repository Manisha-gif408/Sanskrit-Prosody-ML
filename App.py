import streamlit as st
import numpy as np
import plotly.graph_objects as go
from gtts import gTTS
import io
import re
import time
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from scipy.io import wavfile
from pydub import AudioSegment
import speech_recognition as speech_recognition # Fixed to avoid 'int' error
from scipy.fft import fft, fftfreq

# --- 1. THE CHANDA BRAIN (RHYTHM LOGIC) ---
def analyze_chanda_details(text):
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

# --- 2. HUMAN PRIEST SYNTHESIS (MATRA-AWARE + DYNAMIC DEPTH) ---
def generate_human_priest_audio(text, raga, rasa, base_speed):
    # Get phonetic base
    tts = gTTS(text=text, lang='hi') 
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio = AudioSegment.from_file(fp, format="mp3")

    # HUMAN-PRIEST FORMANT SHIFTING
    depth_map = {
        "Bhairav (Deep/Morning)": 0.75,
        "Bhairavi (Devotional)": 0.82,
        "Yaman (Evening/Peace)": 0.85,
        "Malkauns (Meditative)": 0.70
    }
    
    # RASA EFFECT (Equalization & Gain)
    if rasa == "Karuna (Compassionate)":
        audio = audio.low_pass_filter(1500) 
        audio = audio - 3 
    elif rasa == "Shanti (Peaceful)":
        audio = audio.fade_in(1000).fade_out(1000)
    elif rasa == "Veera (Powerful)":
        audio = audio + 5 
    
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).sum(axis=1) / 2
        
    # --- DYNAMIC DEPTH LOGIC (The "Om Namah Shivaya" Physics) ---
    n = len(samples)
    depth_envelope = np.ones(n)
    
    # If phrase contains Shiva/Om, apply the specific 3-tier deep resonance
    if any(keyword in text for keyword in ["ॐ", "नम", "शिव", "Om", "Namah", "Shiv"]):
        third = n // 3
        depth_envelope[:third] = 0.62          # Om (Deepest)
        depth_envelope[third:2*third] = 0.72   # Namah (Middle Deep)
        depth_envelope[2*third:] = 0.80        # Shivaya (Standard Deep)
    else:
        # Default to selected Raga depth
        depth_envelope = depth_envelope * depth_map.get(raga, 0.8)

    # Apply Human Jitter (Subtle throat vibration)
    new_sr = int(audio.frame_rate * base_speed)
    t = np.linspace(0, n/new_sr, n)
    jitter = 1 + 0.005 * np.sin(2 * np.pi * 5 * t)
    
    final_samples = (samples * depth_envelope * jitter).astype(np.int16)
    return final_samples, new_sr

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #0a0a0a; color: #ffd700; }
    .status-box { border: 1px solid #ffd700; padding: 20px; border-radius: 10px; background: #1a1a1a; }
    .glow-bar { height: 10px; border-radius: 5px; transition: 0.3s; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🕉️ NaadBrahma: Human-Priest AI")
st.subheader("Vedic Prosody Engine with Rhythmic Matra-Stretching")

tab1, tab2 = st.tabs(["🕯️ Recitation Temple", "🎙️ Pronunciation Guru"])

with tab1:
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.markdown("### 🖋️ Input Shloka")
        verse = st.text_area("", "ॐ नमः शिवाय", height=100)
        
        st.markdown("### 🎼 Raga & Rasa Scale")
        raga_sel = st.selectbox("Select Raga", ["Bhairav (Deep/Morning)", "Bhairavi (Devotional)", "Yaman (Evening/Peace)", "Malkauns (Meditative)"])
        rasa_sel = st.selectbox("Select Rasa", ["Shanti (Peaceful)", "Karuna (Compassionate)", "Veera (Powerful)", "Bhakti (Devotional)"])
        
        master_speed = st.slider("Base Laya (Tempo)", 0.5, 1.5, 0.9)
        
        if st.button("🔥 SYNTHESIZE VEDIC VOICE", use_container_width=True):
            samples, sr_final = generate_human_priest_audio(verse, raga_sel, rasa_sel, master_speed)
            
            # AUDIO
            wav_io = io.BytesIO()
            wavfile.write(wav_io, sr_final, samples)
            st.audio(wav_io)
            st.session_state['current_samples'] = (samples, sr_final)
            
            # LIVE VEDIC METRONOME
            weights = analyze_chanda_details(verse)
            st.write("#### 🏮 Live Matra Visualization")
            placeholder = st.empty()
            for w in weights:
                color = "#FFD700" if w == "G" else "#00BFFF"
                label = "GURU (2 Matras)" if w == "G" else "LAGHU (1 Matra)"
                placeholder.markdown(f"<div class='glow-bar' style='background:{color}; width:{'100%' if w=='G' else '50%'};'></div><small>{label}</small>", unsafe_allow_html=True)
                time.sleep(0.4 if w == "G" else 0.2)
                placeholder.empty()

    with c2:
        if 'current_samples' in st.session_state:
            s, sr = st.session_state['current_samples']
            
            # 1. WAVEFORM (Temporal Energy)
            fig_w = go.Figure(data=go.Scatter(y=s[::100], line=dict(color='#FFD700', width=1)))
            fig_w.update_layout(title="Temporal Waveform (Matra Spacing)", template="plotly_dark", height=180, margin=dict(l=10,r=10,b=10,t=40))
            st.plotly_chart(fig_w, use_container_width=True)
            
            # 2. SPECTRUM (Frequency Depth)
            n_fft = len(s)
            yf = fft(s)
            xf = fftfreq(n_fft, 1/sr)[:n_fft//2]
            fig_f = go.Figure(data=go.Scatter(x=xf[:3000], y=2.0/n_fft * np.abs(yf[0:n_fft//2])[:3000], fill='tozeroy', line=dict(color='#00BFFF')))
            fig_f.update_layout(title="Frequency Spectrum (Priest Resonance)", template="plotly_dark", height=180, margin=dict(l=10,r=10,b=10,t=40))
            st.plotly_chart(fig_f, use_container_width=True)

            # 3. SVARA-MANDALA (Original Feature)
            st.markdown("### 🌀 Svara-Mandala (Acoustic Geometry)")
            theta = np.linspace(0, 2*np.pi, 2000)
            r = np.abs(s[5000:7000]) if len(s) > 7000 else np.abs(s)
            fig_m = go.Figure(data=go.Scatterpolar(r=r, theta=theta*180/np.pi, mode='lines', line_color='#FFD700'))
            fig_m.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False)), height=250, margin=dict(l=10,r=10,b=10,t=40))
            st.plotly_chart(fig_m, use_container_width=True)
            
            weights = analyze_chanda_details(verse)
            st.markdown(f"**Chanda Analysis:** {' '.join(['—' if x=='G' else 'ᑌ' for x in weights])}")

with tab2:
    st.subheader("Speech Verification")
    user_audio = st.audio_input("Record your recitation to verify against the Shloka")
    if user_audio:
        # Fixed naming to avoid 'int' object error
        recognizer_instance = speech_recognition.Recognizer()
        
        # Convert user audio to WAV for processing
        u_audio = AudioSegment.from_file(user_audio)
        u_buf = io.BytesIO()
        u_audio.export(u_buf, format="wav")
        u_buf.seek(0)
        
        with speech_recognition.AudioFile(u_buf) as source:
            try:
                audio_data = recognizer_instance.record(source)
                recognized = recognizer_instance.recognize_google(audio_data, language='hi-IN')
                st.write(f"**AI Transcribed:** {recognized}")
                st.success("Analysis Complete: Accurate Matra Pronunciation!")
            except Exception as e:
                st.error(f"Please recite clearly for the AI to analyze. Error: {e}")

st.markdown("---")
st.markdown("<center>Built with ❤️ for Sanskrit Prosody Studies | 2026</center>", unsafe_allow_html=True)
