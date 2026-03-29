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
import speech_recognition as speech_recognition 
from scipy.fft import fft, fftfreq

# --- 1. RHYTHM LOGIC ---
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

# --- 2. DYNAMIC SYNTHESIS ---
def generate_human_priest_audio(text, raga, rasa, base_speed):
    tts = gTTS(text=text, lang='hi') 
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio = AudioSegment.from_file(fp, format="mp3")

    depth_map = {
        "Bhairav (Deep/Morning)": 0.75,
        "Bhairavi (Devotional)": 0.82,
        "Yaman (Evening/Peace)": 0.85,
        "Malkauns (Meditative)": 0.70
    }
    
    if rasa == "Karuna (Compassionate)":
        audio = audio.low_pass_filter(1500) 
        audio = audio - 3 
    elif rasa == "Veera (Powerful)":
        audio = audio + 5 
    
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).sum(axis=1) / 2
        
    n = len(samples)
    depth_envelope = np.ones(n)
    
    # Dynamic logic for Om Namah Shivaya
    if any(k in text for k in ["ॐ", "नम", "शिव", "Om", "Namah", "Shiv"]):
        third = n // 3
        depth_envelope[:third] = 0.62          
        depth_envelope[third:2*third] = 0.72   
        depth_envelope[2*third:] = 0.80        
    else:
        depth_envelope = depth_envelope * depth_map.get(raga, 0.8)

    new_sr = int(audio.frame_rate * base_speed)
    t = np.linspace(0, n/new_sr, n)
    jitter = 1 + 0.005 * np.sin(2 * np.pi * 5 * t)
    
    final_samples = (samples * depth_envelope * jitter).astype(np.int16)
    return final_samples, new_sr

# --- 3. UI ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide")

st.title("🕉️ NaadBrahma: Reactive Human-Priest AI")

tab1, tab2 = st.tabs(["🕯️ Recitation Temple", "🎙️ Pronunciation Guru"])

with tab1:
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        verse = st.text_area("Input Shloka", "ॐ नमः शिवाय", height=100)
        raga_sel = st.selectbox("Select Raga", ["Bhairav (Deep/Morning)", "Bhairavi (Devotional)", "Yaman (Evening/Peace)", "Malkauns (Meditative)"])
        rasa_sel = st.selectbox("Select Rasa", ["Shanti (Peaceful)", "Karuna (Compassionate)", "Veera (Powerful)", "Bhakti (Devotional)"])
        master_speed = st.slider("Base Laya (Tempo)", 0.5, 1.5, 0.9)
        
        # Trigger synthesis automatically if parameters change OR button is clicked
        if st.button("🔥 SYNTHESIZE VEDIC VOICE", use_container_width=True):
            samples, sr_final = generate_human_priest_audio(verse, raga_sel, rasa_sel, master_speed)
            st.session_state['s'] = samples
            st.session_state['sr'] = sr_final
            st.session_state['v'] = verse

    with c2:
        # Check if we have data to display
        if 's' in st.session_state:
            s = st.session_state['s']
            sr = st.session_state['sr']
            
            # AUDIO PLAYER
            wav_io = io.BytesIO()
            wavfile.write(wav_io, sr, s)
            st.audio(wav_io)

            # 1. WAVEFORM
            fig_w = go.Figure(data=go.Scatter(y=s[::100], line=dict(color='#FFD700', width=1)))
            fig_w.update_layout(title="Reactive Waveform", template="plotly_dark", height=180)
            st.plotly_chart(fig_w, use_container_width=True)
            
            # 2. SPECTRUM
            n_fft = len(s)
            yf = fft(s)
            xf = fftfreq(n_fft, 1/sr)[:n_fft//2]
            fig_f = go.Figure(data=go.Scatter(x=xf[:3000], y=2.0/n_fft * np.abs(yf[0:n_fft//2])[:3000], fill='tozeroy', line=dict(color='#00BFFF')))
            fig_f.update_layout(title="Frequency Resonance", template="plotly_dark", height=180)
            st.plotly_chart(fig_f, use_container_width=True)

            # 3. MANDALA
            theta = np.linspace(0, 2*np.pi, 2000)
            r = np.abs(s[5000:7000]) if len(s) > 7000 else np.abs(s)
            fig_m = go.Figure(data=go.Scatterpolar(r=r, theta=theta*180/np.pi, mode='lines', line_color='#FFD700'))
            fig_m.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False)), height=250)
            st.plotly_chart(fig_m, use_container_width=True)

# (Validator tab code remains same)
