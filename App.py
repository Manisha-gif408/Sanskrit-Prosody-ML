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

# --- 1. PROSODY & CHANDA ENGINE ---
def analyze_syllabic_rhythm(text):
    hk = transliterate(text, sanscript.DEVANAGARI, sanscript.HK)
    syllables = re.findall(r'[b-df-hj-np-tv-z]*[aeiouAIUReo]+[Mh]?', hk)
    weights = []
    for syl in syllables:
        if any(v in syl for v in "AIUReo") or len(syl) > 2:
            weights.append("G")
        else:
            weights.append("L")
    return weights, syllables

# --- 2. HUMAN VOCAL PHYSICS ---
def apply_priest_physics(samples, sampling_rate):
    n = len(samples)
    t = np.linspace(0, n/sampling_rate, n)
    # Jitter: Human vocal cord vibration instability
    jitter = 1 + 0.004 * np.sin(2 * np.pi * 5 * t) 
    # Shimmer: Amplitude instability
    shimmer = 1 + 0.015 * np.random.normal(0, 0.5, n)
    return (samples * jitter * shimmer).astype(np.int16)

def generate_human_priest_voice(text, raga, rasa, base_speed):
    tts = gTTS(text=text, lang='hi') 
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio = AudioSegment.from_file(fp, format="mp3")

    depth_map = {"Bhairav": 0.72, "Bhairavi": 0.81, "Yaman": 0.85, "Malkauns": 0.66}
    depth = depth_map.get(raga.split()[0], 0.8)
    
    weights, _ = analyze_syllabic_rhythm(text)
    guru_bias = (weights.count("G") * 1.6 + weights.count("L") * 0.9) / len(weights) if weights else 1.0
    
    final_sr = int(audio.frame_rate * depth * base_speed * (1/guru_bias))
    
    if rasa == "Karuna (Compassionate)":
        audio = audio.low_pass_filter(1250)
    elif rasa == "Veera (Powerful)":
        audio = audio + 5

    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).sum(axis=1) / 2
        
    final_samples = apply_priest_physics(samples, final_sr)
    return final_samples, final_sr, weights

# --- 3. UI ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide", page_icon="🕉️")
st.markdown("""
    <style>
    .stApp { background: #050505; color: #ffd700; }
    .metronome-glow { height: 15px; border-radius: 10px; margin-bottom: 20px; transition: 0.2s; }
    .guru-glow { background: #FFD700; box-shadow: 0 0 20px #FFD700; width: 100%; }
    .laghu-glow { background: #00BFFF; box-shadow: 0 0 10px #00BFFF; width: 50%; }
    .download-btn { background-color: #ffd700; color: black; border-radius: 5px; padding: 10px; text-decoration: none; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🕉️ NaadBrahma AI")
tab1, tab2 = st.tabs(["🕯️ Chanting Sanctuary", "🎙️ Validator"])

with tab1:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        verse = st.text_area("Sanskrit Verse", "ॐ नमः शिवाय", height=80)
        raga_sel = st.selectbox("Raga", ["Bhairav (Deep)", "Bhairavi (Devotional)", "Yaman (Smooth)"])
        rasa_sel = st.selectbox("Rasa", ["Shanti (Meditative)", "Karuna (Compassionate)", "Veera (Powerful)"])
        laya = st.slider("Tempo (Laya)", 0.5, 1.5, 0.95)
        
        if st.button("✨ INVOKE HUMAN VOICE", use_container_width=True):
            samples, sr_val, w_pat = generate_human_priest_voice(verse, raga_sel, rasa_sel, laya)
            
            # Export to Bytes for Audio Player and Download
            wav_io = io.BytesIO()
            wavfile.write(wav_io, sr_val, samples)
            st.audio(wav_io)
            
            # Metadata-enriched MP3 for Ringtone/Export
            mp3_io = io.BytesIO()
            audio_seg = AudioSegment(samples.tobytes(), frame_rate=sr_val, sample_width=samples.dtype.itemsize, channels=1)
            audio_seg.export(mp3_io, format="mp3", tags={'artist': 'NaadBrahma AI', 'title': f'Chant_{raga_sel}'})
            
            st.download_button(
                label="📥 Export as Ringtone (MP3)",
                data=mp3_io.getvalue(),
                file_name="Vedic_Chant_NaadBrahma.mp3",
                mime="audio/mp3",
                use_container_width=True
            )
            
            # LIVE VEDIC METRONOME
            st.write("#### 🏮 Vedic Metronome")
            m_placeholder = st.empty()
            for weight in w_pat:
                glow_class = 'guru-glow' if weight == 'G' else 'laghu-glow'
                m_placeholder.markdown(f"<div class='metronome-glow {glow_class}'></div>", unsafe_allow_html=True)
                time.sleep(0.4 if weight == 'G' else 0.2)
                m_placeholder.empty()
                time.sleep(0.05)
            st.session_state['data'] = (samples, sr_val, w_pat)

    with c2:
        if 'data' in st.session_state:
            s, sr, w = st.session_state['data']
            fig_w = go.Figure(data=go.Scatter(y=s[::100], line=dict(color='#FFD700', width=1)))
            fig_w.update_layout(title="Temporal Waveform (Matra Rhythm)", template="plotly_dark", height=200)
            st.plotly_chart(fig_w, use_container_width=True)
            
            n = len(s)
            yf = fft(s)
            xf = fftfreq(n, 1/sr)[:n//2]
            fig_f = go.Figure(data=go.Scatter(x=xf[:3500], y=2.0/n * np.abs(yf[0:n//2])[:3500], fill='tozeroy', line=dict(color='#00BFFF')))
            fig_f.update_layout(title="Frequency Spectrum (Priest Resonance)", template="plotly_dark", height=200)
            st.plotly_chart(fig_f, use_container_width=True)

with tab2:
    st.subheader("🎙️ Pronunciation Validator")
    user_voice = st.audio_input("Record your recitation")
    if user_voice:
        try:
            u_audio = AudioSegment.from_file(user_voice)
            buf = io.BytesIO()
            u_audio.export(buf, format="wav")
            buf.seek(0)
            
            recognizer = speech_recognition.Recognizer()
            with speech_recognition.AudioFile(buf) as source:
                audio_recorded = recognizer.record(source)
                text_out = recognizer.recognize_google(audio_recorded, language='hi-IN')
            
            st.markdown(f"### AI Detected: `{text_out}`")
            if any(word in verse for word in text_out.split()):
                st.success("✅ Matra Accuracy High!")
                st.balloons()
            else:
                st.warning("⚠️ Pronunciation Mismatch.")
        except Exception as e:
            st.error(f"Validator Analysis Error: {e}")
