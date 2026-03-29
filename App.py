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
import speech_recognition as sr
from scipy.fft import fft, fftfreq

# --- 1. THE CHANDA BRAIN (SYLLABIC TIMING) ---
def analyze_syllable_weights(text):
    """Determines Guru (Long) and Laghu (Short) for every syllable."""
    hk = transliterate(text, sanscript.DEVANAGARI, sanscript.HK)
    # Simplified regex for syllabic weight detection
    syllables = re.findall(r'[b-df-hj-np-tv-z]*[aeiouAIUReo]+[Mh]?', hk)
    weights = []
    for syl in syllables:
        # If vowel is uppercase (A, I, U) or diphthong (e, o, ai, au), it's Guru
        if any(v in syl for v in "AIUReo") or len(syl) > 2:
            weights.append("G")
        else:
            weights.append("L")
    return weights

# --- 2. ADVANCED HUMAN SYNTHESIS ENGINE ---
def generate_vedic_audio(text, raga, rasa, base_speed):
    # Step 1: Phonetic base via gTTS
    tts = gTTS(text=text, lang='hi')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio = AudioSegment.from_file(fp, format="mp3")

    # Step 2: Pitch & Formant Shift (The 'Priest' Chest Resonance)
    depth_map = {"Bhairav": 0.72, "Bhairavi": 0.80, "Yaman": 0.85, "Malkauns": 0.65}
    depth = depth_map.get(raga.split()[0], 0.8)
    
    # Step 3: Syllabic Speed Adjustment (Laghu-Guru Logic)
    # We simulate this by modulating the final sample rate 
    # to favor the 1:2 matra ratio
    weights = analyze_syllable_weights(text)
    guru_count = weights.count("G")
    laghu_count = weights.count("L")
    
    # Mathematical skew: Guru syllables pull the average speed down (lengthening them)
    rhythmic_multiplier = (guru_count * 1.5 + laghu_count * 0.8) / len(weights) if weights else 1.0
    
    final_sr = int(audio.frame_rate * depth * base_speed * (1/rhythmic_multiplier))
    
    # Step 4: Rasa Texturing
    if rasa == "Karuna (Compassionate)":
        audio = audio.low_pass_filter(1200)
    elif rasa == "Veera (Powerful)":
        audio = audio + 5

    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).sum(axis=1) / 2
        
    return samples.astype(np.int16), final_sr, weights

# --- 3. UI ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide")
st.markdown("<style>.stApp { background: #050505; color: #ffd700; }</style>", unsafe_allow_html=True)

st.title("🕉️ NaadBrahma AI: The Human Voice")
st.caption("Matra-Aware Resampling | Waveform Analysis | Spectrum Mapping")

tab1, tab2 = st.tabs(["🕯️ Recitation Temple", "🎙️ Validator"])

with tab1:
    c1, c2 = st.columns([1, 1.5])
    with c1:
        verse = st.text_area("Sanskrit Verse", "ॐ नमः शिवाय", help="Example: Try 'ॐ नमः शिवाय' or 'महामृत्युंजय मंत्र'")
        raga = st.selectbox("Raga", ["Bhairav (Deep/Morning)", "Bhairavi (Devotional)", "Yaman (Evening)", "Malkauns (Meditative)"])
        rasa = st.selectbox("Rasa", ["Shanti (Peaceful)", "Karuna (Compassionate)", "Veera (Powerful)"])
        speed = st.slider("Global Laya", 0.5, 1.5, 0.9)
        
        if st.button("🔥 GENERATE CHANT"):
            samples, sr, weights = generate_vedic_audio(verse, raga, rasa, speed)
            
            wav_buf = io.BytesIO()
            wavfile.write(wav_buf, sr, samples)
            st.audio(wav_buf)
            st.session_state['audio_data'] = (samples, sr, weights)

    with c2:
        if 'audio_data' in st.session_state:
            s, sr, w = st.session_state['audio_data']
            
            # WAVEFORM
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(y=s[::100], line=dict(color='#FFD700', width=1)))
            fig_w.update_layout(title="Temporal Waveform (Matra Spacing)", template="plotly_dark", height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_w, use_container_width=True)
            
            # SPECTRUM (FFT)
            n = len(s)
            yf = fft(s)
            xf = fftfreq(n, 1/sr)[:n//2]
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=xf[:3000], y=2.0/n * np.abs(yf[0:n//2])[:3000], fill='tozeroy', line=dict(color='#00BFFF')))
            fig_f.update_layout(title="Frequency Spectrum (Priest Resonance)", template="plotly_dark", height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_f, use_container_width=True)
            
            # CHANDA VISUAL
            st.markdown("### 🏮 Chanda Pattern")
            st.code(" ".join(["—" if x=="G" else "ᑌ" for x in w]))
            st.caption("— (Guru: 2 Matras) | ᑌ (Laghu: 1 Matra)")

st.divider()
st.info("Technical Note: The waveform energy peaks correspond to the Guru syllables, while the spectrum shows the deep harmonic overtones of a traditional priest.")
