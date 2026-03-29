import streamlit as st
import numpy as np
import plotly.graph_objects as go
from gtts import gTTS
import io
import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.fft import fft, fftfreq

# --- 1. CHANDA PROSODY ENGINE ---
def analyze_matras(text):
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

# --- 2. THE HUMAN-PRIEST SYNTHESIZER ---
def generate_priest_voice(text, raga, rasa, speed):
    tts = gTTS(text=text, lang='hi') 
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio = AudioSegment.from_file(fp, format="mp3")

    # Deepening the Formants for 'Priest' quality
    depth_factor = 0.78 if "Bhairav" in raga else 0.85
    if rasa == "Veera (Powerful)": audio = audio + 5
    elif rasa == "Karuna (Compassionate)": audio = audio.low_pass_filter(1200)

    # Applying the Master Speed & Pitch Resampling
    final_factor = depth_factor * speed
    new_sr = int(audio.frame_rate * final_factor)
    samples = np.array(audio.get_array_of_samples())
    
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).sum(axis=1) / 2
        
    return samples.astype(np.int16), new_sr

# --- 3. WAVEFORM VISUALIZATION ENGINE ---
def plot_waveforms(samples, sr):
    # Time Domain Waveform
    duration = len(samples) / sr
    time_axis = np.linspace(0, duration, len(samples))
    
    # Take a 1-second snapshot for clarity
    start = int(0.5 * sr)
    end = int(1.5 * sr)
    
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=time_axis[start:end], y=samples[start:end], line=dict(color='#FFD700', width=1)))
    fig_time.update_layout(title="Temporal Waveform (Matra Spacing)", template="plotly_dark", 
                          xaxis_title="Time (seconds)", yaxis_title="Amplitude", height=300)

    # Frequency Domain (FFT)
    N = len(samples)
    yf = fft(samples)
    xf = fftfreq(N, 1 / sr)[:N//2]
    
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=xf[:5000], y=2.0/N * np.abs(yf[0:N//2])[:5000], fill='tozeroy', line=dict(color='#00BFFF')))
    fig_freq.update_layout(title="Frequency Spectrum (Priest Resonance/Harmonics)", template="plotly_dark",
                          xaxis_title="Frequency (Hz)", yaxis_title="Power", height=300)
    
    return fig_time, fig_freq

# --- 4. UI ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide")
st.title("🕉️ NaadBrahma: Acoustic Analysis")

c1, c2 = st.columns([1, 1.5])

with c1:
    verse = st.text_area("Sanskrit Shloka", "ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं ।", height=100)
    raga = st.selectbox("Raga", ["Bhairav (Morning/Deep)", "Bhairavi (Evening/Devotional)"])
    rasa = st.selectbox("Rasa", ["Shanti (Peaceful)", "Karuna (Compassionate)", "Veera (Powerful)"])
    speed = st.slider("Chant Speed", 0.5, 1.5, 0.9)
    
    if st.button("✨ GENERATE & ANALYZE", use_container_width=True):
        samples, sr_val = generate_priest_voice(verse, raga, rasa, speed)
        
        # Audio Playback
        wav_io = io.BytesIO()
        wavfile.write(wav_io, sr_val, samples)
        st.audio(wav_io)
        
        # Matra Visual
        weights = analyze_matras(verse)
        st.code("Pattern: " + " ".join(["—" if w=="G" else "ᑌ" for w in weights]))

with c2:
    if 'samples' in locals():
        f_time, f_freq = plot_waveforms(samples, sr_val)
        st.plotly_chart(f_time, use_container_width=True)
        st.plotly_chart(f_freq, use_container_width=True)

st.info("💡 **Acoustic Note:** Notice the peaks in the Frequency Spectrum below 200Hz. This represents the 'Chest Resonance' of the priest voice, which is absent in standard high-pitched AI voices.")
