import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from gtts import gTTS
from transformers import pipeline
from scipy.signal import convolve
from difflib import SequenceMatcher
import re
import os

# Set Page Config
st.set_page_config(page_title="Chanda-Vox Priest AI", layout="wide", page_icon="🕉️")

# --- 1. PROSODY ENGINE ---
def get_syllable_metrics(text):
    guru_indicators = "आईऊएऐओऔाीूेैोौंः"
    # Tokenize Devanagari into Aksharas
    tokens = re.findall(r'[क-ह]्?[अ-औा-ौ]?[ंः]?', text)
    weights = []
    for i in range(len(tokens)):
        char = tokens[i]
        next_char = tokens[i+1] if i+1 < len(tokens) else ""
        if any(c in char for c in guru_indicators) or "्" in next_char:
            weights.append(2) # Guru (Long)
        else:
            weights.append(1) # Laghu (Short)
    return weights, tokens

# --- 2. PRIEST VOICE DSP (Intense & Human-like) ---
def apply_priest_fx(input_path, weights, laya_speed):
    y, sr = librosa.load(input_path, sr=22050)
    
    # A. Pitch Shift: Move to a deep, resonant Baritone (-4 semitones)
    y_deep = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4.0)
    
    # B. Rhythm: Adjust speed based on user Laya
    y_rhythmic = librosa.effects.time_stretch(y_deep, rate=laya_speed)

    # C. Temple Reverb: Convolution with a 'Stone Hall' Impulse Response
    # We create a synthetic IR to mimic a 100ms early reflection and 300ms tail
    ir = np.zeros(int(sr * 0.5))
    ir[0] = 1.0 # Direct path
    ir[int(sr * 0.05)] = 0.6 # Early reflection
    ir[int(sr * 0.15)] = 0.3 # Secondary reflection
    y_final = convolve(y_rhythmic, ir, mode='full')[:len(y_rhythmic)]
    
    output_fn = "priest_recitation.wav"
    sf.write(output_fn, y_final, sr)
    return output_fn

# --- 3. AI MODELS ---
@st.cache_resource
def load_stt_model():
    # Force Sanskrit language for accurate transcription back-check
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", 
                    generate_kwargs={"language": "sanskrit", "task": "transcribe"})

asr_pipe = load_stt_model()

# --- 4. USER INTERFACE ---
st.title("🕉️ Chanda-Vox: Melodic Priest AI")
st.markdown("### Ancient Sanskrit Prosody meets Modern Signal Processing")

# Sidebar
st.sidebar.header("🎛️ Chanting Controls")
laya = st.sidebar.slider("Laya (Recitation Speed)", 0.4, 1.2, 0.75, help="Lower is slower (Vedic style)")
intensity = st.sidebar.selectbox("Vocal Texture", ["Deep Vedic", "Resonant Temple", "Soft Meditation"])

# Input
input_verse = st.text_area("Enter Sanskrit Verse (Devanagari)", "शुक्लाम्वरधरं विष्णुं शशिवर्णं चतुर्भुजम्")

if st.button("Generate Intense Priest Recitation"):
    with st.spinner("Processing Melodic Components..."):
        # 1. Calculate Weights
        weights, tokens = get_syllable_metrics(input_verse)
        
        # 2. Base TTS Synthesis
        tts = gTTS(text=input_verse, lang='hi')
        tts.save("base.mp3")
        
        # 3. Apply Professional Audio FX
        final_audio = apply_priest_fx("base.mp3", weights, laya)
        
        # 4. Audio Playback
        st.subheader("🔊 Audio Output (Priest Mode)")
        st.audio(final_audio)

        # 5. Phonetic Validation (The Example you asked for)
        st.divider()
        st.subheader("🧐 Phonetic Accuracy Validation")
        
        # Load audio for Whisper STT
        audio_check, _ = librosa.load(final_audio, sr=16000)
        stt_result = asr_pipe(audio_check)["text"]
        
        col1, col2 = st.columns(2)
        col1.info(f"**Target Input:**\n{input_verse}")
        col2.success(f"**AI Transcribed (Back-conversion):**\n{stt_result}")
        
        # Calculate score
        acc_score = SequenceMatcher(None, input_verse, stt_result).ratio() * 100
        st.metric("Phonetic Fidelity Score", f"{acc_score:.1f}%")

        # --- 5. VISUALIZATIONS ---
        st.divider()
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.subheader("📊 Pitch Contour (Melody)")
            y_v, sr_v = librosa.load(final_audio)
            pitches, magnitudes = librosa.piptrack(y=y_v, sr=sr_v)
            pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if np.any(pitches[:, t] > 0)]
            
            fig_p = go.Figure(data=go.Scatter(y=pitch_values, line=dict(color='#FF4B4B', width=2)))
            fig_p.update_layout(xaxis_title="Time", yaxis_title="Hz", height=300)
            st.plotly_chart(fig_p, use_container_width=True)

        with v_col2:
            st.subheader("🌌 Power Spectrogram")
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y_v)), ref=np.max)
            fig_s, ax_s = plt.subplots(figsize=(10, 5))
            # Use 'magma' for intense color contrast
            img = librosa.display.specshow(D, sr=sr_v, x_axis='time', y_axis='log', ax=ax_s, cmap='magma')
            ax_s.set_title('Frequency Intensity Over Time')
            st.pyplot(fig_s)

        # Cleanup
        if os.path.exists("base.mp3"): os.remove("base.mp3")
