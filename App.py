import streamlit as st
import plotly.graph_objects as go
from gtts import gTTS
from transformers import pipeline
from chandas_engine import get_syllable_weights, apply_priest_fx
from difflib import SequenceMatcher
import librosa

# Load AI for Verification (Force Sanskrit)
@st.cache_resource
def load_stt():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", 
                    generate_kwargs={"language": "sanskrit", "task": "transcribe"})

asr_pipe = load_stt()

st.title("🕉️ Chanda-Vox: Priest-Voice AI & Validator")

# Input Section
input_text = st.text_area("Enter Sanskrit Verse", "शुक्लाम्वरधरं विष्णुं शशिवर्णं चतुर्भुजम्")
laya = st.sidebar.slider("Laya (Recitation Speed)", 0.5, 1.5, 0.8)

if st.button("Generate Melodic priest Recitation"):
    with st.spinner("Synthesizing Intense Priest Voice..."):
        # Generate raw AI voice
        tts = gTTS(text=input_text, lang='hi')
        tts.save("temp.mp3")
        
        # Apply Priest FX (Deep Bass + Temple Reverb)
        final_audio = apply_priest_fx("temp.mp3", laya)
        
        # --- AUDIO OUTPUT ---
        st.subheader("🔊 Melodic priest Recitation")
        st.audio(final_audio)

        # --- STT VALIDATION (The 'Perfect Translation' Check) ---
        st.divider()
        st.subheader("🧐 Speech-to-Text Validation")
        
        # Load audio for STT
        audio_data, _ = librosa.load(final_audio, sr=16000)
        stt_result = asr_pipe(audio_data)["text"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Original Input:**\n{input_text}")
        with col2:
            st.success(f"**AI Transcribed:**\n{stt_result}")
        
        # Accuracy Logic
        score = SequenceMatcher(None, input_text, stt_result).ratio() * 100
        st.metric("Phonetic Fidelity Score", f"{score:.2f}%")
        
        # --- VIRTUALIZATION ---
        st.divider()
        st.subheader("📈 Pitch Modulation (Melody Visualization)")
        y, sr = librosa.load(final_audio)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_trace = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if np.any(pitches[:, t] > 0)]
        
        fig = go.Figure(data=go.Scatter(y=pitch_trace, line=dict(color='#FF4B4B', width=3)))
        fig.update_layout(xaxis_title="Time", yaxis_title="Frequency (Hz) - priest Bass Range")
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("### Technical Framework")
st.sidebar.write("This POC uses **Phase Vocoding** for speed and **Impulse Response Convolution** for temple reverb, ensuring an unidentifiable AI presence.")
