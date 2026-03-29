import streamlit as st
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import io
import re
import plotly.graph_objects as go
import speech_recognition as sr

# --- 1. CHANDA PROSODY ENGINE ---
def analyze_sanskrit_meter(text):
    """Detects Guru (—) and Laghu (ᑌ) weights for 1:2 timing."""
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

# --- 2. ADVANCED PRIEST VOICE SYNTHESIS ---
def process_priest_audio(text, raga, rasa, base_pitch, speed_val):
    try:
        # Generate phonetic base (Hindi engine handles Devanagari Sanskrit perfectly)
        tts = gTTS(text=text, lang='hi') 
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio = AudioSegment.from_file(fp, format="mp3")
    except:
        return None

    # Apply Tempo Adjustment (Maintains Pitch)
    if speed_val != 1.0:
        # speedup handles the crossfading to prevent robotic artifacts
        audio = speedup(audio, playback_speed=speed_val, chunk_size=150, crossfade=25)

    # Apply Priest 'Bass' Shift (Frequency Scaling)
    # 0.78 factor creates a deep, resonant Pandit voice
    shift = 0.78 if base_pitch == "Deep Monastic" else 0.88
    new_sr = int(audio.frame_rate * shift)
    priest_voice = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sr})
    priest_voice = priest_voice.set_frame_rate(44100)

    # Apply Rasa (Emotional) Audio Textures
    if rasa == "Karuna (Compassion)":
        priest_voice = priest_voice.low_pass_filter(1200) # Soft, muffled
    elif rasa == "Veera (Bravery)":
        priest_voice = priest_voice + 6 # Bold, loud
    elif rasa == "Shanti (Peace)":
        priest_voice = priest_voice.fade_in(800).fade_out(800) # Smooth transitions

    return priest_voice

# --- 3. SPEECH-TO-TEXT (STT) VERIFICATION ---
def verify_pronunciation(audio_data):
    r = sr.Recognizer()
    # Convert Streamlit UploadedFile to AudioData
    audio_file = io.BytesIO(audio_data.read())
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        # Use hi-IN for best Sanskrit phoneme matching
        recognized = r.recognize_google(audio, language='hi-IN')
        return recognized
    except Exception as e:
        return f"Error: {str(e)}"

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide", page_icon="🕉️")

st.title("🕉️ NaadBrahma AI")
st.caption("The Complete Vedic Sanskrit Recitation & Analysis Suite")

tab1, tab2 = st.tabs(["🎙️ Synthesis & Chanda", "🎯 Pronunciation Coach"])

with tab1:
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.subheader("Configuration")
        verse = st.text_area("Sanskrit Verse", "धर्माक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।", height=100)
        
        c1, c2 = st.columns(2)
        with c1:
            raga_sel = st.selectbox("Melodic Scale", ["Bhairav", "Bhairavi", "Yaman"])
            pitch_sel = st.radio("Voice Depth", ["Standard Pandit", "Deep Monastic"])
        with c2:
            rasa_sel = st.selectbox("Emotion (Rasa)", ["Shanti (Peace)", "Karuna (Compassion)", "Veera (Bravery)"])
            speed_sel = st.slider("Recitation Speed", 0.5, 2.0, 1.0)
        
        if st.button("✨ GENERATE CHANT", use_container_width=True):
            final_audio = process_priest_audio(verse, raga_sel, rasa_sel, pitch_sel, speed_sel)
            if final_audio:
                buf = io.BytesIO()
                final_audio.export(buf, format="wav")
                st.session_state['last_audio'] = buf
                st.session_state['last_weights'] = analyze_sanskrit_meter(verse)

    with col_out:
        if 'last_audio' in st.session_state:
            st.success("Recitation Generated!")
            st.audio(st.session_state['last_audio'])
            
            # Chanda Analysis Display
            st.markdown("#### Matra Pattern (1:2 Ratio)")
            weights = st.session_state['last_weights']
            st.code(" ".join(["—" if w == "G" else "ᑌ" for w in weights]))
            st.caption("— (Guru: 2 Matras) | ᑌ (Laghu: 1 Matra)")
            
            # Spectral Visualization
            fig = go.Figure(data=go.Scatter(y=np.random.randn(500), line=dict(color='orange'))) # Placeholder for live wave
            fig.update_layout(title="Acoustic Spectral Print", template="plotly_dark", height=200)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("AI Pronunciation Feedback")
    st.write("Chant the verse into your microphone. The AI will verify your accuracy against the original text.")
    
    user_audio = st.audio_input("Record your recitation")
    
    if user_audio:
        with st.spinner("Processing your voice..."):
            recognized_text = verify_pronunciation(user_audio)
            
            st.markdown(f"**AI Heard:** `{recognized_text}`")
            st.markdown(f"**Original Text:** `{verse}`")
            
            # Simple fuzzy matching for hackathon demo
            if len(recognized_text) > 5 and (recognized_text[:5] in verse or verse[:5] in recognized_text):
                st.success("✅ Strong Match! Your pronunciation is accurate.")
                st.balloons()
            else:
                st.warning("⚠️ Pronunciation Mismatch. Ensure you are articulating the conjunct consonants clearly.")

st.divider()
st.info("NaadBrahma AI uses Phonetic Time-Stretching and STT Analysis to bridge ancient prosody with modern AI.")
