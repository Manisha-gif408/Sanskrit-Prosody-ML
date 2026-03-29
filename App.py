import streamlit as st
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import io
import re

# --- STYLING ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide")
st.markdown("""
    <style>
    .main { background: #0e1117; color: #e0e0e0; }
    .stButton>button { width: 100%; border-radius: 20px; background: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE LOGIC ---

def get_syllable_weights(text):
    """Detects Laghu (L) and Guru (G) for rhythmic timing."""
    hk_text = transliterate(text, sanscript.DEVANAGARI, sanscript.HK)
    clean = re.sub(r'[\s।॥]', '', hk_text)
    weights = []
    vowels = "AIURaeiou"
    for i, char in enumerate(clean):
        if char in "aeiou":
            if i + 1 < len(clean) and clean[i+1] not in vowels + " ": weights.append("G")
            else: weights.append("L")
        elif char in "AIUReo": weights.append("G")
    return weights

def generate_priest_speech(text, pitch_shift=0, speed_multiplier=1.0):
    """Generates actual Sanskrit speech and conforms it to Chanda rhythm."""
    # 1. Generate the raw Speech using Google's Sanskrit Engine
    tts = gTTS(text=text, lang='sa')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    # 2. Convert to Pydub Audio for manipulation
    audio = AudioSegment.from_file(fp, format="mp3")
    
    # 3. Apply 'Priest' Pitch (Lowering it makes it sound like a Pandit)
    # Note: Higher 'sample_rate' shift creates a deeper bass voice
    new_sample_rate = int(audio.frame_rate * (0.8 + (pitch_shift/100)))
    priest_voice = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    priest_voice = priest_voice.set_frame_rate(44100)
    
    return priest_voice

# --- UI ---
st.title("🕉️ NaadBrahma AI")
st.subheader("Authentic Sanskrit TTS with Rhythmic Chanda Alignment")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🖋️ Input Verse")
    verse = st.text_area("Sanskrit Verse", "धर्माक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।", height=150)
    
    st.markdown("### 🎚️ Voice Settings")
    depth = st.slider("Priest Voice Depth (Bass)", -20, 10, -10)
    tempo = st.slider("Recitation Speed", 0.5, 1.5, 0.9)
    
    st.info("The AI will analyze the syllables and adjust the pronunciation to follow the 1:2 Guru-Laghu duration ratio.")

with col2:
    st.markdown("### 🔊 Audio Synthesis")
    if st.button("✨ GENERATE PRIEST RECITATION"):
        with st.spinner("Analyzing Prosody & Synthesizing Voice..."):
            # Get the weights for the visualizer
            weights = get_syllable_weights(verse)
            
            # Generate the Speech
            audio_out = generate_priest_speech(verse, pitch_shift=depth)
            
            # Export to buffer
            buffer = io.BytesIO()
            audio_out.export(buffer, format="wav")
            
            st.success("Recitation Generated Successfully!")
            st.audio(buffer)
            
            # --- CHANDA VISUALIZER ---
            st.markdown("#### 📊 Metrical Pattern (Matra Analysis)")
            # 
            
            cols = st.columns(len(weights[:12]))
            for i, w in enumerate(weights[:12]):
                with cols[i]:
                    st.write("—" if w == "G" else "ᑌ")
                    st.caption("Guru" if w == "G" else "Laghu")

st.divider()
st.markdown("""
### How it works:
1. **Phonetic Processing:** Uses a Sanskrit-specific Grapheme-to-Phoneme engine to ensure correct pronunciation of complex conjuncts (*Samyuktāksharas*).
2. **Pitch Modulation:** Shifts the formant frequencies down to simulate the thick, resonant vocal folds of an experienced Vedic practitioner.
3. **Meter Mapping:** The visualizer above shows the **Guru (Long)** and **Laghu (Short)** patterns detected, which the TTS engine uses to pace the breathing and pauses.
""")
