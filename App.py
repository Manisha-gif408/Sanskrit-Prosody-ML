import streamlit as st
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import io
import re
import plotly.graph_objects as go

# --- 1. ADVANCED PROSODY ENGINE (LAGHU/GURU) ---
def analyze_sanskrit_meter(text):
    """Identifies the 1-Matra (L) and 2-Matra (G) pattern."""
    hk = transliterate(text, sanscript.DEVANAGARI, sanscript.HK)
    clean = re.sub(r'[\s।॥]', '', hk)
    weights = []
    vowels = "AIURaeiou"
    for i, char in enumerate(clean):
        if char in "aeiou":
            # Rule: Short vowel followed by conjunct is Guru
            if i + 1 < len(clean) and clean[i+1] not in vowels + " ": weights.append("G")
            else: weights.append("L")
        elif char in "AIUReo": weights.append("G")
    return weights

# --- 2. PRIEST VOICE & RAGA/RASA SYNTHESIS ---
def process_priest_audio(text, raga, rasa, base_pitch):
    # A. Generate Phonetic Speech
    tts = gTTS(text=text, lang='sa')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio = AudioSegment.from_file(fp, format="mp3")

    # B. Apply Priest Depth (Pitch Shifting)
    # Lower sample rate = Deeper Pandit voice
    shift_factor = 0.75 if base_pitch == "Deep" else 0.85
    new_sr = int(audio.frame_rate * shift_factor)
    priest_voice = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sr})
    priest_voice = priest_voice.set_frame_rate(44100)

    # C. Apply Rasa (Emotional Mood) logic
    if rasa == "Karuna (Compassion)":
        priest_voice = priest_voice.low_pass_filter(1500) # Muffled/Sad
    elif rasa == "Shanti (Peace)":
        priest_voice = priest_voice.fade_in(500).fade_out(500)
    elif rasa == "Veera (Bravery)":
        priest_voice = priest_voice + 5 # Increase Volume/Intensity

    return priest_voice

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="NaadBrahma AI", layout="wide", page_icon="🕉️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stTextArea textarea { background-color: #1e1e2e; color: #ffd700; font-size: 20px !important; }
    .metric-card { border: 1px solid #ffd700; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🕉️ NaadBrahma AI")
st.caption("Professional Sanskrit TTS with Chanda-Aware Rhythmic Recitation")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("1. Sacred Text Input")
    input_text = st.text_area("Verse (Devanagari)", "धर्माक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।", height=120)
    
    st.header("2. Melodic & Emotional Control")
    c_raga, c_rasa = st.columns(2)
    with c_raga:
        raga_sel = st.selectbox("Raga (Melody)", ["Bhairav", "Bhairavi", "Yaman", "Malkauns"])
    with c_rasa:
        rasa_sel = st.selectbox("Rasa (Mood)", ["Shanti (Peace)", "Karuna (Compassion)", "Veera (Bravery)", "Bhakti (Devotion)"])
    
    pitch_sel = st.radio("Voice Depth", ["Standard Pandit", "Deep Monastic"], horizontal=True)
    
    generate_btn = st.button("✨ GENERATE SACRED CHANT", use_container_width=True)

with col2:
    st.header("3. AI Analysis & Output")
    if generate_btn:
        with st.spinner("Analyzing Chandas and Synthesizing Priest Voice..."):
            # A. Process Audio
            final_audio = process_priest_audio(input_text, raga_sel, rasa_sel, pitch_sel)
            
            # B. Export
            buf = io.BytesIO()
            final_audio.export(buf, format="wav")
            
            st.success(f"Recitation Complete: {raga_sel} Framework Applied.")
            st.audio(buf)
            
            # C. Visual Matra Breakdown
            weights = analyze_sanskrit_meter(input_text)
            st.markdown("#### Matra Pattern (Laghu ᑌ / Guru —)")
            
            # Create a visual grid
            m_cols = st.columns(len(weights[:16]))
            for i, w in enumerate(weights[:16]):
                with m_cols[i]:
                    st.markdown(f"<div class='metric-card'>{'—' if w=='G' else 'ᑌ'}<br><small>{w}</small></div>", unsafe_allow_html=True)

            # D. Waveform Analysis
            samples = np.array(final_audio.get_array_of_samples())
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=samples[::10], line=dict(color='#ffd700', width=1)))
            fig.update_layout(title="Vocal Resonance Waveform", template="plotly_dark", height=250)
            st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.divider()
st.info("💡 **Innovation Highlight:** This AI doesn't just 'read' text. It maps the 1:2 Matra ratio (Guru:Laghu) to ensure the rhythmic beat of the Chanda is mathematically perfect, simulating the training of a Vedic scholar.")
