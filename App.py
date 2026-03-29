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
from scipy.signal import spectrogram

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="NaadBrahma AI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: #050505;
        color: #ffd700;
    }
    .chanda-box {
        border: 2px solid #ffd700;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        background: #111;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- CHANDA CLASSIFIER ----------------
def detect_chanda(text):
    try:
        hk = transliterate(text, sanscript.DEVANAGARI, sanscript.HK)
    except Exception:
        hk = text

    clean = re.sub(r'[\s।॥,;:!?\'"()\-\n\r]', '', hk)
    count = len(clean)

    if count == 24:
        return "Gāyatrī (24 Aksharas)"
    elif count == 32:
        return "Anuṣṭubh (32 Aksharas - Shloka)"
    elif count == 44:
        return "Triṣṭubh (44 Aksharas)"
    elif count == 48:
        return "Jagatī (48 Aksharas)"
    else:
        return f"Mixed / Muktaka ({count} Aksharas)"

# ---------------- AUDIO ENGINE ----------------
def generate_vedic_master(text, raga, rasa, speed):
    # TTS generation
    tts = gTTS(text=text, lang='hi')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)

    audio = AudioSegment.from_file(fp, format="mp3")

    # Raga / lineage effect
    depth = 0.73 if "Deep" in raga else 0.81

    # Rasa effect
    if "Veera" in rasa:
        audio = audio + 4
    elif "Karuna" in rasa:
        audio = audio.low_pass_filter(1200)
    elif "Shanti" in rasa:
        audio = audio.low_pass_filter(1800)

    # Convert to mono if stereo
    samples = np.array(audio.get_array_of_samples())

    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)

    # Keep sample rate valid
    new_sr = int(audio.frame_rate * depth * speed)
    new_sr = max(8000, min(new_sr, 48000))

    samples = np.clip(samples, -32768, 32767).astype(np.int16)

    return samples, new_sr

# ---------------- HELPERS ----------------
def create_wav_bytes(samples, sr_val):
    buf = io.BytesIO()
    wavfile.write(buf, sr_val, samples)
    buf.seek(0)
    return buf

def make_waveform_plot(samples, sr_val):
    max_points = 3000
    if len(samples) > max_points:
        step = max(1, len(samples) // max_points)
        samples_plot = samples[::step]
    else:
        samples_plot = samples

    time_axis = np.linspace(0, len(samples) / sr_val, num=len(samples_plot))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=samples_plot, mode='lines', name='Waveform'))
    fig.update_layout(
        title="Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=300
    )
    return fig

def make_spectrogram_plot(samples, sr_val):
    if len(samples) < 512:
        return None

    f, t, sxx = spectrogram(samples, sr_val, nperseg=min(512, len(samples)))
    sxx_db = 10 * np.log10(sxx + 1e-10)

    fig = go.Figure(
        data=go.Heatmap(
            x=t[:300],
            y=f[:150],
            z=sxx_db[:150, :300],
            colorscale="Hot"
        )
    )
    fig.update_layout(
        title="Acoustic Svara Footprint",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_dark",
        height=450
    )
    return fig

# ---------------- SESSION STATE ----------------
if "samples" not in st.session_state:
    st.session_state.samples = None
if "sr_val" not in st.session_state:
    st.session_state.sr_val = None

# ---------------- UI ----------------
st.title("🕉️ NaadBrahma AI: The Vedic Master")
st.caption("Chanda Detection | Svara Modulation | Human-Grade Synthesis")

c1, c2 = st.columns([1, 1.5])

with c1:
    verse = st.text_area(
        "Sanskrit Verse",
        "धर्माक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।",
        height=120
    )

    chanda_name = detect_chanda(verse)
    st.markdown(
        f"<div class='chanda-box'><b>Detected Meter:</b><br>"
        f"<span style='font-size:24px;'>{chanda_name}</span></div>",
        unsafe_allow_html=True
    )

    st.divider()

    raga = st.selectbox("Lineage", ["Deep Rigvedic", "Resonant Samavedic"])
    rasa = st.selectbox("Rasa", ["Shanti (Meditative)", "Karuna (Compassion)", "Veera (Commanding)"])
    laya = st.slider("Laya (Tempo)", 0.5, 1.5, 0.9, 0.05)

    if st.button("🔥 SYNTHESIZE VEDIC VOICE", use_container_width=True):
        try:
            samples, sr_val = generate_vedic_master(verse, raga, rasa, laya)
            st.session_state.samples = samples
            st.session_state.sr_val = sr_val
            st.success("Audio synthesized successfully.")
        except Exception as e:
            st.error(f"Error generating audio: {e}")

    if st.session_state.samples is not None:
        wav_buf = create_wav_bytes(st.session_state.samples, st.session_state.sr_val)
        st.audio(wav_buf, format="audio/wav")

        st.download_button(
            label="⬇ Download Chant WAV",
            data=wav_buf.getvalue(),
            file_name="naadbrahma_vedic_output.wav",
            mime="audio/wav",
            use_container_width=True
        )

with c2:
    if st.session_state.samples is not None:
        waveform_fig = make_waveform_plot(st.session_state.samples, st.session_state.sr_val)
        st.plotly_chart(waveform_fig, use_container_width=True)

        spec_fig = make_spectrogram_plot(st.session_state.samples, st.session_state.sr_val)
        if spec_fig is not None:
            st.plotly_chart(spec_fig, use_container_width=True)
        else:
            st.warning("Audio is too short to generate spectrogram.")

st.divider()
st.markdown(
    "<center>NaadBrahma AI: Merging Ancient Chanda-Shastra with Digital Signal Processing</center>",
    unsafe_allow_html=True
)
