import re
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import convolve

def get_syllable_weights(text):
    guru_marks = "आईऊएऐओऔाीूेैोौंः"
    tokens = re.findall(r'[क-ह]्?[अ-औा-ौ]?[ंः]?', text)
    weights = []
    for i in range(len(tokens)):
        char = tokens[i]
        next_char = tokens[i+1] if i+1 < len(tokens) else ""
        if any(c in char for c in guru_marks) or "्" in next_char:
            weights.append(2)
        else:
            weights.append(1)
    return weights, tokens

def apply_priest_fx(audio_path, speed):
    y, sr = librosa.load(audio_path, sr=22050)
    
    # 1. Pitch Shift Down: Makes the voice deep and authoritative
    y_deep = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3.5)
    
    # 2. Temple Reverb: Simulates a stone hall using a slight echo
    # We create a simple impulse response for a 'hall' effect
    ir = np.zeros(int(sr * 0.5))
    ir[0] = 1.0
    ir[int(sr * 0.1)] = 0.4
    ir[int(sr * 0.2)] = 0.2
    y_temple = convolve(y_deep, ir, mode='full')[:len(y_deep)]
    
    # 3. Laya (Speed) Management
    y_final = librosa.effects.time_stretch(y_temple, rate=speed)
    
    output_path = "priest_recitation.wav"
    sf.write(output_path, y_final, sr)
    return output_path
