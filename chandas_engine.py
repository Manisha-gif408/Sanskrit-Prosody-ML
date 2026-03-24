import re

# Database of Metres from your slides
METRE_RULES = {
    "Anuṣṭubh": {"pattern": "11111111", "yati": [4, 8]},
    "Mandākrāntā": {"pattern": "22221111122122122", "yati": [4, 10, 17]},
    "Śārdūlavikrīḍita": {"pattern": "2221121211122212212", "yati": [12, 19]}
}

def get_syllable_weights(text):
    # Rule: Long vowels, Anusvara (m), Visarga (h) are Guru (2)
    guru_marks = "आईऊएऐओऔाीूेैोौंः"
    # Syllable Tokenizer
    tokens = re.findall(r'[क-ह]्?[अ-औा-ौ]?[ंः]?', text)
    weights = []
    
    for i in range(len(tokens)):
        char = tokens[i]
        next_char = tokens[i+1] if i+1 < len(tokens) else ""
        # Rule: Short vowel + Conjunct = Guru
        if any(c in char for c in guru_marks) or "्" in next_char:
            weights.append(2)
        else:
            weights.append(1)
    return weights, tokens

def validate_recitation(transcribed_text, original_text):
    # Simple check to see if STT matches Input
    return transcribed_text.strip() == original_text.strip()