"""
Phoneme Dictionary & Clinical Target Inventory for Speech Sound Disorders (SSD).
Provides ARPAbet to IPA mappings, target sound classification, and clinical metadata.
"""

from typing import Dict, List, Optional, Tuple

# Mapping of ARPAbet phoneme symbols to International Phonetic Alphabet (IPA)
ARPABET_TO_IPA = {
    # Liquids & Glides
    "R": "ɹ",
    "L": "l",
    "W": "w",
    "Y": "j",
    # Fricatives & Sibilants
    "S": "s",
    "Z": "z",
    "SH": "ʃ",
    "ZH": "ʒ",
    "TH": "θ",
    "DH": "ð",
    "F": "f",
    "V": "v",
    "HH": "h",
    # Affricates
    "CH": "tʃ",
    "JH": "dʒ",
    # Stops / Plosives
    "P": "p",
    "B": "b",
    "T": "t",
    "D": "d",
    "K": "k",
    "G": "ɡ",
    # Nasals
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    # Vowels & Diphthongs
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
    "AX": "ə",
}

# Clinical Target Sound Inventory for Speech Sound Disorders
# Each target definition contains clinical metadata, typical articulatory deviations,
# expected acoustic hallmarks, and targeted therapeutic focus.
CLINICAL_TARGET_SOUNDS = {
    "R": {
        "ipa": "ɹ",
        "display_name": "/r/ (Rhotic Approximant)",
        "clinical_category": "Rhotacism / Liquid Articulation",
        "primary_error_type": "Derhoticization / Gliding (/r/ -> [w])",
        "acoustic_signatures": {
            "hallmark": "Dramatic lowering of 3rd formant (F3) close to F2 (F3 - F2 < 600 Hz)",
            "difficulty_marker": "Elevated F3 (> 2200 Hz), wide F3-F2 gap, characteristic of labiovelar glide [w]",
        },
        "description": "Difficulty elevating or bunching tongue body/blade while maintaining vocal tract constriction without lip rounding.",
        "sample_words": ["rabbit", "red", "run", "brown", "tree", "car", "bird", "star"]
    },
    "S": {
        "ipa": "s",
        "display_name": "/s/ (Voiceless Alveolar Fricative)",
        "clinical_category": "Sigmatism / Lisping",
        "primary_error_type": "Interdental Lisp ([θ]) or Lateral Lisp",
        "acoustic_signatures": {
            "hallmark": "High-frequency spectral centroid (> 5500 - 7500 Hz) with sharp spectral peak",
            "difficulty_marker": "Low spectral centroid (< 4500 Hz), diffuse mid-frequency turbulence, or excessive spectral bandwidth",
        },
        "description": "Difficulty directing narrow central airstream against alveolar ridge without tongue protrusion (frontal) or lateral air escape.",
        "sample_words": ["sun", "sister", "bus", "see", "fast", "castle", "soup", "house"]
    },
    "Z": {
        "ipa": "z",
        "display_name": "/z/ (Voiced Alveolar Fricative)",
        "clinical_category": "Sigmatism / Voiced Lisping",
        "primary_error_type": "Interdental / Lateral Lisp or Devoicing ([s] / [ð])",
        "acoustic_signatures": {
            "hallmark": "High spectral energy coupled with fundamental frequency (F0) voicing bar",
            "difficulty_marker": "Loss of high-frequency friction or mid-frequency energy dispersion",
        },
        "description": "Difficulty sustaining vocal cord vibration while producing narrow alveolar groove frication.",
        "sample_words": ["zoo", "zebra", "busy", "buzz", "nose", "music", "freeze"]
    },
    "SH": {
        "ipa": "ʃ",
        "display_name": "/ʃ/ ('sh' Postalveolar Fricative)",
        "clinical_category": "Depalatalization / Fronting",
        "primary_error_type": "Fronting to [s] or lateral distortion",
        "acoustic_signatures": {
            "hallmark": "Dominant spectral energy in the 3000 - 4500 Hz band with lower frequency boundary than /s/",
            "difficulty_marker": "Spectral peak shifted up towards > 6000 Hz (fronting) or irregular low turbulence",
        },
        "description": "Difficulty posturing tongue blade behind alveolar ridge with slight lip rounding.",
        "sample_words": ["ship", "shoe", "fish", "washing", "push", "ocean", "station"]
    },
    "CH": {
        "ipa": "tʃ",
        "display_name": "/tʃ/ ('ch' Voiceless Postalveolar Affricate)",
        "clinical_category": "Affricate Deaffrication / Fronting",
        "primary_error_type": "Deaffrication (/tʃ/ -> [ʃ] or [t]) or fronting to [ts]",
        "acoustic_signatures": {
            "hallmark": "Complete silent stop closure followed by rapid sharp fricative burst at 3.5 - 5.5 kHz",
            "difficulty_marker": "Absence of stop closure or lack of postalveolar noise burst",
        },
        "description": "Difficulty executing combined tongue-tip alveolar closure with postalveolar friction release.",
        "sample_words": ["chair", "church", "beach", "catch", "chicken", "match"]
    },
    "TH": {
        "ipa": "θ",
        "display_name": "/θ/ (Voiceless Dental Fricative)",
        "clinical_category": "Stopping / Labialization",
        "primary_error_type": "Stopping (/θ/ -> [t]) or Labialization (/θ/ -> [f])",
        "acoustic_signatures": {
            "hallmark": "Diffuse, flat low-intensity spectral envelope without strong isolated peaks",
            "difficulty_marker": "Sharp stop transient burst (if stopped to [t]) or formant transitions of [f]",
        },
        "description": "Difficulty maintaining tongue tip protrusion between incisors for light laminar frication.",
        "sample_words": ["think", "three", "thumb", "bath", "mouth", "teeth", "healthy"]
    },
    "K": {
        "ipa": "k",
        "display_name": "/k/ (Voiceless Velar Plosive)",
        "clinical_category": "Velar Fronting",
        "primary_error_type": "Fronting to Alveolar Stop [t] (e.g. 'cat' -> 'tat')",
        "acoustic_signatures": {
            "hallmark": "Compact mid-frequency burst peak (1500 - 2500 Hz) dependent on adjacent vowel",
            "difficulty_marker": "High-frequency burst (> 3500 Hz) characteristic of alveolar [t] release",
        },
        "description": "Difficulty elevating back of tongue dorsum against soft palate (velum).",
        "sample_words": ["cat", "cup", "cookie", "cake", "duck", "kite", "back"]
    },
    "L": {
        "ipa": "l",
        "display_name": "/l/ (Alveolar Lateral Approximant)",
        "clinical_category": "Gliding / Vocalization",
        "primary_error_type": "Gliding (/l/ -> [w] or [j]) or vowelization to [o]",
        "acoustic_signatures": {
            "hallmark": "Prominent low F1 (~350 Hz) and F2 (~1000 - 1500 Hz) with anti-formants/zeros around 2-3 kHz",
            "difficulty_marker": "Lack of lateral anti-resonance; high F2/F3 resembling [w] or [j]",
        },
        "description": "Difficulty maintaining tongue-tip contact at alveolar ridge while lowering tongue margins for lateral airflow.",
        "sample_words": ["lamp", "light", "yellow", "ball", "apple", "look", "blue"]
    }
}

# Curated high-frequency English pronunciation lexicon (Word -> ARPAbet phoneme sequence)
# Strips stress markers (e.g., AH0 -> AH) for acoustic alignment
COMMON_LEXICON: Dict[str, List[str]] = {
    # Exact forms used by the prototype prompt bank (ARPAbet; stress omitted).
    "speak": ["S", "P", "IY", "K"],
    "three": ["TH", "R", "IY"],
    "times": ["T", "AY", "M", "Z"],
    "thinks": ["TH", "IH", "NG", "K", "S"],
    "easy": ["IY", "Z", "IY"],
    "please": ["P", "L", "IY", "Z"],
    "close": ["K", "L", "OW", "Z"],
    "door": ["D", "AO", "R"],
    "boy": ["B", "OY"],
    "likes": ["L", "AY", "K", "S"],
    "fresh": ["F", "R", "EH", "SH"],
    "very": ["V", "EH", "R", "IY"],
    "fast": ["F", "AE", "S", "T"],
    "car": ["K", "AA", "R"],
    "call": ["K", "AO", "L"],
    "red": ["R", "EH", "D"],
    "she": ["SH", "IY"],
    "thinks": ["TH", "IH", "NG", "K", "S"],
    "this": ["DH", "IH", "S"],
    # Target R words
    "rabbit": ["R", "AE", "B", "AH", "T"],
    "red": ["R", "EH", "D"],
    "run": ["R", "AH", "N"],
    "running": ["R", "AH", "N", "IH", "NG"],
    "ran": ["R", "AE", "N"],
    "rain": ["R", "EY", "N"],
    "ring": ["R", "IH", "NG"],
    "road": ["R", "OW", "D"],
    "rope": ["R", "OW", "P"],
    "room": ["R", "UW", "M"],
    "radio": ["R", "EY", "D", "IY", "OW"],
    "river": ["R", "IH", "V", "ER"],
    "rose": ["R", "OW", "Z"],
    "rock": ["R", "AA", "K"],
    "rocket": ["R", "AA", "K", "AH", "T"],
    "brown": ["B", "R", "AW", "N"],
    "tree": ["T", "R", "IY"],
    "green": ["G", "R", "IY", "N"],
    "car": ["K", "AA", "R"],
    "bird": ["B", "ER", "D"],
    "star": ["S", "T", "AA", "R"],
    "door": ["D", "AO", "R"],
    "bear": ["B", "EH", "R"],
    "frog": ["F", "R", "AA", "G"],
    "train": ["T", "R", "EY", "N"],
    "grass": ["G", "R", "AE", "S"],
    
    # Target S & Z words
    "sun": ["S", "AH", "N"],
    "sunny": ["S", "AH", "N", "IY"],
    "sister": ["S", "IH", "S", "T", "ER"],
    "bus": ["B", "AH", "S"],
    "see": ["S", "IY"],
    "saw": ["S", "AO"],
    "soup": ["S", "UW", "P"],
    "soap": ["S", "OW", "P"],
    "sand": ["S", "AE", "N", "D"],
    "sandwich": ["S", "AE", "N", "D", "W", "IH", "CH"],
    "fast": ["F", "AE", "S", "T"],
    "castle": ["K", "AE", "S", "AH", "L"],
    "house": ["HH", "AW", "S"],
    "mouse": ["M", "AW", "S"],
    "snake": ["S", "N", "EY", "K"],
    "spoon": ["S", "P", "UW", "N"],
    "star": ["S", "T", "AA", "R"],
    "stop": ["S", "T", "AA", "P"],
    "smile": ["S", "M", "AY", "L"],
    "sky": ["S", "K", "AY"],
    "zoo": ["Z", "UW"],
    "zebra": ["Z", "IY", "B", "R", "AH"],
    "zero": ["Z", "IH", "R", "OW"],
    "buzz": ["B", "AH", "Z"],
    "busy": ["B", "IH", "Z", "IY"],
    "nose": ["N", "OW", "Z"],
    "music": ["M", "Y", "UW", "Z", "IH", "K"],
    "freeze": ["F", "R", "IY", "Z"],
    
    # Target SH & CH words
    "ship": ["SH", "IH", "P"],
    "shoe": ["SH", "UW"],
    "shoes": ["SH", "UW", "Z"],
    "fish": ["F", "IH", "SH"],
    "fishing": ["F", "IH", "SH", "IH", "NG"],
    "washing": ["W", "AA", "SH", "IH", "NG"],
    "wash": ["W", "AA", "SH"],
    "push": ["P", "UH", "SH"],
    "shine": ["SH", "AY", "N"],
    "sheep": ["SH", "IY", "P"],
    "shirt": ["SH", "ER", "T"],
    "shop": ["SH", "AA", "P"],
    "ocean": ["OW", "SH", "AH", "N"],
    "chair": ["CH", "EH", "R"],
    "church": ["CH", "ER", "CH"],
    "chicken": ["CH", "IH", "K", "AH", "N"],
    "beach": ["B", "IY", "CH"],
    "catch": ["K", "AE", "CH"],
    "match": ["M", "AE", "CH"],
    "cheese": ["CH", "IY", "Z"],
    "child": ["CH", "AY", "L", "D"],
    "children": ["CH", "IH", "L", "D", "R", "AH", "N"],
    
    # Target TH & DH words
    "think": ["TH", "IH", "NG", "K"],
    "three": ["TH", "R", "IY"],
    "thumb": ["TH", "AH", "M"],
    "bath": ["B", "AE", "TH"],
    "mouth": ["M", "AW", "TH"],
    "teeth": ["T", "IY", "TH"],
    "tooth": ["T", "UW", "TH"],
    "birthday": ["B", "ER", "TH", "D", "EY"],
    "the": ["DH", "AH"],
    "this": ["DH", "IH", "S"],
    "that": ["DH", "AE", "T"],
    "they": ["DH", "EY"],
    "there": ["DH", "EH", "R"],
    "mother": ["M", "AH", "DH", "ER"],
    "father": ["F", "AA", "DH", "ER"],
    "brother": ["B", "R", "AH", "DH", "ER"],
    
    # Target K & G words
    "cat": ["K", "AE", "T"],
    "cup": ["K", "AH", "P"],
    "cookie": ["K", "UH", "K", "IY"],
    "cake": ["K", "EY", "K"],
    "duck": ["D", "AH", "K"],
    "kite": ["K", "AY", "T"],
    "car": ["K", "AA", "R"],
    "king": ["K", "IH", "NG"],
    "key": ["K", "IY"],
    "game": ["G", "EY", "M"],
    "girl": ["G", "ER", "L"],
    "go": ["G", "OW"],
    "good": ["G", "UH", "D"],
    "big": ["B", "IH", "G"],
    "dog": ["D", "AA", "G"],
    
    # Target L words
    "lamp": ["L", "AE", "M", "P"],
    "light": ["L", "AY", "T"],
    "yellow": ["Y", "EH", "L", "OW"],
    "ball": ["B", "AO", "L"],
    "apple": ["AE", "P", "AH", "L"],
    "look": ["L", "UH", "K"],
    "blue": ["B", "L", "UW"],
    "black": ["B", "L", "AE", "K"],
    "fly": ["F", "L", "AY"],
    "play": ["P", "L", "EY"],
    
    # Common Sentence Function Words
    "a": ["AH"],
    "an": ["AE", "N"],
    "and": ["AE", "N", "D"],
    "is": ["IH", "Z"],
    "it": ["IH", "T"],
    "in": ["IH", "N"],
    "on": ["AA", "N"],
    "at": ["AE", "T"],
    "to": ["T", "UW"],
    "for": ["F", "AO", "R"],
    "of": ["AH", "V"],
    "with": ["W", "IH", "DH"],
    "he": ["HH", "IY"],
    "she": ["SH", "IY"],
    "we": ["W", "IY"],
    "i": ["AY"],
    "you": ["Y", "UW"],
    "my": ["M", "AY"],
    "was": ["W", "AA", "Z"],
    "jumped": ["JH", "AH", "M", "P", "T"],
    "over": ["OW", "V", "ER"],
    "fence": ["F", "EH", "N", "S"],
    "under": ["AH", "N", "D", "ER"],
    "little": ["L", "IH", "T", "AH", "L"],
    "big": ["B", "IH", "G"],
    "happy": ["HH", "AE", "P", "IY"],
    "sad": ["S", "AE", "D"],
    "like": ["L", "AY", "K"],
    "can": ["K", "AE", "N"],
    "said": ["S", "EH", "D"],
    "have": ["HH", "AE", "V"],
    "has": ["HH", "AE", "Z"],
    "had": ["HH", "AE", "D"],
}


def clean_phone(phone: str) -> str:
    """Strips numeric stress indicators from ARPAbet phones (e.g. 'R0' -> 'R')."""
    return "".join([c for c in phone.upper() if not c.isdigit()])


def get_word_phonemes(word: str) -> List[str]:
    """
    Returns the canonical ARPAbet phoneme sequence for a given word.
    Falls back to a standard rule-based grapheme-to-phoneme mapping if not in the dictionary.
    """
    cleaned = word.lower().strip(".,!?:;\"'()[]{}")
    if not cleaned:
        return []
    
    if cleaned in COMMON_LEXICON:
        return [clean_phone(p) for p in COMMON_LEXICON[cleaned]]
    
    # Rule-based fallback phoneme approximation
    return _simple_g2p(cleaned)


def _simple_g2p(word: str) -> List[str]:
    """Rule-based G2P fallback for out-of-vocabulary English words."""
    phones = []
    i = 0
    w = word.lower()
    n = len(w)
    
    while i < n:
        # Multi-letter digraphs
        if i + 1 < n and w[i:i+2] == "sh":
            phones.append("SH")
            i += 2
        elif i + 1 < n and w[i:i+2] == "ch":
            phones.append("CH")
            i += 2
        elif i + 1 < n and w[i:i+2] == "th":
            phones.append("TH")
            i += 2
        elif i + 1 < n and w[i:i+2] == "ee":
            phones.append("IY")
            i += 2
        elif i + 1 < n and w[i:i+2] == "oo":
            phones.append("UW")
            i += 2
        elif i + 1 < n and w[i:i+2] == "ea":
            phones.append("IY")
            i += 2
        elif i + 1 < n and w[i:i+2] == "ck":
            phones.append("K")
            i += 2
        elif i + 1 < n and w[i:i+2] == "ng":
            phones.append("NG")
            i += 2
        elif i + 1 < n and w[i:i+2] == "ph":
            phones.append("F")
            i += 2
        elif i + 1 < n and w[i:i+2] == "qu":
            phones.extend(["K", "W"])
            i += 2
        # Single characters
        else:
            c = w[i]
            mapping = {
                "a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH",
                "f": "F", "g": "G", "h": "HH", "i": "IH", "j": "JH",
                "k": "K", "l": "L", "m": "M", "n": "N", "o": "AA",
                "p": "P", "q": "K", "r": "R", "s": "S", "t": "T",
                "u": "AH", "v": "V", "w": "W", "x": "K", "y": "Y",
                "z": "Z"
            }
            if c in mapping:
                phones.append(mapping[c])
                if c == "x":
                    phones.append("S")
            i += 1
            
    return phones


def is_clinical_target(phone: str) -> bool:
    """Checks if a phone is in our designated Speech Sound Disorder target inventory."""
    return clean_phone(phone) in CLINICAL_TARGET_SOUNDS


def get_target_metadata(phone: str) -> Optional[dict]:
    """Retrieves clinical diagnosis metadata for a target phoneme."""
    return CLINICAL_TARGET_SOUNDS.get(clean_phone(phone))
