"""
Speech synthesis/voice cloning module. Subject to change.
"""

import gtts
import librosa
import numpy as np


fn = "out/synth.mp3"


def synthesize(_voice, text, lang):
    tts = gtts.gTTS(text, lang)
    tts.save(fn)
    x, sr = librosa.load(fn)
    x = x.astype(np.float64)
    return x, sr
