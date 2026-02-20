# --new code here ->
import requests

from flask import Flask, render_template, request
import sounddevice as sd
import numpy as np
import librosa
import joblib
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
import noisereduce as nr

app = Flask(__name__)


rf_model = joblib.load("model_v2.pkl")
label_encoder = joblib.load("label_encoder_v2.pkl")


def bandpass_filter(y, sr, lowcut=80, highcut=2000, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, y)


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    y_harmonic, _ = librosa.effects.hpss(y)
    y = y_harmonic

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = np.concatenate([
        np.mean(chroma, axis=1),
        np.mean(mfcc, axis=1)
    ])

    return features
#New test code starts from here
def get_ai_feedback(chord):
    prompt = f"""
    You are a keyboard tutor.
    The user played the chord: {chord}.
    
    Explain:
    - Whether it is major or minor
    - Notes in the chord
    - One simple practice tip
    
    Keep it short and beginner-friendly.
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]
#new test code end here

@app.route("/", methods=["GET", "POST"])

def index():
    
    predicted_chord = None
    ai_feedback = None # demo single line code 
    if request.method == "POST":

        #new code start here
        predicted_chord = None
        ai_feedback = f"You played {predicted_chord}. Try maintaining even pressure on all keys."
        #new code end here


        duration = 2        # seconds
        sample_rate = 22050

      
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1
        )
        sd.wait()

      
        energy = np.mean(audio ** 2)
        if energy < 0.001:
            predicted_chord = "Too much noise or too soft — play louder"
            return render_template(
                "index.html",
                predicted_chord=predicted_chord,#new code with small changes 
                ai_feedback=ai_feedback
            )


        write("mic_input.wav", sample_rate, audio)

        features = extract_features("mic_input.wav").reshape(1, -1)
        prediction = rf_model.predict(features)

        predicted_chord = label_encoder.inverse_transform(prediction)[0]
        ai_feedback = get_ai_feedback(predicted_chord) # test single line code start here


    # return render_template("index.html", predicted_chord=predicted_chord)
#New code start from here
    return render_template(
    "index.html",
    predicted_chord=predicted_chord,
    ai_feedback=ai_feedback
)
#End here


if __name__ == "__main__":
    app.run(debug=True)


## new code works fine