from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Disable GPU (safe for Mac)
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

app = Flask(__name__)

print("🚀 Loading model...")

# ✅ FINAL FIX: safer loading for older .h5 model
model = load_model(
    "water_model.h5",
    compile=False,
    safe_mode=False  # VERY IMPORTANT
)

print("✅ Model loaded successfully")


# 🔥 Feature extraction (same as training)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.max(mfcc, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1), np.max(delta, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1), np.max(delta2, axis=1),
    ])

    return features.reshape(1, -1)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filepath = "temp.wav"
        file.save(filepath)

        print("✅ Received file")

        features = extract_features(filepath)
        print("Feature shape:", features.shape)

        # ✅ Ensure correct shape
        if features.shape[1] != model.input_shape[1]:
            return jsonify({
                "error": f"Feature mismatch: expected {model.input_shape[1]}, got {features.shape[1]}"
            }), 500

        prediction = model.predict(features, verbose=0)

        classes = ["filling", "cleaning", "idle"]
        result = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            "activity": result,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Server starting...")
    app.run(host="0.0.0.0", port=5001)
