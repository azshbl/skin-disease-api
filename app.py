
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# تحميل النموذج
model = load_model("skin_disease_model.h5")

# الأصناف (نفس ترتيب التدريب)
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

@app.route("/predict", methods=["POST"])
def predict():
    print("====== REQUEST RECEIVED ======")
    if 'image' not in request.files:
        print("====== No Image Found ======")
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    print("====== Image Received ======")



    img_file = request.files["image"]
    img = load_img(img_file, target_size=(64, 64))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_name = labels[class_index]

    return jsonify({
        "prediction": class_name,
        "confidence": float(np.max(prediction))
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
