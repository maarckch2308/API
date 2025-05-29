from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# === CARGAR VARIABLES DE ENTORNO ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY no está definida en el archivo .env")

genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("models/gemini-1.5-flash")

# === CARGA DEL MODELO CNN ===
model_cnn = tf.keras.models.load_model('modelo_emociones.h5')
class_names_cnn = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Ajusta si es necesario
img_height, img_width = 48, 48

# === CARGA DEL MODELO DE TEXTO ===
MODEL_PATH = "./model"  # Ruta local del modelo robertuito
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model_text = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

labels_text = [
'alegría', 'ira', 'tristeza', 'asco',
'miedo', 'neutral', 'confianza', 'sorpresa', 'anticipación'
]


# === APP FASTAPI ===
app = FastAPI()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    text: str = Form(None)  # Texto opcional que puede venir junto con la imagen
):
    try:
        # --- Preprocesar imagen ---
        img = Image.open(file.file).convert("RGB")
        img = img.resize((img_width, img_height))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # --- Predicción CNN (imagen) ---
        prediction_img = model_cnn.predict(img_array)
        predicted_label_img = class_names_cnn[np.argmax(prediction_img)]
        confidence_img = float(np.max(prediction_img))

        # --- Predicción modelo texto (si se recibe texto) ---
        predicted_label_text = None
        confidence_text = None
        if text:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                logits = model_text(**inputs).logits
                probs = F.softmax(logits, dim=1).squeeze().tolist()
            
            predicted_idx_text = torch.argmax(torch.tensor(probs)).item()
            predicted_label_text = labels_text[predicted_idx_text]
            confidence_text = probs[predicted_idx_text]

        # --- Construir prompt para Gemini ---
        if predicted_label_text and text:
            prompt = (
                f"Lee el siguiente texto como si fuera una entrada de diario personal: '{text}'. "
                f"Además, la expresión facial de la persona refleja la emoción de '{predicted_label_img}'. "
                f"La persona expresa la emoción de '{predicted_label_text}' en sus palabras. "
                "Como un buen amigo, responde con un mensaje breve, personal y empático que ayude a sentirse comprendido y ofrezca un consejo sencillo de salud mental. No menciones datos ni porcentajes. Sé cálido, directo y cercano."
            )
        else:
            prompt = (
                f"Imagina que una persona está experimentando la emoción de '{predicted_label_img}' en este momento. "
                "Como un buen amigo, escribe un mensaje breve, personal y empático que le ayude a sentirse comprendido y le ofrezca un consejo sencillo de salud mental. No menciones datos ni porcentajes. Sé cálido, directo y cercano."
            )

        # --- Generar respuesta con Gemini ---
        response = model_gemini.generate_content(prompt)

        # --- Respuesta final ---
        return {
            "image_emotion": {
                "label": predicted_label_img,
                "confidence": confidence_img
            },
            "text_emotion": {
                "label": predicted_label_text,
                "confidence": confidence_text
            },
            "mental_health_message": response.text
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


##uvicorn main:app --reload
