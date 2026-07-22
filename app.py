import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Disable TensorFlow warnings for a cleaner console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

from eye_detector import detect_eye

app = FastAPI(
    title="Eye Conjunctiva Anemia Detection API",
    description="Backend API for predicting anemia from conjunctiva images.",
    version="1.0.0"
)

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "modelo_anemia.keras")

print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("EXISTE:", os.path.exists(MODEL_PATH))

FRONTEND_DIR = BASE_DIR

# Global variable to store loaded model
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")
        return

    try:
        print(f"Loading Keras model from {MODEL_PATH}...")
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception:
        import traceback
        print(traceback.format_exc())
        raise

@app.get("/api/health")
def health_check():
    """Verify that the API is running and the model is loaded."""
    if model is None:
        return {"status": "error", "message": "Model not loaded. Check server logs."}
    return {"status": "ok", "message": "API running and model is loaded."}

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Validates the uploaded image, confirms a human eye is present, runs
    inference using the trained MobileNetV2 model, and returns class
    probabilities with quality warnings if applicable.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo de clasificación no está cargado en el servidor."
        )

    # 1. Validate File Format Extension
    filename_lower = file.filename.lower()
    if not filename_lower.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400,
            detail="Formato de archivo inválido. Solo se permiten imágenes JPG, JPEG o PNG."
        )

    # 2. Read file contents and validate file corruption
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Verify image integrity (catches corrupted/truncated images)

        # Re-open after verify() since verify() closes the file pointer
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="La imagen subida está corrupta o no se puede decodificar."
        )

    # 3. Subject validation: reject images that don't contain a human eye
    # before spending any time on the anemia classifier. Uses OpenCV's
    # pretrained Haar cascades (see eye_detector.py) rather than the
    # MobileNetV2 model, which was only ever trained to distinguish
    # anemia vs. no anemia and has no notion of "not an eye at all".
    eye_result = detect_eye(image)
    if not eye_result.eye_found:
        raise HTTPException(
            status_code=400,
            detail="La imagen no corresponde a un ojo humano. Capture nuevamente una imagen de la conjuntiva."
        )

    # 3b. The eye must be open: a closed eyelid hides the conjunctiva
    # entirely, so the model would be classifying skin/eyelash texture
    # instead of the actual tissue it was trained on.
    if not eye_result.eyes_open:
        raise HTTPException(
            status_code=400,
            detail="El ojo aparece cerrado en la imagen. Abra bien el ojo y capture nuevamente la conjuntiva."
        )

    # 3c. The model was trained specifically on the lower palpebral
    # conjunctiva. A photo showing only the upper eye/iris, or taken from
    # too far away (whole face), doesn't expose that tissue even though
    # the eye itself is open — reject those before they reach the model.
    if not eye_result.conjunctiva_visible:
        raise HTTPException(
            status_code=400,
            detail="No se observa la conjuntiva inferior en la imagen. Tire suavemente del párpado inferior hacia abajo y capture de cerca esa zona."
        )

    # 4. Quality checks and warnings
    warnings_list = []
    width, height = image.size

    # Check minimum resolution (e.g. 150x150 pixels)
    if width < 150 or height < 150:
        warnings_list.append(
            f"Resolución muy baja ({width}x{height}px). Se recomienda una imagen de al menos 224x224px para evitar pérdida de detalles en la conjuntiva."
        )
    elif width < 224 or height < 224:
        warnings_list.append(
            f"Resolución menor a la nativa del modelo ({width}x{height}px). El modelo MobileNetV2 trabaja a 224x224px."
        )

    # Check aspect ratio distortion
    aspect_ratio = width / height
    if aspect_ratio > 2.0 or aspect_ratio < 0.5:
        warnings_list.append(
            "La imagen tiene un formato muy estirado. Esto podría deformar el ojo al procesarlo."
        )

    # Check average brightness to detect very dark or overexposed images
    try:
        grayscale_image = image.convert("L")
        np_gray = np.array(grayscale_image)
        mean_brightness = np.mean(np_gray)
        if mean_brightness < 40:
            warnings_list.append(
                "La imagen parece demasiado oscura. Asegúrate de capturar la conjuntiva con buena iluminación."
            )
        elif mean_brightness > 220:
            warnings_list.append(
                "La imagen parece sobreexpuesta (demasiado brillo). Intenta evitar reflejos de luz directos."
            )
    except Exception:
        pass  # Non-blocking check

    # 5. Image Preprocessing (Replicating Training Pipeline)
    try:
        # Convert image to RGB (discards Alpha channels in PNG)
        rgb_image = image.convert("RGB")

        # Resize to 224x224 (MobileNetV2 standard input size) using bilinear filter
        resized_image = rgb_image.resize((224, 224), Image.Resampling.BILINEAR)

        # Convert image to numpy array in range [0, 255] (matching load_img/img_to_array)
        img_array = np.array(resized_image, dtype=np.float32)

        # Expand dimensions to create batch: (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        # 6. Run Inference
        # Running directly through model() is faster and thread-safe for single images than model.predict()
        predictions_tensor = model(img_batch, training=False)
        predictions = predictions_tensor.numpy()[0]  # Softmax output: [p_anemia, p_no_anemia]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el procesamiento o la inferencia: {str(e)}"
        )

    # 7. Build response object
    # Class order is alphabetical: [anemia, no_anemia]
    p_anemia = float(predictions[0])
    p_no_anemia = float(predictions[1])

    # Get index of highest probability
    predicted_class_idx = int(np.argmax(predictions))
    classes = ["ANEMIA", "NO ANEMIA"]
    predicted_label = classes[predicted_class_idx]
    confidence_percentage = float(predictions[predicted_class_idx]) * 100

    return {
        "result": predicted_label,
        "confidence": round(confidence_percentage, 1),
        "probabilities": {
            "Anemia": round(p_anemia * 100, 1),
            "No Anemia": round(p_no_anemia * 100, 1)
        },
        "quality_warnings": warnings_list
    }

# Serve static frontend files (must be mounted after API routes to avoid shadowing them)
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"⚠️ Warning: Frontend static directory not found at {FRONTEND_DIR}")
