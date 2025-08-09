
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Load saved model (make sure saved_model.keras is present in the container)
model = tf.keras.models.load_model("saved_model.keras")

IMG_SIZE = (128, 128)

app = FastAPI(title="CNN Image Classifier")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return JSONResponse({
        "class_index": class_idx,
        "confidence": confidence
    })
