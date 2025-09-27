from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import tensorflow as tf
import logging
import os
from pathlib import Path

# Initialize router (only once, without tags)
router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
CLASS_NAMES = ["sunny", "cloudy", "rainy", "snowy", "foggy", "stormy"]
MODEL_PATH = "C:\\Users\\best-model.h5"

def verify_model_structure():
    global model
    try:
        if model is None:
            return False
        if len(model.output_shape) != 2:
            logger.error(f"Invalid model output shape: {model.output_shape}")
            return False
        if model.output_shape[1] != len(CLASS_NAMES):
            logger.error(f"Model expects {model.output_shape[1]} classes but we have {len(CLASS_NAMES)}")
            return False
        return True
    except Exception as e:
        logger.error(f"Model structure verification failed: {str(e)}")
        return False

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at: {MODEL_PATH}")
            return False
        file_size = os.path.getsize(MODEL_PATH)
        if file_size < 1024:
            logger.error(f"Model file is too small (possibly corrupted). Size: {file_size} bytes")
            return False
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        if not verify_model_structure():
            model = None
            return False
        logger.info("Model loaded and verified successfully")
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        model = None
        return False

@router.on_event("startup")
async def startup_event():
    logger.info("Starting server...")
    if not load_model():
        logger.error("Main model failed to load, creating test model...")
        create_test_model()

def create_test_model():
    global model
    try:
        logger.warning("Creating test model as fallback")
        test_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])
        test_model.compile(optimizer='adam', loss='categorical_crossentropy')
        model = test_model
        logger.info("Test model created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create test model: {str(e)}")
        return False

@router.get("/model-status")
async def get_model_status():
    status = {
        "loaded": model is not None,
        "model_file": MODEL_PATH,
        "file_exists": os.path.exists(MODEL_PATH),
        "file_size": f"{os.path.getsize(MODEL_PATH)/1024:.2f} KB" if os.path.exists(MODEL_PATH) else 0,
        "input_shape": model.input_shape if model else None,
        "output_shape": model.output_shape if model else None,
        "classes_count": len(CLASS_NAMES),
        "model_type": "REAL" if model and hasattr(model, '_is_graph_network') else "TEST" if model else None,
        "status": "READY" if model and verify_model_structure() else "NOT_READY"
    }
    return status

@router.post("/predict")
async def predict_weather(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please try again later."
        )
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB").resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        logger.info(f"Raw predictions: {predictions.tolist()}")
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions[0]))
        result = {
            "status": "success",
            "prediction": predicted_class,
            "confidence": confidence,
            "file_name": file.filename,
            "file_size": f"{len(contents)/1024:.2f} KB",
            "all_predictions": {
                name: float(pred) 
                for name, pred in zip(CLASS_NAMES, predictions[0])
            }
        }
        return result
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot identify image file. Please upload a valid image."
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

@router.get("/")
async def home():
    return {
        "message": "Weather Prediction API",
        "endpoints": {
            "/predict": "POST with image file",
            "/model-status": "GET model information"
        },
        "model_status": "READY" if model and verify_model_structure() else "NOT_READY"
    }