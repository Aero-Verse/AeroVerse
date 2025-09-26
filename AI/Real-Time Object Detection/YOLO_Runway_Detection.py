from __future__ import annotations
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io
import json
import tempfile
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Load YOLO model
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
try:
    model = YOLO(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Could not load YOLO model from '{MODEL_PATH}'.\n↳ {exc}\n"
        "Ensure the .pt file exists and is compatible with your Ultralytics version."
    )

router = APIRouter()

def run_detection(img: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Run YOLO detection on BGR image and return (detections, annotated image)"""
    results = model(img)[0]
    detections: List[Dict[str, Any]] = []
    
    for box in results.boxes:
        cls_id = int(box.cls)
        detections.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": float(box.conf),
            "bbox": list(map(float, box.xyxy[0]))
        })
    
    return detections, results.plot()

@router.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    return_image: bool = True
):
    """
    Detect objects in image and return results with or without annotated image
    
    Parameters:
    - file: Uploaded image file
    - return_image: If True returns annotated image, otherwise returns only detection results
    """
    try:
        # Read image
        contents = await file.read()
        np_img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        if np_img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run detection
        detections, annotated_img = run_detection(np_img)

        if not return_image:
            return {"detections": detections}

        # Encode annotated image
        success, encoded_img = cv2.imencode(".jpg", annotated_img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode annotated image")

        # Return annotated image with results in header
        return StreamingResponse(
            io.BytesIO(encoded_img.tobytes()),
            media_type="image/jpeg",
            headers={
                "detections": json.dumps(detections),
                "Access-Control-Expose-Headers": "detections"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    download: bool = True
):
    """
    Detect objects in video and return annotated video
    
    Parameters:
    - file: Uploaded video file
    - download: If True returns as downloadable file, otherwise streams in browser
    """
    try:
        # Save temporary video
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(await file.read())
            video_path = Path(tmp_file.name)

        # Setup VideoCapture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            video_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Invalid video file (cannot open)")

        # Setup VideoWriter
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        output_path = Path(tempfile.mktemp(suffix=".mp4"))
        out = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        # Process video
        frame_count = 0
        start_time = time.time()
        sample_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections, annotated_frame = run_detection(frame)
            out.write(annotated_frame)
            
            if frame_count == 0:
                sample_detections = detections
                
            frame_count += 1

        # Cleanup resources
        cap.release()
        out.release()
        video_path.unlink(missing_ok=True)

        # Processing info
        processing_info = {
            "processing_time": time.time() - start_time,
            "frame_count": frame_count,
            "fps": frame_count / (time.time() - start_time) if frame_count else 0,
            "sample_detections": sample_detections,
        }

        # Prepare response
        disposition = "attachment" if download else "inline"
        headers = {
            "processing_info": json.dumps(processing_info),
            "Access-Control-Expose-Headers": "processing_info",
            "Content-Disposition": f"{disposition}; filename=annotated.mp4",
        }

        def generate_video():
            with open(output_path, "rb") as video_file:
                yield from video_file
            output_path.unlink(missing_ok=True)

        return StreamingResponse(
            generate_video(),
            media_type="video/mp4",
            headers=headers
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/classes")
async def get_classes():
    """Get list of classes the model can detect"""
    return {
        "classes": [
            {"id": class_id, "name": class_name}
            for class_id, class_name in model.names.items()
        ]
    }