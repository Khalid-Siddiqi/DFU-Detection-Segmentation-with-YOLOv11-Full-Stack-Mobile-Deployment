from fastapi.responses import StreamingResponse
import io
from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()
model = YOLO("Instance_Segementation_Model.pt")  # Update with your model path

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read and decode the uploaded image
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection with YOLO
    results = model.predict(image)

    # Annotate the image with segmentation masks and bounding boxes
    annotated_image = image.copy()
    alpha = 0.5  # Transparency factor for the mask overlay

    for result in results:
        for mask, box in zip(result.masks, result.boxes):  # Assuming masks and boxes are paired
            # Convert the mask tensor to a NumPy array and resize if needed
            mask_array = mask.data.cpu().numpy()
            if mask_array.shape[1:] != annotated_image.shape[:2]:
                mask_array = cv2.resize(mask_array[0], (annotated_image.shape[1], annotated_image.shape[0]))

            # Create a color overlay for the mask (blue with transparency)
            green_overlay = np.zeros_like(annotated_image, dtype=np.uint8)
            green_overlay[mask_array.astype(bool)] = (255, 0, 0)
            annotated_image = cv2.addWeighted(green_overlay, alpha, annotated_image, 1 - alpha, 0)

            # Draw bounding box around the mask
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box

    # Convert annotated image to JPEG and stream
    _, buffer = cv2.imencode('.jpg', annotated_image)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/jpeg")
