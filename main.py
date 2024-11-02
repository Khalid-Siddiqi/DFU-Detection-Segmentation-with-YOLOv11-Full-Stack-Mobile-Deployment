from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from ultralytics import YOLO  # Ensure you have the YOLOv11 model loaded correctly

app = FastAPI()
model = YOLO("Model.pt")  # Update with your model path

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read the uploaded image
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection with YOLO
    results = model.predict(image)

    # Annotate the image (you can customize this part)
    annotated_image = image.copy()
    for result in results:
        for box in result.boxes:
            # Draw bounding boxes on the image
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Encode the annotated image to send back
    _, buffer = cv2.imencode('.jpg', annotated_image)
    return {"image": buffer.tobytes()}  # Send the image back as bytes