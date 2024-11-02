from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from torchvision import transforms
import torch
from PIL import Image
import io

app = FastAPI()

# Load the PyTorch model
model = torch.load("model.pt")
model.eval()  # Set the model to evaluation mode

# Define the image transformation (modify as per your model requirements)
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to the input size of your model
    transforms.ToTensor(),  # Convert image to tensor
])

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    # Read the image file from the uploaded file
    image = Image.open(io.BytesIO(await file.read()))

    # Apply the transformations
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Check if a GPU is available and if not, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)

    # Forward pass through the model
    with torch.no_grad():  # Disable gradient tracking
        output = model(input_batch)

    # Process the output as needed (e.g., converting to image, applying thresholds, etc.)
    # Here we assume you want to convert the output back to an image format
    # Replace this with your actual output processing
    processed_image = output.squeeze(0).cpu()  # Move the output back to CPU
    processed_image = transforms.ToPILImage()(processed_image)  # Convert to PIL Image

    # Save the processed image to an in-memory buffer
    img_byte_arr = io.BytesIO()
    processed_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # Return the processed image back to the React Native app
    return StreamingResponse(img_byte_arr, media_type="image/png")
