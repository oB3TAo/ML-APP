from torchvision import transforms
from PIL import Image
import torch
from io import BytesIO

# Image preprocessing function
def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    return input_batch

# Prediction function
def get_prediction(input_batch, model):
    model.eval()
    # Check if a GPU is available and move the input and model to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities
