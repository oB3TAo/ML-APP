import torch
from torchvision import models, transforms
from PIL import Image

# Load the image from your local file system
img_path = "sample_image.jpg"
img = Image.open(img_path)  # Open the image

# Load the pretrained DenseNet121 model
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.eval()

# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess the image
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Check if a GPU is available and move the input and model to GPU if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# Perform inference
with torch.no_grad():
    output = model(input_batch)

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Read the ImageNet class labels
with open("../imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Get the top 5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
