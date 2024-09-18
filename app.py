from flask import Flask, render_template, request, redirect
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from commons import preprocess_image, get_prediction
from torchvision import models
import torch

app = Flask(__name__)

# Load the pretrained DenseNet121 model directly from torchvision (for image classification)
image_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
image_model.eval()

# Load ImageNet class labels
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Load Mistral-7B model for chat
tokenizer = AutoTokenizer.from_pretrained("mistral-community/Mistral-7B-v0.2")
chat_model = AutoModelForCausalLM.from_pretrained("mistral-community/Mistral-7B-v0.2")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the selected feature from the dropdown
        feature = request.form.get('feature')

        # If feature is sentiment analysis
        if feature == 'sentiment':
            text = request.form.get('text')
            sentiment_pipeline = pipeline('sentiment-analysis')
            sentiment = sentiment_pipeline(text)[0]
            return render_template('result.html', predictions=[{
                "category": sentiment['label'],
                "probability": sentiment['score']
            }])

        # If feature is image classification
        elif feature == 'image':
            file = request.files.get('file')
            if not file:
                return redirect(request.url)
            img_bytes = file.read()
            input_batch = preprocess_image(img_bytes)
            probabilities = get_prediction(input_batch, image_model)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            top_predictions = [
                {"category": categories[top5_catid[i]], "probability": top5_prob[i].item()}
                for i in range(top5_prob.size(0))
            ]
            return render_template('result.html', predictions=top_predictions)

        # If feature is chat
        elif feature == 'chat':
            user_message = request.form.get('chat_message')
            inputs = tokenizer(user_message, return_tensors="pt")
            response = chat_model.generate(**inputs, max_length=100, num_return_sequences=1)
            decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)
            return render_template('result.html', chat_response=decoded_response)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
