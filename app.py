from flask import Flask, render_template, request, redirect
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from commons import preprocess_image, get_prediction
from torchvision import models
import torch

app = Flask(__name__)

# Charger le modèle DenseNet121 préentraîné pour la classification d'images
image_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
image_model.eval()

# Charger les étiquettes des classes ImageNet
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Charger le modèle GPT-2 pour le chat
tokenizer = AutoTokenizer.from_pretrained("gpt2")
chat_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Créer une pipeline pour la génération de texte
generator = pipeline('text-generation', model=chat_model, tokenizer=tokenizer)

# Route pour la page d'accueil
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        feature = request.form.get('feature')

        if feature == 'sentiment':
            text = request.form.get('text')
            sentiment_pipeline = pipeline('sentiment-analysis')
            sentiment = sentiment_pipeline(text)[0]
            return render_template('result.html', predictions=[{
                "category": sentiment['label'],
                "probability": sentiment['score']
            }])

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

        elif feature == 'chat':
            return redirect('/chat')

    return render_template('index.html')


# Route pour le chat avec GPT-2
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    return render_template('chat.html')


# Route pour générer la réponse complète au message utilisateur
@app.route('/generate', methods=['POST'])
def generate():
    user_message = request.form.get('chat_message')

    # Use the pipeline to generate a response with controlled parameters
    response = generator(
        user_message,
        max_length=50,  # Limit the length to avoid long, repetitive answers
        num_return_sequences=1,
        temperature=0.7,  # Keep creativity but limit randomness
        top_k=50,  # Reduces the number of choices at each step
        top_p=0.9  # Limits options based on cumulative probability
    )

    # Decode the generated response
    decoded_response = response[0]['generated_text']

    # Simple post-processing to remove repetitive sentences
    # Split the response into sentences and remove duplicates
    sentences = decoded_response.split('. ')
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)

    # Rejoin unique sentences into a final response
    final_response = '. '.join(unique_sentences).strip()

    return final_response


if __name__ == '__main__':
    app.run(debug=True)
