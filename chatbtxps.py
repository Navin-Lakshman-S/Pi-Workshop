import json
import random
import torch
import numpy as np
import gradio as gr

from nltk_utils import tokenize, bag_of_words  # Ensure these functions are correctly implemented
from model import NeuralNet  # Your neural network class

# Load the intents file (contains responses and tags)
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load the saved model data
FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Set device and instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def chat_response(user_input):
    """
    Processes the user input, runs the model, and returns an appropriate response.
    """
    # Tokenize the user input and convert to a bag-of-words vector
    sentence_tokens = tokenize(user_input)
    X = bag_of_words(sentence_tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    # Run the model to predict the tag
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    # Compute the probability for the predicted tag
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    # If confidence is high, select a response; otherwise, return a default message.
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                break
    else:
        response = "I do not understand..."
    return response

# Create a Gradio Interface
iface = gr.Interface(
    fn=chat_response,
    inputs="text",
    outputs="text",
    title="RPi Chatbot",
    description="Chat with the Raspberry Pi chatbot powered by PyTorch.",
    examples=[["Hello"], ["How are you?"], ["Tell me about the company."]]
)

if __name__ == "__main__":
    # Launch the Gradio app on all network interfaces (accessible on the same network)
    iface.launch(server_name="0.0.0.0", server_port=5002)
