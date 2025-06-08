import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Set the device to CPU (adjust to GPU if available)
device = torch.device('cpu')

# Load pre-trained BERT model and tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    bert_model.to(device)
except Exception as e:
    st.error(f"Error loading BERT model: {e}")
    st.stop()

# Define the MLP Classifier
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load the trained model
model = MLPClassifier(input_dim=768, hidden_dim=256, output_dim=3)
try:
    model.load_state_dict(torch.load("mlp_classifier.pth", map_location=device))
    model.eval()
except FileNotFoundError:
    st.error("Model file 'mlp_classifier.pth' not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to generate embeddings for a sentence pair
def embed_sentence_pair(premise, hypothesis):
    try:
        encoded = tokenizer(premise, hypothesis, return_tensors='pt', padding=True, truncation=True, max_length=128)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = bert_model(**encoded)
        cls_embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
        return cls_embedding
    except Exception as e:
        st.error(f"Error in embedding generation: {e}")
        st.stop()

# Prediction function with accuracy
def predict_nli(premise, hypothesis):
    try:
        embedding = embed_sentence_pair(premise, hypothesis)
        X_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(X_tensor)
            pred = torch.argmax(output, dim=1).item()
            # Calculate confidence as percentage
            softmax_output = torch.nn.functional.softmax(output, dim=1)
            confidence = softmax_output.max().item() * 100  # Get the highest confidence percentage
        pred_label = ["contradiction", "entailment", "neutral"][pred]
        return pred_label, confidence
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Streamlit Chatbot Interface
st.title("NLI Chatbot")
st.write("Enter a premise and a hypothesis to check their relationship.")

# Input fields
premise = st.text_area("Premise", "A person is riding a horse.")
hypothesis = st.text_area("Hypothesis", "The person is outdoors.")

if st.button("Predict Relationship"):
    if premise and hypothesis:
        result, confidence = predict_nli(premise, hypothesis)
        if result:
            st.success(f"The relationship is: **{result}** with confidence: **{confidence:.2f}%**")
    else:
        st.error("Please enter both a premise and a hypothesis.")
