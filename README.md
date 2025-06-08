# Textual-Entailment-Recognition-System
A Natural Language Processing (NLP) system that determines whether a given hypothesis logically follows from a premise. It uses semantic embeddings from transformer models and a classifier to output one of: entailment, contradiction, or neutral.

##  Project Overview

This system performs **Textual Entailment Recognition (TER)** by:
- Generating sentence embeddings using pre-trained **BERT**.
- Classifying semantic relationships with a **Multi-Layer Perceptron (MLP)** or fine-tuned **BERT** model.
- Presenting results through a clean and interactive **Streamlit GUI**.

---

##  Objectives
- Recognize semantic relationships between two sentences.
- Utilize **semantic embeddings** from BERT.
- Develop an efficient **entailment classifier**.
- Build an interactive **GUI** to visualize predictions and confidence scores.
- Target accuracy: **85%** on SNLI dataset.

---

##  Technologies Used
- Python
- PyTorch / HuggingFace Transformers
- Streamlit
- NumPy / Pandas / scikit-learn
- SNLI Dataset

---

##  Methodology

### 1. Preprocessing
- Load SNLI dataset (`snli_1.0_train.jsonl`)
- Filter valid examples (entailment, contradiction, neutral)
- Generate BERT embeddings for each (premise, hypothesis) pair

### 2. Model Architecture
- **MLP Classifier**: Input (768-d), Hidden (256-d), Output (3)
- Activation: ReLU, Loss: CrossEntropyLoss, Optimizer: Adam

### 3. Evaluation
- Training Accuracy: ~82.8%
- Validation Accuracy: ~70.91%
- F1 Scores: Entailment (0.7321), Contradiction (0.7198), Neutral (0.6517)

### 4. GUI (Streamlit)
- Input: Premise + Hypothesis
- Output: Predicted Label + Confidence Score
- Features:
  - Dropdown for test pairs
  - Real-time prediction
  - Bar chart visualization

---

##  How to Run

### Step 1: Clone the repo
```bash
git clone https://github.com/yourusername/textual-entailment-system.git
cd textual-entailment-system
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the app

```bash
streamlit run chatbot.py
```

# Contributing

Contributions are welcome! If you find issues or have improvements, please:

1. Fork the repository

2. Create a feature branch
  ```
   git checkout -b feature-name
```

3. Commit your changes
```
   git commit -m "Added new feature"
```

4. Push to GitHub
  ```
   git push origin feature-name
```
5. Submit a pull request 

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this project in accordance with the terms of the license.

## 1).Dataset

[Dataset](assets.Picture1.png)

