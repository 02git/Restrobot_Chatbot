# 🤖 RestroBot – Deep Learning Intent-Based Chatbot

RestroBot is an intent classification chatbot built using TensorFlow and deployed using Streamlit.  
The model is trained on a custom dataset and performs real-time intent prediction to generate contextual responses.

🔗 Live Demo: https://restrobotchatbot19238.streamlit.app/
---

## 🚀 Features

- Intent classification using Deep Learning (TensorFlow / Keras)
- NLP preprocessing using NLTK
- Bag-of-Words vectorization
- Trained neural network model saved and reused for inference
- Web-based chat interface built with Streamlit
- Deployment-ready (CPU compatible, no GPU required for inference)

---

## 🧠 Tech Stack

- Python 3.12
- TensorFlow 2.20
- NLTK
- NumPy
- Streamlit

---

## 🏗️ Project Structure
restrobot-chatbot/
│
├── app.py # Streamlit web application
├── src/
│ ├── train.py # Model training pipeline
│ └── predict.py # Standalone prediction logic
├── models/
│ ├── chatbot_model.keras
│ ├── words.pkl
│ └── classes.pkl
├── data/
│ └── Chatbot_Dataset.json
├── requirements.txt
└── README.md
---

## ⚙️ How It Works

1. User enters a message in the web UI.
2. Input is tokenized and lemmatized using NLTK.
3. A Bag-of-Words vector is created.
4. The trained neural network predicts intent probabilities.
5. The highest-confidence intent is selected.
6. A random response from that intent is returned.

---

## 🏋️ Model Architecture

- Fully Connected Neural Network
- Dense layers with ReLU activation
- Softmax output layer
- Categorical cross-entropy loss
- Optimized using Adam optimizer

The model is trained locally (GPU accelerated) and saved in `.keras` format for portable deployment.

---

## 💻 Local Setup

Clone the repository:
git clone https://github.com/02git/Restrobot_Chatbot.git

cd restrobot-chatbot
Install dependencies:
pip install -r requirements.txt
Run the application:
streamlit run app.py

---

## ☁️ Deployment

This project is deployed using Streamlit Cloud.  
The deployed version runs on CPU and loads the pre-trained model for inference.

---

## 📌 Key Learning Outcomes

- Building and training an NLP model using TensorFlow
- Handling text preprocessing and tokenization
- Saving and loading trained models
- Structuring ML projects for deployment
- Deploying ML applications to the cloud

---

## 📷 Screenshots

(Add screenshots of your Streamlit UI here)

---

## 📄 License

This project is for educational and portfolio purposes.

## 🔬 Future Improvements

- Replace Bag-of-Words with word embeddings
- Upgrade model to LSTM / Transformer architecture
- Add database-backed conversation logging
- Deploy with Docker
