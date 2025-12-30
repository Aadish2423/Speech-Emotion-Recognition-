**Overview**

Speech Emotion Recognition (SER) is a crucial task in human–computer interaction that focuses on identifying emotional states from speech signals. This repository presents a deep learning–based SER system that classifies emotions from audio data using two distinct model architectures:

Multi-Layer Perceptron (MLP) implemented using TensorFlow

Transformer-based model implemented using PyTorch

The project emphasizes model comparison, robustness, and generalization by employing K-Fold Cross-Validation, ensuring reliable performance evaluation across multiple data splits.

----------------------------------------------------------------------------------------------------------------------------------
**Objectives**

Build a reliable system to classify emotions from speech

Compare traditional neural networks with attention-based architectures

Evaluate performance using K-Fold Cross-Validation

Demonstrate framework interoperability (TensorFlow vs PyTorch)

----------------------------------------------------------------------------------------------------------------------------------
**Models Implemented**

🔹 Multi-Layer Perceptron (MLP)

Framework: TensorFlow

Fully connected architecture

Operates on extracted audio features

Serves as a strong baseline for emotion classification

Lightweight and computationally efficient

🔹 Transformer Model

Framework: PyTorch

Uses self-attention mechanisms

Captures temporal and contextual dependencies in speech

Better suited for sequential audio representations

Demonstrates improved modeling of long-range dependencies

----------------------------------------------------------------------------------------------------------------------------------
**Methodology**

1.Audio Preprocessing

Audio signals are loaded and normalized

Noise and inconsistencies are handled

Relevant speech segments are extracted

2. Feature Extraction

Acoustic features (e.g., MFCCs and spectral features) are extracted

Features are transformed into numerical representations

Data is prepared for model input

3. Model Training

Separate pipelines for MLP and Transformer models

Training conducted using K-Fold Cross-Validation

Each fold acts once as validation and remaining as training data

4. Evaluation

Performance is evaluated across all folds

Metrics are averaged for unbiased assessment

----------------------------------------------------------------------------------------------------------------------------------
**Tech Stack**

Python

TensorFlow

PyTorch

NumPy

Librosa

Scikit-learn

Google Colab

----------------------------------------------------------------------------------------------------------------------------------
**Setup and Running**

🔹 Option 1: Google Colab (Recommended)

Upload SER.ipynb to Google Colab

Mount Google Drive (if required)

Install dependencies

Run cells sequentially

🔹 Option 2: Local Setup
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt

Run the notebook using Jupyter:

jupyter notebook SER.ipynb

----------------------------------------------------------------------------------------------------------------------------------
**Results & Observations**

MLP provides efficient and stable baseline performance

Transformer captures richer contextual information

K-Fold Cross-Validation improves reliability of evaluation

Trade-offs observed between computational cost and accuracy

----------------------------------------------------------------------------------------------------------------------------------
**Future Enhancements**

Fine-tuning transformer hyperparameters

Adding CNN + Transformer hybrid architectures

Applying data augmentation techniques

Real-time emotion recognition from live audio

Deployment as a web or desktop application

----------------------------------------------------------------------------------------------------------------------------------
**Author**

Aadish Bane

Interests: Artificial Intelligence, Machine Learning, Deep Learning
