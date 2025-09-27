Weather Image Classification using EfficientNetB0
📌 Project Overview

This project is a part of the Mixed Reality & AI-Based Airport Navigation System.
It focuses on classifying sky images into different weather conditions using a deep learning model based on EfficientNetB0.
The goal is to support aviation safety by providing real-time weather awareness that can be integrated into airport navigation and decision-making systems.

🎯 Objectives

Build a CNN-based model to classify weather images (e.g., Clear, Cloudy, Foggy, Rainy, Snowy).

Reduce overfitting using data augmentation, dropout, and L2 regularization.

Fine-tune the EfficientNetB0 model for better accuracy.

Provide predictions that can be integrated with airport safety and MR visualization systems.

🛠 Tech Stack

Programming Language: Python

Frameworks & Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib

Model: EfficientNetB0 (pre-trained on ImageNet, fine-tuned on weather dataset)

Tools: Jupyter Notebook / Google Colab

📂 Dataset

The dataset contains sky images categorized into 11 weather classes (as provided in the original dataset).

Images were preprocessed (resizing, normalization) and augmented (rotation, flipping, zoom) to improve model generalization.

🔑 Methodology

Data Preprocessing

Resize images to fit EfficientNetB0 input (224x224).

Normalize pixel values.

Apply data augmentation to increase dataset diversity.

Model Development

Load EfficientNetB0 with pre-trained ImageNet weights.

Add custom classification head with Dense layers + Dropout.

Use L2 regularization to minimize overfitting.

Training

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

Evaluation

Evaluate on test set.

Generate confusion matrix & classification report.

🚀 Results

The model achieved high accuracy in classifying different weather types.

EfficientNetB0 proved effective due to its balance of accuracy and computational efficiency.

📌 Future Work

Integrate with real-time camera feeds from airports.

Deploy as a REST API for live weather classification.

Combine predictions with weather forecasting models (XGBoost + OpenWeather API) for more robust aviation safety solutions.
