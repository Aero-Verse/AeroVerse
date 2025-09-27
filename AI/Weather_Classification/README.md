Introduction

This project implements weather image classification using EfficientNetB0.
It classifies sky images into multiple categories (e.g., Clear, Cloudy, Foggy, Rainy, Snowy) to support flight safety and airport operations.

Methodology

Dataset of sky images with 11 weather classes.

Preprocessing: image resizing (224x224), normalization, and data augmentation.

Model: EfficientNetB0 pre-trained on ImageNet, fine-tuned on weather dataset.

Techniques used to reduce overfitting: Dropout and L2 regularization.

Training with Adam optimizer, categorical crossentropy, and callbacks (EarlyStopping, ModelCheckpoint).

Results

Achieved high classification accuracy across all 11 classes.

Generated confusion matrix and classification report to validate performance.

Model is efficient and suitable for real-time weather classification in airport systems.

Benefits

Enhanced Safety: Provides instant recognition of adverse weather conditions.

Real-Time Awareness: Can be integrated with airport systems for immediate decision support.

Operational Efficiency: Assists pilots and air traffic controllers in planning during low-visibility situations.


big project link https://drive.google.com/file/d/1zjSgLe0BtEaM15L3sNbqL35z8pT-hMKL/view?usp=drive_link
