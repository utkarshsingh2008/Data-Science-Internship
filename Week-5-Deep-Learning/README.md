
# Week 5 – Deep Learning Application in Data Science

## MNIST Handwritten Digit Classification

This project is part of my Data Science Internship Week 5 task. In this project, I explored deep learning fundamentals and built a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.

### What I Did

- Loaded and explored the MNIST dataset
- Analyzed class distribution
- Performed image and pixel-level analysis
- Preprocessed the image data
- Created training, validation and testing sets
- Used PyTorch DataLoaders for model training
- Designed and implemented a CNN architecture
- Trained the model using CrossEntropyLoss and Adam optimizer
- Analyzed training and validation accuracy
- Analyzed training and validation loss
- Evaluated the model on the test dataset
- Created a confusion matrix
- Generated a classification report
- Performed detailed error analysis
- Checked for overfitting
- Designed an improved CNN using Batch Normalization
- Compared the original CNN with the improved CNN
- Saved the trained improved model

### Dataset

The project uses the **MNIST Handwritten Digit Dataset**.

- Training samples: 60,000
- Testing samples: 10,000
- Number of classes: 10
- Image size: 28 × 28 pixels
- Image type: Grayscale

### Model Architecture

The CNN consists of:

- Convolutional Layer – 32 filters
- ReLU Activation
- Max Pooling
- Convolutional Layer – 64 filters
- ReLU Activation
- Max Pooling
- Flatten Layer
- Fully Connected Layer – 128 neurons
- ReLU Activation
- Dropout
- Output Layer – 10 classes

An improved version of the CNN was also implemented using **Batch Normalization** after the convolutional layers.

### Training Configuration

- Framework: PyTorch
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 64
- Train/Validation Split: 80/20

### Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report
- Error Analysis

Training and validation accuracy/loss curves were also analyzed to understand model performance and overfitting.

### Files

- `prog.py` – Complete Python implementation
- `mnist_improved_cnn_model.pth` – Saved trained improved CNN model
- `README.md` – Project documentation

### Tools & Libraries

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

### Key Learning

Through this project, I learned how a CNN can automatically learn image features and perform multi-class classification. I also understood the importance of validation data, performance metrics, overfitting analysis and techniques such as Batch Normalization for improving deep learning models.

### Note

This project was created for learning and internship purposes.
