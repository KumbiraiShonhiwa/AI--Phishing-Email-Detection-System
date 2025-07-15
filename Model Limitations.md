# Phishing Email Classification with SVM - Documentation

## Overview

This document provides comprehensive information about a machine learning model designed to classify emails as either phishing attempts (malicious) or legitimate messages. The model uses Support Vector Machine (SVM) classification with text feature extraction to identify potential phishing emails based on various email components.

## Table of Contents

1. [Purpose and Applications](https://claude.ai/chat/f4ccdea1-3083-4b56-ae10-72f3972b844c#purpose-and-applications)
2. [Model Architecture](https://claude.ai/chat/f4ccdea1-3083-4b56-ae10-72f3972b844c#model-architecture)
3. [Datasets](https://claude.ai/chat/f4ccdea1-3083-4b56-ae10-72f3972b844c#datasets)
4. [Features and Preprocessing](https://claude.ai/chat/f4ccdea1-3083-4b56-ae10-72f3972b844c#features-and-preprocessing)
5. [Training Process](https://claude.ai/chat/f4ccdea1-3083-4b56-ae10-72f3972b844c#training-process)
6. [Performance Metrics](https://claude.ai/chat/f4ccdea1-3083-4b56-ae10-72f3972b844c#performance-metrics)
7. [Model Interpretation](https://claude.ai/chat/f4ccdea1-3083-4b56-ae10-72f3972b844c#model-interpretation)
8. [Usage Guide](https://claude.ai/chat/f4ccdea1-3083-4b56-ae10-72f3972b844c#usage-guide)
9. [Limitations and Considerations](https://claude.ai/chat/f4ccdea1-3083-4b56-ae10-72f3972b844c#limitations-and-considerations)
10. [Future Improvements](https://claude.ai/chat/f4ccdea1-3083-4b56-ae10-72f3972b844c#future-improvements)

## Purpose and Applications

The primary purpose of this model is to automatically detect phishing emails, which are fraudulent messages designed to steal sensitive information from recipients. This model can be integrated into email security systems, personal email clients, or security training platforms to:

- Filter incoming emails for potential phishing attempts
- Provide explanations of why an email might be suspicious
- Support security teams in email threat analysis
- Create educational materials about phishing detection

## Model Architecture

The system employs a linear Support Vector Machine (SVM) classifier with a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer for text feature extraction. This architecture was chosen for its:

- Effectiveness with high-dimensional, sparse text data
- Ability to handle imbalanced datasets (with resampling)
- Interpretability of feature importance
- Good performance with limited training data

## Datasets

The model is trained on a combination of publicly available email datasets:

|Dataset|Description|Spam Count|Non-Spam Count|
|---|---|---|---|
|SpamAssasin|Collection of spam and legitimate emails|✓|✓|
|Nigerian_Fraud|Nigerian scam/fraud emails|✓|-|
|phishing_email|Dedicated phishing email collection|✓|-|
|CEAS_08|Conference on Email and Anti-Spam dataset|✓|✓|
|Enron|Business emails from Enron Corporation|✓|✓|
|Ling|Linguistics-focused email dataset|✓|✓|
|Nazario|Phishing corpus collected by Jose Nazario|✓|-|

The primary dataset used for model training is CEAS_08, which provides a balanced distribution of both phishing and legitimate emails.

## Features and Preprocessing

The model extracts and analyzes the following components from emails:

- **Sender**: Email address and display name of the sender
- **Subject**: Subject line text
- **Body**: Main email content
- **URLs**: Links contained within the email

Preprocessing steps include:

1. Combining the text features into a single text field
2. Converting to lowercase
3. Removing English stop words (common words like "the", "and", etc.)
4. Converting text to TF-IDF vectors (with max_features=1000)
5. Handling class imbalance using RandomUnderSampler

## Training Process

The training process follows these steps:

1. **Data Loading**: Multiple email datasets are loaded and explored
2. **Feature Extraction**: Email components are combined and vectorized using TF-IDF
3. **Train-Test Split**: Data is split into 80% training, 20% testing (stratified by label)
4. **Class Balancing**: The majority class is undersampled to address class imbalance
5. **Model Training**: Linear SVM is trained with C=1.0 (regularization parameter)
6. **Evaluation**: Model is evaluated on the test set using multiple metrics
7. **Model Saving**: The trained model is saved as 'svm_model.pkl'

## Performance Metrics

The model's performance is evaluated using:

- **Confusion Matrix**: Shows true positives, false positives, true negatives, and false negatives
- **Accuracy**: Overall correct classification rate
- **Precision**: Proportion of predicted phishing emails that are actually phishing
- **Recall**: Proportion of actual phishing emails correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Precision-Recall Curve**: Visualization of precision-recall tradeoff

Key metrics from the latest model evaluation:

- The best model achieves optimal precision and recall balance at the point identified in the precision-recall curve
- Detailed classification report is generated for both classes (phishing and non-phishing)

## Model Interpretation

The model provides interpretable results through:

1. **Feature Importance Analysis**: For each classification, the model identifies which words or phrases most strongly influenced the decision
2. **Direction of Influence**: Shows whether each feature pushed the classification toward phishing (+) or legitimate (-)
3. **Quantified Impact**: The magnitude of each feature's contribution is calculated

This interpretability is crucial for:

- Understanding why certain emails are flagged
- Identifying emerging phishing patterns
- Explaining decisions to users or security teams
- Continuous improvement of the model

## Usage Guide

### Prerequisites

- Python 3.6+
- Required libraries: numpy, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib, tabulate

### Installation

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn joblib tabulate
```

### Basic Usage

The model can be used in two main ways:

1. **Training a new model**:

```python
from phishing_classifier import train_model
train_model()
```

2. **Evaluating emails with an existing model**:

```python
from phishing_classifier import evaluate_emails
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load('svm_model.pkl')
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
vectorizer.fit(df_training['text_combined'])  # Fit on your training data

# Prepare email data
email_data = {
    'descriptor': ["Email subject or short description"],
    'email_text': ["Full text of the email including sender, subject, and body"],
    'true_labels': [0]  # 0 for legitimate, 1 for phishing (if known)
}

# Evaluate emails
evaluate_emails(email_data, vectorizer, model)
```

### Example Output

```
Descriptor: Unknown loan request
Predicted label: Phishing
True label: Phishing
Top contributing features:
  loan: +0.5864
  bank: +0.4215
  click: +0.3987
  link: +0.3721
  confirm: +0.3269
  details: +0.2845
  approved: +0.2311
  $10,000: +0.1982
  now: +0.1745
  your: +0.0962
```

## Limitations and Considerations

1. **Feature Limitations**:
    
    - Does not analyze email headers beyond sender
    - Limited processing of HTML structure
    - No image analysis capabilities
2. **Dataset Biases**:
     
    - Training data may not represent newest phishing techniques
    - Geographic and language biases in the training data
3. **Security Considerations**:
    
    - Should not be the only layer of email security
    - Regular updates required to maintain effectiveness
    - May be vulnerable to adversarial techniques
4. **Performance Trade-offs**:
    
    - Higher precision may lead to missed phishing attempts
    - Higher recall may increase false positives

## Future Improvements

1. **Enhanced Feature Engineering**:
    
    - HTML structure analysis
    - Header analysis (SPF, DKIM, DMARC)
    - Image-based feature extraction
2. **Advanced Model Architectures**:
    
    - Ensemble methods combining multiple classifiers
    - Deep learning approaches for sequential analysis
    - Transfer learning from large language models
3. **Active Learning Integration**:
    
    - System for incorporating user feedback
    - Continuous model updating
4. **Expanded Evaluation**:
    
    - Time-based evaluation to measure concept drift
    - Industry-specific benchmark tests
5. **Deployment Optimizations**:
    
    - Model compression for faster inference
    - API development for easy integration
    - Containerized deployment options