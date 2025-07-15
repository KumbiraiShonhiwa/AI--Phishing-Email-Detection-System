sequenceDiagram
    participant User
    participant System as PhishingClassificationSystem
    participant Data as DataLoader
    participant Preprocess as DataPreprocessor
    participant Features as FeatureExtractor
    participant Model as SVMClassifier
    participant Evaluator as ModelEvaluator

    User->>System: Initiate train_model()
    activate System
    
    System->>Data: Load email datasets
    activate Data
    Data-->>System: Return loaded datasets
    deactivate Data
    
    System->>Data: Explore dataset statistics
    activate Data
    Data-->>System: Return dataset statistics
    deactivate Data
    
    System->>Preprocess: Select features and target
    activate Preprocess
    Preprocess->>Preprocess: Handle missing values
    Preprocess->>Features: Combine email components
    activate Features
    Features-->>Preprocess: Return combined text
    deactivate Features
    Preprocess->>Preprocess: Split train/test data
    Preprocess-->>System: Return preprocessed data
    deactivate Preprocess
    
    System->>Features: Fit TF-IDF vectorizer
    activate Features
    Features->>Features: Learn vocabulary
    Features->>Features: Transform training data
    Features-->>System: Return vectorized data
    deactivate Features
    
    System->>Preprocess: Balance classes
    activate Preprocess
    Preprocess->>Preprocess: Apply RandomUnderSampler
    Preprocess-->>System: Return balanced data
    deactivate Preprocess
    
    System->>Model: Train SVM model
    activate Model
    Model->>Model: Fit model on resampled data
    Model-->>System: Return trained model
    deactivate Model
    
    System->>Model: Request predictions
    activate Model
    Model->>Model: Predict on test data
    Model-->>System: Return predictions
    deactivate Model
    
    System->>Evaluator: Evaluate model performance
    activate Evaluator
    Evaluator->>Evaluator: Calculate accuracy
    Evaluator->>Evaluator: Generate classification report
    Evaluator->>Evaluator: Create confusion matrix
    Evaluator->>Evaluator: Plot precision-recall curve
    Evaluator-->>System: Return evaluation metrics
    deactivate Evaluator
    
    System->>Model: Save trained model
    activate Model
    Model->>Model: Save model to disk
    Model-->>System: Confirm save successful
    deactivate Model
    
    System-->>User: Return training results
    deactivate System