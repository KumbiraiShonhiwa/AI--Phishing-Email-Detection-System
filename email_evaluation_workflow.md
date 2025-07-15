sequenceDiagram
    participant User
    participant System as PhishingClassificationSystem
    participant Features as FeatureExtractor
    participant Model as SVMClassifier
    participant Interpreter as EmailInterpreter
    
    User->>System: Call evaluate_emails(email_data)
    activate System
    
    System->>Model: Load trained model
    activate Model
    Model-->>System: Return loaded model
    deactivate Model
    
    System->>Features: Load vectorizer
    activate Features
    Features-->>System: Return loaded vectorizer
    deactivate Features
    
    loop For each email
        System->>Features: Vectorize email text
        activate Features
        Features->>Features: Apply TF-IDF transformation
        Features-->>System: Return vectorized email
        deactivate Features
        
        System->>Model: Predict label
        activate Model
        Model->>Model: Apply SVM classification
        Model-->>System: Return prediction
        deactivate Model
        
        System->>Interpreter: Extract influential features
        activate Interpreter
        Interpreter->>Model: Request coefficients
        activate Model
        Model-->>Interpreter: Return feature coefficients
        deactivate Model
        Interpreter->>Features: Map features to words
        activate Features
        Features-->>Interpreter: Return feature to word mapping
        deactivate Features
        Interpreter->>Interpreter: Calculate feature importance
        Interpreter->>Interpreter: Sort by importance
        Interpreter-->>System: Return top contributing features
        deactivate Interpreter
        
        System->>System: Compile results
    end
    
    System->>System: Generate summary report
    System-->>User: Display results and explanations
    deactivate System