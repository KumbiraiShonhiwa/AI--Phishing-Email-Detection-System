classDiagram
    class PhishingClassificationSystem {
        +train_model()
        +evaluate_emails()
        +main()
    }

    class DataLoader {
        -load_datasets()
        -explore_datasets()
        -print_dataset_stats()
    }

    class FeatureExtractor {
        -TfidfVectorizer vectorizer
        +preprocess_data(data)
        +combine_features(sender, subject, body, urls)
        +vectorize_text(text)
        +fit_vectorizer(text_data)
    }

    class SVMClassifier {
        -SVC model
        +train(X_train, y_train)
        +predict(X_test)
        +extract_feature_importance(text, top_n)
        +save_model(filepath)
        +load_model(filepath)
    }

    class DataPreprocessor {
        +handle_missing_values(data)
        +split_train_test(X, y)
        +balance_classes(X, y)
    }

    class ModelEvaluator {
        +calculate_metrics(y_true, y_pred)
        +plot_confusion_matrix(y_true, y_pred)
        +plot_precision_recall_curve(y_true, y_scores)
        +generate_classification_report(y_true, y_pred)
    }

    class EmailInterpreter {
        +extract_influential_features(email, model, vectorizer)
        +explain_prediction(email, prediction, features)
        +visualize_feature_importance(features)
    }

    PhishingClassificationSystem --> DataLoader
    PhishingClassificationSystem --> FeatureExtractor
    PhishingClassificationSystem --> SVMClassifier
    PhishingClassificationSystem --> DataPreprocessor
    PhishingClassificationSystem --> ModelEvaluator
    PhishingClassificationSystem --> EmailInterpreter
    FeatureExtractor --> SVMClassifier
    ModelEvaluator --> SVMClassifier
    EmailInterpreter --> SVMClassifier
    EmailInterpreter --> FeatureExtractor