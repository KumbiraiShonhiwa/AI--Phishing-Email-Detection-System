import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def process_text_directory(directory_path):
    """
    Process all text files in a directory and create a DataFrame with
    descriptors, email text, and true labels based on the filename.
    
    Parameters:
        directory_path (str): Path to directory containing text files.
        
    Returns:
        pd.DataFrame: DataFrame with 'descriptor', 'email_text', and 'true_labels' columns.
    """
    phishing_files = ['Test1.txt', 'Test2.txt', 'Test5.txt', 'Test6.txt' ,'Test8.txt', 'Test9.txt']
    
    data = {
        'descriptor': [],
        'email_text': [],
        'true_labels': []
    }
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist.")
    
    # Process each text file in the directory
    for filename in sorted(os.listdir(directory_path)):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            
            # Read the content of the text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    email_text = f.read()
                
                # Determine true label based on filename
                true_label = 1 if filename in phishing_files else 0
                
                # Add to data dictionary
                data['descriptor'].append(filename.replace('.txt', ''))
                data['email_text'].append(email_text)
                data['true_labels'].append(true_label)
                
                print(f"Processed {filename} - True label: {'Phishing' if true_label == 1 else 'Non-phishing'}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Convert to DataFrame
    return pd.DataFrame(data)

def evaluate_emails(email_directory=None, email_data=None, vectorizer=None, model=None, top_n=10):
    """
    Evaluates emails using a trained SVM model and TF-IDF vectorizer with explanation.

    Parameters:
        email_directory (str): Path to directory containing text files (optional).
        email_data (pd.DataFrame or dict): Contains 'descriptor', 'email_text', and 'true_labels' (optional).
        vectorizer (TfidfVectorizer): Pretrained TF-IDF vectorizer.
        model (sklearn classifier): Trained linear SVM classifier.
        top_n (int): Number of top features to show in interpretation.

    Returns:
        None
    """
    # Check if we need to process directory
    if email_directory:
        df = process_text_directory(email_directory)
    elif isinstance(email_data, dict):
        df = pd.DataFrame(email_data)
    elif isinstance(email_data, pd.DataFrame):
        df = email_data.copy()
    else:
        raise ValueError("Either email_directory or email_data must be provided.")
    
    # Validate that vectorizer and model are provided
    if vectorizer is None or model is None:
        raise ValueError("Both vectorizer and model must be provided.")

    descriptors = []
    predictions = []
    true_labels = []
    email_texts = []

    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_.toarray()[0]  # only for binary classification

    for _, row in df.iterrows():
        descriptor = row['descriptor']
        email_text = row['email_text']
        true_label = row['true_labels']

        X_email = vectorizer.transform([email_text])
        predicted_label = model.predict(X_email)[0]

        descriptors.append(descriptor)
        predictions.append(predicted_label)
        true_labels.append(true_label)
        email_texts.append(email_text)

        # Interpretation: extract non-zero TF-IDF features
        feature_index = X_email.nonzero()[1]
        tfidf_scores = X_email.data
        word_importance = [(feature_names[i], coef[i] * tfidf_scores[j]) 
                           for j, i in enumerate(feature_index)]

        word_importance_sorted = sorted(word_importance, key=lambda x: abs(x[1]), reverse=True)[:top_n]

        print(f"\nDescriptor: {descriptor}")
        print(f"Predicted label: {'Phishing' if predicted_label == 1 else 'Non-phishing'}")
        print(f"True label: {'Phishing' if true_label == 1 else 'Non-phishing'}")
        print("Top contributing features:")
        for word, score in word_importance_sorted:
            print(f"  {word}: {'+' if score >= 0 else '-'}{abs(score):.4f}")

    print("\n--- Summary Report ---")
    print(classification_report(true_labels, predictions, target_names=['Non-phishing', 'Phishing']))
    
    # Return data that could be useful for further analysis
    return {
        'descriptors': descriptors,
        'predictions': predictions,
        'true_labels': true_labels,
        'email_texts': email_texts
    }

# Example usage:
if __name__ == "__main__":
    
    # Example directory path - replace with your actual path
    text_output_dir = "TextOutput"
    
    # Load the trained model and vectorizer
    model = joblib.load("svm_model.pkl")

    # Load the vectorizer from a saved file (if saved separately). If not saved, fit once and save it.
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    df_phishing_email = pd.read_csv('phishing_email.csv')  # Load again to refit vectorizer
    vectorizer.fit(df_phishing_email['text_combined'])

    
    try:
        print("Processing email files and evaluating...")
        # This is just a placeholder - in practice, you need to have properly trained models
        print("Note: This is a demonstration - you need to load properly trained models")
        
        # Call the function with your directory
        results = evaluate_emails(email_directory=text_output_dir, vectorizer=vectorizer, model=model)
        
    except Exception as e:
        print(f"Error: {str(e)}")