# Phishing Email Classification with SVM

# Load libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tabulate import tabulate


def train_model():
    # Explore Dataset
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # Load datasets
    df_SpamAssasin = pd.read_csv('SpamAssasin.csv')
    df_Nigerian_Fraud = pd.read_csv('Nigerian_Fraud.csv')
    df_phishing_email = pd.read_csv('phishing_email.csv')
    df_CEAS_08 = pd.read_csv('CEAS_08.csv')
    df_Enron = pd.read_csv('Enron.csv')
    df_Ling = pd.read_csv('Ling.csv')
    df_Nazario = pd.read_csv('Nazario.csv')

    # Check dataset sizes
    file_paths = {
        'SpamAssasin': df_SpamAssasin,
        'Nigerian_Fraud': df_Nigerian_Fraud,
        'Phishing_Email': df_phishing_email,
        'CEAS_08': df_CEAS_08,
        'Enron': df_Enron,
        'Ling': df_Ling,
        'Nazario': df_Nazario
    }

    spam_nonspam_counts = {}
    for name, df in file_paths.items():
        if 'label' in df.columns:
            spam_count = df[df['label'] == 1].shape[0]
            non_spam_count = df[df['label'] == 0].shape[0]
        else:
            spam_count = df.shape[0]
            non_spam_count = 0
        spam_nonspam_counts[name] = {
            'spam_count': spam_count,
            'non_spam_count': non_spam_count
        }

    for dataset, counts in spam_nonspam_counts.items():
        print(
            f"{dataset}: {counts['spam_count']} spam emails, {counts['non_spam_count']} non-spam emails")

    # Define features and labels
    df = df_CEAS_08  # or the merged dataframe
    X = df[['sender', 'subject', 'body', 'urls']].fillna(
        '').astype(str)  # select multiple relevant features
    X_combined = X['sender'] + ' ' + X['subject'] + \
        ' ' + X['body'] + ' ' + X['urls']
    y = df['label']

    # Define a column transformer to apply TF-IDF to each text column
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y)

    # Fit and transform the train data, and transform the test data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Undersample majority class
    undersampler = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(
        X_train_vectorized, y_train)

    # Train SVM model
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train_resampled, y_train_resampled)

    # Evaluate model
    y_pred = svm.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    report = classification_report(y_test, y_pred, output_dict=True)
    table_data = [
        [label, metrics['precision'], metrics['recall'],
            metrics['f1-score'], metrics['support']]
        for label, metrics in report.items() if isinstance(metrics, dict)
    ]
    print(tabulate(table_data, headers=[
          'Class', 'Precision', 'Recall', 'F1-score', 'Support'], tablefmt='github'))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False,
                annot_kws={'size': 14}, linewidths=0.5, linecolor='black')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_f1_idx = np.argmax(f1_scores)
    print(
        f"Best Precision: {precision[best_f1_idx]:.2f}, Best Recall: {recall[best_f1_idx]:.2f}, F1-score: {f1_scores[best_f1_idx]:.2f}")

    # Save the model
    joblib.dump(svm, 'svm_model.pkl')



def evaluate_emails(email_data, vectorizer, model, top_n=10):
    """
    Evaluates emails using a trained SVM model and TF-IDF vectorizer with explanation.

    Parameters:
        email_data (pd.DataFrame or dict): Must contain 'descriptor', 'email_text', and 'true_labels'.
        vectorizer (TfidfVectorizer): Pretrained TF-IDF vectorizer.
        model (sklearn classifier): Trained linear SVM classifier.
        top_n (int): Number of top features to show in interpretation.

    Returns:
        None
    """
    if isinstance(email_data, dict):
        df = pd.DataFrame(email_data)
    elif isinstance(email_data, pd.DataFrame):
        df = email_data.copy()
    else:
        raise ValueError(
            "email_data must be a dictionary or a pandas DataFrame.")

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

        combined_text = ' '.join(
            [row['sender'], row['descriptor'], row['email_text']])

        X_email = vectorizer.transform([combined_text])
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

        word_importance_sorted = sorted(
            word_importance, key=lambda x: abs(x[1]), reverse=True)[:top_n]

        print(f"\nDescriptor: {descriptor}")
        print(
            f"Predicted label: {'Phishing' if predicted_label == 1 else 'Non-phishing'}")
        print(
            f"True label: {'Phishing' if true_label == 1 else 'Non-phishing'}")
        print("Top contributing features:")
        for word, score in word_importance_sorted:
            print(f"  {word}: {'+' if score >= 0 else '-'}{abs(score):.4f}")

    print("\n--- Summary Report ---")
    print(classification_report(true_labels, predictions,
          target_names=['Non-phishing', 'Phishing']))


def main():
    # Step 1: Train the model
    train_model()

    # Step 2: Load vectorizer and model
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    # Load again to refit vectorizer
    df_phishing_email = pd.read_csv('phishing_email.csv')
    vectorizer.fit(df_phishing_email['text_combined'])

    # Load trained model
    model = joblib.load('svm_model.pkl')

    # Step 3: Define test emails (e.g., from Gmail or external source)
    email_data = {
        'descriptor': [
            "Flight booking confirmation",
            "Lottery scam",
            "Monthly billing statement",
            "Account suspension warning",
            "Password reset request",
            "Bank account alert",
            "Schedule change notice",
            "Meeting reminder",
            "Event invitation",
            "Project update",
            "Fake invoice attached",
            "Receipt for purchase",
            "Fake job offer",
            "Password reset request",
            "Compromised password alert",
            "Customer feedback request",
            "Bank account alert",
            "Newsletter subscription",
            "Prize win notification",
            "Tax refund scam",
            "Account confirmation",
            "Project update",
            "Work from home offer",
            "Unauthorized login attempt",
            "You won a MacBook"
        ],
        'sender': [  # 1. Legit (Flight booking confirmation)
            "itinerary@flights.example.com",

            # 2. Phishing (Lottery scam)
            "lotto@claim-prize-now.com",

            # 3. Legit (Monthly billing)
            "billing@services.example.com",

            # 4. Phishing (Account suspension warning)
            "support@verify-now-login.com",

            # 5. Legit (Password reset)
            "noreply@secure-reset.example.com",

            # 6. Phishing (Bank alert)
            "alerts@secure-bank-check.com",

            # 7. Legit (Schedule change)
            "admin@university.example.edu",

            # 8. Legit (Meeting reminder)
            "reminder@calendar.example.com",

            # 9. Legit (Event invitation)
            "events@techsummit.org",

            # 10. Legit (Project update)
            "updates@companyprojects.example.com",

            # 11. Phishing (Fake invoice)
            "billing@invoices-fake.com",

            # 12. Legit (Purchase receipt)
            "receipts@store.example.com",

            # 13. Phishing (Fake job offer)
            "hr@easyjobonline.com",

            # 14. Legit (Password reset)
            "support@safe-reset.example.org",

            # 15. Phishing (Password alert)
            "security@alert-security.com",

            # 16. Legit (Feedback request)
            "feedback@service.example.com",

            # 17. Phishing (Bank account alert)
            "security@securebank-verify.com",

            # 18. Legit (Newsletter)
            "newsletter@updates.example.com",

            # 19. Phishing (Prize win)
            "promo@macbook-promo-fake.com",

            # 20. Phishing (Tax refund)
            "revenue@taxrefund-now.com",

            # 21. Legit (Account confirmation)
            "no-reply@account.example.com",

            # 22. Legit (Project update)
            "project@updates.example.com",

            # 23. Phishing (Work from home)
            "careers@remotejobs123.com",

            # 24. Phishing (Unauthorized login)
            "security@secure-access-check.com",

            # 25. Phishing (You won a MacBook)
            "giveaway@prizes-fakeclaim.com"],
        'email_text': [
            "Your flight has been booked. Check your itinerary and download your boarding pass at https://flights.example.com/ticket12345.",
            "You have won $1,000,000 in the lottery. Send your details via https://claim-prize-now.com to receive the prize.",
            "Your monthly bill is ready. View it at https://billing.example.com/account.",
            "Your account will be suspended unless you verify your information immediately at https://verify-now-login.com.",
            "You requested a password reset. If this was you, follow the link to create a new one: https://secure-reset.example.com.",
            "Suspicious activity was detected in your bank account. Confirm your details now at https://secure-bank-check.com.",
            "There is a change in your class schedule. Check the updated timetable online at https://university.example.edu/schedule.",
            "This is a reminder that your team meeting is scheduled for 3 PM today in Room 4B.",
            "You are invited to the annual tech conference. RSVP by the end of the week at https://events.techsummit.org/rsvp.",
            "The latest update on the project is now available. Please review and respond.",
            "Attached is your unpaid invoice. Failure to pay will result in legal action. View the invoice at https://invoices-fake.com/urgent.",
            "Your receipt for the recent purchase is attached. Thank you for shopping with us.",
            "We are hiring! Submit your personal and banking details at https://easyjobonline.com/start-now to start working from home.",
            "You requested a password reset. If this was you, follow the link to create a new one: https://safe-reset.example.org.",
            "Your password has been compromised. Reset it immediately by clicking this link: https://alert-security.com/reset-now.",
            "We would love your feedback on your recent experience with our customer service. Visit https://feedback.example.com.",
            "Security alert: someone tried accessing your bank account. Verify details now at https://securebank-verify.com.",
            "Thank you for subscribing to our monthly newsletter. Stay tuned for updates.",
            "Congratulations! You've won a new MacBook Pro. Claim it now before it expires at https://macbook-promo-fake.com.",
            "You are eligible for a tax refund. Submit your bank details to claim it at https://taxrefund-now.com.",
            "Please confirm your account by clicking the verification link: https://account.example.com/confirm.",
            "Project documents are attached. Let us know if you have any questions.",
            "Work from home opportunity! Share your bank info and start today at https://remotejobs123.com/join.",
            "We noticed a login attempt from an unknown device. Verify your identity at https://secure-access-check.com.",
            "Congratulations! You've won a new MacBook Pro. Claim it now before it expires at https://prizes-fakeclaim.com."
        ],
        'true_labels': [
            0, 1, 0, 1, 0,
            1, 0, 0, 0, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 1,
            0, 0, 1, 1, 1
        ]
    }

    # Step 4: Evaluate the emails using the trained model
    evaluate_emails(email_data, vectorizer, model)


if __name__ == '__main__':
    main()
