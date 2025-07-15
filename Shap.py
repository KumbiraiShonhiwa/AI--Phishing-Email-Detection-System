import pandas as pd
import joblib
import shap
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model
model = joblib.load("svm_model.pkl")

# Load data again and fit the vectorizer (if not saved separately)
df_phishing_email = pd.read_csv("phishing_email.csv")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
vectorizer.fit(df_phishing_email["text_combined"])

# Transform the text data
X_all_vectorized = vectorizer.transform(df_phishing_email["text_combined"])

# Generate SHAP explanations
print("\nGenerating SHAP explanations for SVM model...")

# Use a subset of the data for explanation
X_sample = X_all_vectorized[:1]  # e.g., first email

# SHAP requires dense array for some explainers
X_dense = X_all_vectorized.toarray()

# Create explainer using the trained model and the background dataset (the full training set or a sample)
explainer = shap.LinearExplainer(model, X_dense, feature_perturbation="interventional")

# Compute SHAP values for a single sample
shap_values = explainer.shap_values(X_dense[0:1])

# Generate interactive force plot
shap.initjs()
shap_html = shap.force_plot(
    explainer.expected_value, 
    shap_values[0], 
    features=X_dense[0], 
    feature_names=vectorizer.get_feature_names_out()
)

# Save to HTML
with open("shap_explanation.html", "w") as f:
    f.write(shap_html.html())

print("SHAP explanation saved to shap_explanation.html")
