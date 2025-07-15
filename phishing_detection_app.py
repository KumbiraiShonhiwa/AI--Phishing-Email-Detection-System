from flask import Flask, render_template, request, redirect, url_for, session, abort
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import re
import secrets
import hashlib
import logging
import html
from werkzeug.utils import secure_filename
import traceback
from logging.handlers import RotatingFileHandler
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=10000, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with secure configurations
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)  # Generate a secure random secret key
app.config['SESSION_COOKIE_SECURE'] = True  # Only send cookies over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to cookies
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # Session timeout in seconds (30 minutes)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # Limit file uploads to 1MB
app.config['UPLOAD_FOLDER'] = 'temp_uploads'  # Temporary folder for file uploads
app.config['ALLOWED_EXTENSIONS'] = {'txt'}  # Only allow plain text files

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize rate limiting variables
# A simple in-memory rate limiting implementation
request_history = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def rate_limit(func):
    """Decorator to implement basic rate limiting"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        client_ip = request.remote_addr
        current_time = pd.Timestamp.now()
        
        # Initialize or update request history
        if client_ip not in request_history:
            request_history[client_ip] = []
        
        # Remove requests older than 1 minute
        request_history[client_ip] = [t for t in request_history[client_ip] 
                                    if (current_time - t).total_seconds() < 60]
        
        # If more than 10 requests in the last minute, rate limit
        if len(request_history[client_ip]) >= 10:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return render_template('index.html', error="Too many requests. Please try again later."), 429
        
        # Add current request to history
        request_history[client_ip].append(current_time)
        
        return func(*args, **kwargs)
    return wrapper

def sanitize_input(text):
    """Sanitize user input to prevent XSS attacks"""
    if text is None:
        return ""
    # Escape HTML special characters
    sanitized = html.escape(text)
    # Additional sanitization - remove potentially dangerous patterns
    sanitized = re.sub(r'<script.*?>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    return sanitized

def safely_load_model():
    """Safely load the trained model and vectorizer with error handling"""
    try:
        model = joblib.load("svm_model.pkl")
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        # Load the training data to fit the vectorizer
        df_phishing_email = pd.read_csv('phishing_email.csv')
        vectorizer.fit(df_phishing_email['text_combined'])
        
        return model, vectorizer
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

# Load the trained model and vectorizer
model, vectorizer = safely_load_model()

def validate_email_text(email_text):
    """Validate email text input"""
    if not email_text or not isinstance(email_text, str):
        return False, "Invalid email text provided."
    
    if len(email_text) > 100000:  # Reasonable limit for email size
        return False, "Email text too large. Please provide a shorter email."
    
    # Check for potential harmful content
    dangerous_patterns = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'eval\(',
        r'document\.cookie',
        r'document\.location',
        r'<iframe',
        r'<object',
        r'<embed'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, email_text, re.IGNORECASE | re.DOTALL):
            logger.warning(f"Potentially dangerous content detected: {pattern}")
            return False, "Potentially malicious content detected."
    
    return True, ""

def evaluate_emails(email_data, vectorizer, model):
    """
    Evaluates emails using a trained SVM model and TF-IDF vectorizer.
    
    Parameters:
        email_data (pd.DataFrame or dict): Must contain 'descriptor', 'email_text', and 'true_labels'.
        vectorizer (TfidfVectorizer): Pretrained TF-IDF vectorizer.
        model (sklearn classifier): Trained classification model (e.g., SVC).
        
    Returns:
        str: 'Phishing' or 'Non-Phishing'
    """
    try:
        # Input validation
        if not isinstance(email_data, (dict, pd.DataFrame)):
            raise ValueError("email_data must be a dictionary or a pandas DataFrame.")
        
        if vectorizer is None or model is None:
            raise ValueError("Model or vectorizer not properly loaded.")
        
        # Convert to DataFrame if given as dict
        if isinstance(email_data, dict):
            # Validate dictionary structure
            required_keys = ['descriptor', 'email_text', 'true_labels']
            if not all(key in email_data for key in required_keys):
                raise ValueError(f"email_data dictionary must contain keys: {required_keys}")
            
            # Validate content types
            if not all(isinstance(email_data[key], list) for key in required_keys):
                raise ValueError("All fields in email_data must be lists.")
            
            # Convert to DataFrame
            df = pd.DataFrame(email_data)
        elif isinstance(email_data, pd.DataFrame):
            required_columns = ['descriptor', 'email_text', 'true_labels']
            if not all(col in email_data.columns for col in required_columns):
                raise ValueError(f"email_data DataFrame must contain columns: {required_columns}")
            df = email_data.copy()
        
        descriptors = []
        predictions = []
        true_labels = []
        email_texts = []

        for _, row in df.iterrows():
            descriptor = row['descriptor']
            email_text = row['email_text']
            true_label = row['true_labels']

            # Validate inputs
            if not isinstance(email_text, str):
                raise ValueError("Email text must be a string.")
            
            if not isinstance(descriptor, str):
                descriptor = str(descriptor)
            
            # Transform and predict
            try:
                X_email = vectorizer.transform([email_text])
                predicted_label = model.predict(X_email)

                descriptors.append(descriptor)
                predictions.append(predicted_label[0])
                true_labels.append(true_label)
                email_texts.append(email_text)
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Failed to process email: {str(e)}")

        # Only log minimal information for security
        logger.info(f"Successfully processed {len(descriptors)} emails")

        return 'Phishing' if predictions[0] == 1 else 'Non-Phishing'
    
    except Exception as e:
        logger.error(f"Error in evaluate_emails: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@rate_limit
def predict():
    """Handle email prediction requests with robust validation and error handling"""
    try:
        # Check if models are loaded properly
        if model is None or vectorizer is None:
            logger.error("Model or vectorizer not loaded")
            return render_template('index.html', error="System error: Models not properly loaded. Please try again later.")

        # CSRF token validation (if implemented)
        # if request.form.get('csrf_token') != session.get('csrf_token'):
        #     logger.warning("CSRF token mismatch")
        #     abort(403)  # Forbidden
        
        email_text = ""
        
        # Process text input
        if 'email_text' in request.form:
            email_text = request.form.get('email_text', '')
            email_text = sanitize_input(email_text)
            
        # Process file upload
        elif 'email_file' in request.files:
            uploaded_file = request.files['email_file']
            
            if uploaded_file.filename == '':
                return render_template('index.html', error="No file selected.")
                
            if not allowed_file(uploaded_file.filename):
                logger.warning(f"Invalid file type attempted: {uploaded_file.filename}")
                return render_template('index.html', error="Only .txt files are allowed.")
            
            try:
                # Secure the filename to prevent path traversal attacks
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save file temporarily
                uploaded_file.save(file_path)
                
                # Read file content with size limit
                with open(file_path, 'r', encoding='utf-8') as f:
                    email_text = f.read(100000)  # Limit to 100KB
                
                # Remove the file after processing
                os.remove(file_path)
                
            except UnicodeDecodeError:
                logger.warning(f"Unicode decode error for file: {uploaded_file.filename}")
                return render_template('index.html', error="File encoding not supported. Please use UTF-8 encoded text files.")
            except Exception as e:
                logger.error(f"File processing error: {str(e)}")
                return render_template('index.html', error="Error processing file. Please try again.")
        
        # Validate input
        if not email_text.strip():
            return render_template('index.html', error="Please enter or upload some email content.")
        
        valid, error_message = validate_email_text(email_text)
        if not valid:
            return render_template('index.html', error=error_message)
        
        # Process the email using our model
        try:
            label = evaluate_emails(
                email_data={'descriptor': ['User Input'], 'email_text': [email_text], 'true_labels': [0]},
                vectorizer=vectorizer,
                model=model
            )
            
            return render_template('index.html', prediction=label, email_content=email_text[:200]+"..." if len(email_text) > 200 else email_text)
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return render_template('index.html', error="An error occurred while processing your request. Please try again.")
            
    except Exception as e:
        logger.error(f"Unhandled exception in predict route: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('index.html', error="An unexpected error occurred. Please try again later.")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', error="Page not found."), 404

@app.errorhandler(413)
def request_entity_too_large(e):
    return render_template('index.html', error="File too large. Please upload a smaller file."), 413

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return render_template('index.html', error="Internal server error. Please try again later."), 500

if __name__ == '__main__':
    # Run the app with security settings
    # In production, set debug=False and use a proper WSGI server with HTTPS
    app.run(debug=False, host='127.0.0.1')
