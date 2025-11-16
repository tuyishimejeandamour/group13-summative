"""
Multimodal User Authentication & Product Recommendation System
=================================================================

This Streamlit application integrates three ML models:
1. Face Recognition - Verifies user identity via facial image
2. Voice Verification - Confirms transaction via voice authentication
3. Product Recommendation - Predicts product category based on user behavior

Usage:
    streamlit run system_integration.py

Requirements:
    - face_recognition model files in face_recognition/models/
    - voice_recognition model files in voice_recognition/models/
    - product_recommender model files in product_recommender/models/best_models/
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import joblib
import librosa
import tempfile
from PIL import Image
import io

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import configurations
from src.config import Config

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Multimodal Authentication System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_face_recognition_model():
    """Load face recognition model and preprocessors"""
    try:
        face_model_dir = project_root / 'face_recognition' / 'models'
        model = joblib.load(face_model_dir / 'face_recognition_model.pkl')
        scaler = joblib.load(face_model_dir / 'face_recognition_scaler.pkl')
        
        # Load VGG16 for feature extraction
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.applications.vgg16 import preprocess_input
        from tensorflow.keras.models import Model
        
        base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_extractor': feature_extractor,
            'users': Config.get_users()
        }
    except Exception as e:
        st.error(f"Failed to load face recognition model: {e}")
        return None

@st.cache_resource
def load_voice_verification_model():
    """Load voice verification model and preprocessors"""
    try:
        voice_model_dir = project_root / 'voice_recognition' / 'models'
        model = joblib.load(voice_model_dir / 'voice_verification_model.pkl')
        scaler = joblib.load(voice_model_dir / 'voice_verification_scaler.pkl')
        
        return {
            'model': model,
            'scaler': scaler
        }
    except Exception as e:
        st.error(f"Failed to load voice verification model: {e}")
        return None

@st.cache_resource
def load_product_recommendation_model():
    """Load product recommendation model and preprocessors"""
    try:
        product_model_dir = project_root / 'product_recommender' / 'models' / 'best_models'
        model = joblib.load(product_model_dir / 'product_recommendation_model.pkl')
        scaler = joblib.load(product_model_dir / 'scaler.pkl')
        encoder = joblib.load(product_model_dir / 'label_encoder.pkl')
        
        return {
            'model': model,
            'scaler': scaler,
            'encoder': encoder
        }
    except Exception as e:
        st.error(f"Failed to load product recommendation model: {e}")
        return None

# ==================== FEATURE EXTRACTION ====================
def extract_face_features(image_array, feature_extractor):
    """Extract VGG16 features from face image"""
    from tensorflow.keras.applications.vgg16 import preprocess_input
    
    # Resize to 224x224
    img_resized = cv2.resize(image_array, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = feature_extractor.predict(img_array, verbose=0)
    return features.flatten()

def extract_voice_features(audio_path):
    """Extract audio features (MFCCs, spectral features)"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Extract spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Combine features
        features = np.concatenate([
            mfccs_mean,
            mfccs_std,
            [spectral_centroid, spectral_rolloff, zero_crossing_rate]
        ])
        
        return features
    except Exception as e:
        st.error(f"Error extracting voice features: {e}")
        return None

# ==================== PREDICTION FUNCTIONS ====================
def predict_face(image, face_models):
    """Predict user identity from face image"""
    try:
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Convert RGB to BGR if needed
        if img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Extract features
        features = extract_face_features(img_array, face_models['feature_extractor'])
        
        # Scale features
        features_scaled = face_models['scaler'].transform(features.reshape(1, -1))
        
        # Predict
        prediction = face_models['model'].predict(features_scaled)[0]
        probabilities = face_models['model'].predict_proba(features_scaled)[0]
        
        # Get user name
        users = face_models['users']
        predicted_user = users[prediction]
        confidence = probabilities[prediction] * 100
        
        # Get probabilities for all users
        user_probs = {users[i]: probabilities[i] * 100 for i in range(len(users))}
        
        return {
            'user': predicted_user,
            'confidence': confidence,
            'probabilities': user_probs
        }
    except Exception as e:
        st.error(f"Face prediction error: {e}")
        return None

def verify_voice(audio_file, voice_models):
    """Verify voice authentication"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract features
        features = extract_voice_features(tmp_path)
        if features is None:
            return None
        
        # Scale features
        features_scaled = voice_models['scaler'].transform(features.reshape(1, -1))
        
        # Predict
        prediction = voice_models['model'].predict(features_scaled)[0]
        probabilities = voice_models['model'].predict_proba(features_scaled)[0]
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        is_approved = prediction == 1  # Assuming 1 = approved
        confidence = probabilities[prediction] * 100
        
        return {
            'approved': is_approved,
            'confidence': confidence
        }
    except Exception as e:
        st.error(f"Voice verification error: {e}")
        return None

def recommend_product(user_data, product_models):
    """Recommend product category based on user behavior"""
    try:
        # Expected features in order
        expected_features = [
            'purchase_interest_score_mean', 'purchase_interest_score',
            'purchase_amount', 'purchase_amount_mean', 'is_weekend', 'purchase_month',
            'engagement_score', 'purchase_amount_max', 'purchase_day_of_month',
            'customer_rating_mean', 'engagement_interest_interaction',
            'purchase_quarter', 'purchase_day_of_week', 'customer_rating_max',
            'purchase_amount_min', 'customer_rating_min', 'purchase_amount_std',
            'customer_rating', 'engagement_score_std', 'purchase_amount_sum',
            'amount_rating_interaction', 'sentiment_numeric',
            'purchase_interest_score_std', 'engagement_score_mean',
            'engagement_score_max', 'amount_per_engagement'
        ]
        
        # Fill missing features with defaults
        for feat in expected_features:
            if feat not in user_data:
                user_data[feat] = 0
        
        # Create DataFrame with exact feature order
        X = pd.DataFrame([user_data], columns=expected_features)
        
        # Scale features
        X_scaled = product_models['scaler'].transform(X)
        
        # Predict
        prediction = product_models['model'].predict(X_scaled)[0]
        probabilities = product_models['model'].predict_proba(X_scaled)[0]
        
        # Decode prediction
        product = product_models['encoder'].inverse_transform([prediction])[0]
        confidence = probabilities[prediction] * 100
        
        # Get all product probabilities
        all_products = product_models['encoder'].classes_
        product_probs = {all_products[i]: probabilities[i] * 100 for i in range(len(all_products))}
        
        return {
            'product': product,
            'confidence': confidence,
            'probabilities': product_probs
        }
    except Exception as e:
        st.error(f"Product recommendation error: {e}")
        return None

# ==================== SESSION STATE INITIALIZATION ====================
if 'face_verified' not in st.session_state:
    st.session_state.face_verified = False
if 'voice_verified' not in st.session_state:
    st.session_state.voice_verified = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'prediction_ready' not in st.session_state:
    st.session_state.prediction_ready = False

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown('<div class="main-header">🔐 Multimodal Authentication System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>System Flow:</strong><br>
    1️⃣ Face Recognition → Verify user identity<br>
    2️⃣ Voice Verification → Confirm transaction approval<br>
    3️⃣ Product Recommendation → Get personalized product suggestions
    </div>
    """, unsafe_allow_html=True)
    
    # Load all models
    with st.spinner("Loading AI models..."):
        face_models = load_face_recognition_model()
        voice_models = load_voice_verification_model()
        product_models = load_product_recommendation_model()
    
    if not all([face_models, voice_models, product_models]):
        st.error("Failed to load one or more models. Please check model files.")
        return
    
    st.success(" All models loaded successfully!")
    
    # ==================== STEP 1: FACE RECOGNITION ====================
    st.markdown('<div class="step-header">Step 1: Face Recognition 👤</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_face = st.file_uploader(
            "Upload your facial image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of your face"
        )
        
        if uploaded_face:
            image = Image.open(uploaded_face)
            st.image(image, caption="Uploaded Face Image", use_container_width=True)
            
            if st.button(" Verify Face", type="primary"):
                with st.spinner("Analyzing face..."):
                    result = predict_face(image, face_models)
                    
                    if result and result['confidence'] > Config.CONFIDENCE_THRESHOLD * 100:
                        st.session_state.face_verified = True
                        st.session_state.current_user = result['user']
                        
                        st.markdown(f"""
                        <div class="success-box">
                        <strong> Face Verified!</strong><br>
                        User: <strong>{result['user']}</strong><br>
                        Confidence: <strong>{result['confidence']:.2f}%</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.session_state.face_verified = False
                        st.markdown("""
                        <div class="error-box">
                        <strong>❌ Face Not Recognized</strong><br>
                        Access Denied. Please try again or contact administrator.
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.face_verified:
            st.markdown("""
            <div class="success-box">
            <strong>Face Recognition Status</strong><br>
             User verified successfully<br>
            Proceed to voice verification
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            <strong>Face Recognition Status</strong><br>
             Awaiting face verification<br>
            Please upload and verify your face image
            </div>
            """, unsafe_allow_html=True)
    
    # ==================== STEP 2: VOICE VERIFICATION ====================
    if st.session_state.face_verified:
        st.markdown('<div class="step-header">Step 2: Voice Verification 🎤</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_voice = st.file_uploader(
                "Upload voice confirmation",
                type=['wav', 'mp3'],
                help="Record yourself saying 'Yes, approve' or 'Confirm transaction'"
            )
            
            if uploaded_voice:
                st.audio(uploaded_voice, format='audio/wav')
                
                if st.button(" Verify Voice", type="primary"):
                    with st.spinner("Analyzing voice..."):
                        result = verify_voice(uploaded_voice, voice_models)
                        
                        if result and result['approved']:
                            st.session_state.voice_verified = True
                            st.session_state.prediction_ready = True
                            
                            st.markdown(f"""
                            <div class="success-box">
                            <strong> Voice Verified!</strong><br>
                            Authentication complete<br>
                            Confidence: <strong>{result['confidence']:.2f}%</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.session_state.voice_verified = False
                            st.markdown("""
                            <div class="error-box">
                            <strong>❌ Voice Not Recognized</strong><br>
                            Transaction denied. Please try again.
                            </div>
                            """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.voice_verified:
                st.markdown("""
                <div class="success-box">
                <strong>Voice Verification Status</strong><br>
                 Voice authenticated successfully<br>
                Ready for product recommendation
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                <strong>Voice Verification Status</strong><br>
                 Awaiting voice confirmation<br>
                Please upload and verify your voice
                </div>
                """, unsafe_allow_html=True)
    
    # ==================== STEP 3: PRODUCT RECOMMENDATION ====================
    if st.session_state.prediction_ready:
        st.markdown('<div class="step-header">Step 3: Product Recommendation 🛍️</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
        <strong>Welcome, {st.session_state.current_user}!</strong><br>
        Based on your profile and behavior, we'll recommend products you might like.
        </div>
        """, unsafe_allow_html=True)
        
        # Sample user data input (in real scenario, this would come from database)
        with st.expander(" View/Edit User Behavior Data (Optional)", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                purchase_amount = st.number_input("Purchase Amount ($)", value=49.99, min_value=0.0)
                engagement_score = st.slider("Engagement Score", 0, 100, 75)
                customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.5)
            
            with col2:
                purchase_interest = st.slider("Purchase Interest", 0.0, 10.0, 7.5)
                is_weekend = st.checkbox("Is Weekend Purchase", value=False)
                sentiment = st.selectbox("Review Sentiment", ["Positive", "Neutral", "Negative"])
            
            with col3:
                purchase_month = st.selectbox("Purchase Month", list(range(1, 13)), index=5)
                purchase_day = st.slider("Day of Month", 1, 31, 15)
                purchase_quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=1)
        
        if st.button("🎯 Get Product Recommendation", type="primary"):
            with st.spinner("Generating recommendation..."):
                # Prepare user data
                sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
                
                user_data = {
                    'purchase_amount': purchase_amount,
                    'purchase_amount_mean': purchase_amount * 0.9,
                    'purchase_amount_sum': purchase_amount * 4.5,
                    'purchase_amount_max': purchase_amount * 1.2,
                    'purchase_amount_min': purchase_amount * 0.65,
                    'purchase_amount_std': purchase_amount * 0.25,
                    'customer_rating': customer_rating,
                    'customer_rating_mean': customer_rating,
                    'customer_rating_max': min(5.0, customer_rating + 0.5),
                    'customer_rating_min': max(1.0, customer_rating - 0.5),
                    'engagement_score': engagement_score,
                    'engagement_score_mean': engagement_score,
                    'engagement_score_max': min(100, engagement_score + 10),
                    'engagement_score_std': engagement_score * 0.15,
                    'purchase_interest_score': purchase_interest,
                    'purchase_interest_score_mean': purchase_interest,
                    'purchase_interest_score_std': purchase_interest * 0.2,
                    'purchase_month': purchase_month,
                    'purchase_day_of_month': purchase_day,
                    'purchase_quarter': purchase_quarter,
                    'purchase_day_of_week': 2,
                    'is_weekend': 1 if is_weekend else 0,
                    'sentiment_numeric': sentiment_map[sentiment],
                    'amount_rating_interaction': purchase_amount * customer_rating,
                    'engagement_interest_interaction': engagement_score * purchase_interest,
                    'amount_per_engagement': purchase_amount / (engagement_score + 1)
                }
                
                result = recommend_product(user_data, product_models)
                
                if result:
                    st.markdown(f"""
                    <div class="success-box">
                    <h3> Recommended Product Category</h3>
                    <h2>{result['product']}</h2>
                    <p>Confidence: <strong>{result['confidence']:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show all probabilities
                    st.subheader(" All Product Probabilities")
                    prob_df = pd.DataFrame(
                        list(result['probabilities'].items()),
                        columns=['Product', 'Probability (%)']
                    ).sort_values('Probability (%)', ascending=False)
                    
                    st.bar_chart(prob_df.set_index('Product'))
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("🔐 System Status")
        
        st.markdown(f"""
        **Face Recognition:**  
        {' Verified' if st.session_state.face_verified else ' Not Verified'}
        
        **Voice Verification:**  
        {' Approved' if st.session_state.voice_verified else ' Not Approved'}
        
        **Current User:**  
        {st.session_state.current_user if st.session_state.current_user else 'None'}
        """)
        
        st.markdown("---")
        
        if st.button(" Reset Session"):
            st.session_state.face_verified = False
            st.session_state.voice_verified = False
            st.session_state.current_user = None
            st.session_state.prediction_ready = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ About")
        st.markdown("""
        This system uses three AI models:
        - **Face Recognition** (Random Forest + VGG16)
        - **Voice Verification** (Audio Feature Classification)
        - **Product Recommendation** (Ensemble ML)
        """)

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()