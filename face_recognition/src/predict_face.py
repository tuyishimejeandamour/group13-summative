"""
Enhanced Face Recognition Script with Unknown Person Detection

This script includes confidence threshold checking to detect when an image
contains a person who is not in the training set.

Usage:
    python predict_face.py <image_path>
    python predict_face.py <image_path> --threshold 0.85
"""

import sys
import os
import argparse
import json
import numpy as np
import cv2
import joblib
from pathlib import Path
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Add src directory to path for Config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import Config


class EnhancedFaceRecognitionPredictor:
    """
    Enhanced face recognition predictor with unknown person detection
    """
    
    def __init__(self, model_dir='../models', confidence_threshold=0.80):
        """
        Initialize the predictor
        
        Args:
            model_dir: Directory containing the saved model files
            confidence_threshold: Minimum confidence to accept prediction (0.0-1.0)
        """
        # Get the script directory and construct absolute path
        script_dir = Path(__file__).parent.absolute()
        self.model_dir = (script_dir / model_dir).resolve()
        
        self.target_size = (224, 224)
        self.confidence_threshold = confidence_threshold
        
        # Load the trained model and scaler
        self.load_model()
        
        # Load metadata to get the class names used during training
        # This ensures label mapping matches what was used during training
        self.load_metadata()
        
        # Load VGG16 for feature extraction
        self.load_feature_extractor()
    
    def load_model(self):
        """Load the trained Random Forest model and scaler"""
        try:
            model_path = self.model_dir / 'face_recognition_model.pkl'
            scaler_path = self.model_dir / 'face_recognition_scaler.pkl'
            
            if not model_path.exists():
                model_path = self.model_dir / 'face_recognition.pkl'
            if not scaler_path.exists():
                scaler_path = self.model_dir / 'face_recognition_scaler.pkl'
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            print(f"Model loaded from: {model_path}")
            print(f"Scaler loaded from: {scaler_path}")
            print(f"Confidence threshold: {self.confidence_threshold*100:.0f}%")
            
        except FileNotFoundError:
            print(f"ERROR: Could not find model files in {self.model_dir}")
            sys.exit(1)
    
    def load_metadata(self):
        """
        Load metadata to get the class names used during training.
        This ensures label mapping matches what was used during training,
        preventing misidentification if filesystem changes after training.
        """
        try:
            metadata_path = self.model_dir / 'face_recognition_metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            class_names = metadata['class_names']
            # Create reverse label mapping: label_index -> user_name
            self.user_mapping = {i: name for i, name in enumerate(class_names)}
            
            print(f"Loaded class names from training: {class_names}")
            
        except FileNotFoundError:
            print(f"WARNING: Metadata file not found at {metadata_path}")
            print("Falling back to Config defaults (may cause incorrect predictions)")
            # Fallback to Config defaults
            default_users = Config.get_users()
            self.user_mapping = Config.get_reverse_label_mapping(users=default_users)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"WARNING: Error reading metadata: {e}")
            print("Falling back to Config defaults")
            default_users = Config.get_users()
            self.user_mapping = Config.get_reverse_label_mapping(users=default_users)
    
    def load_feature_extractor(self):
        """Load VGG16 model for feature extraction"""
        print("\nLoading VGG16 feature extractor...")
        base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.feature_extractor = base_model
        print("VGG16 loaded successfully")
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess an image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, self.target_size)
        img_array = np.expand_dims(img_resized, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array, img
    
    def extract_features(self, img_array):
        """Extract features from image using VGG16"""
        features = self.feature_extractor.predict(img_array, verbose=0)
        return features.flatten()
    
    def predict(self, image_path, show_probabilities=True):
        """
        Predict with unknown person detection
        
        Args:
            image_path: Path to the image file
            show_probabilities: Whether to show probability scores
            
        Returns:
            prediction: Dictionary containing prediction results
        """
        print(f"\n{'='*70}")
        print(f"ENHANCED FACE RECOGNITION PREDICTION")
        print(f"{'='*70}")
        print(f"\nImage: {image_path}")
        
        # Load and preprocess image
        print("\nStep 1: Loading and preprocessing image...")
        img_array, original_img = self.load_and_preprocess_image(image_path)
        print(f"Image loaded: {original_img.shape}")
        
        # Extract features
        print("\nStep 2: Extracting features using VGG16...")
        features = self.extract_features(img_array)
        print(f"Features extracted: {features.shape[0]} dimensions")
        
        # Scale features
        print("\nStep 3: Scaling features...")
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        print("Features scaled")
        
        # Make prediction
        print("\nStep 4: Making prediction with confidence check...")
        prediction_label = self.model.predict(features_scaled)[0]
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        confidence = prediction_proba[prediction_label]
        
        # Check if confidence meets threshold
        is_known_person = confidence >= self.confidence_threshold
        predicted_user = self.user_mapping[prediction_label] if is_known_person else "UNKNOWN"
        
        # Calculate uncertainty metrics
        max_prob = np.max(prediction_proba)
        prob_diff = max_prob - np.partition(prediction_proba, -2)[-2]  # Difference between top 2
        entropy = -np.sum(prediction_proba * np.log(prediction_proba + 1e-10))
        
        # Prepare results
        result = {
            'predicted_user': predicted_user,
            'predicted_label': int(prediction_label) if is_known_person else -1,
            'confidence': confidence * 100,
            'is_known_person': is_known_person,
            'threshold': self.confidence_threshold * 100,
            'probabilities': {
                self.user_mapping[i]: prob * 100 
                for i, prob in enumerate(prediction_proba)
            },
            'uncertainty_metrics': {
                'max_probability': max_prob,
                'probability_gap': prob_diff,
                'entropy': entropy
            }
        }
        
        # Display results
        print(f"\n{'='*70}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*70}")
        
        if is_known_person:
            print(f"\nStatus: KNOWN PERSON")
            print(f"Predicted User: {predicted_user}")
            print(f"Confidence: {confidence*100:.2f}%")
            print(f"Threshold: {self.confidence_threshold*100:.0f}% (PASSED)")
        else:
            print(f"\nStatus: UNKNOWN PERSON DETECTED")
            print(f"Best Match: {self.user_mapping[prediction_label]} (LOW CONFIDENCE)")
            print(f"Confidence: {confidence*100:.2f}%")
            print(f"Threshold: {self.confidence_threshold*100:.0f}% (NOT MET)")
            print(f"\nInterpretation:")
            print(f"   The model cannot confidently identify this person as one of the")
            print(f"   trained users (Alice, cedric, yassin). This person may be:")
            print(f"   - Not in the training dataset")
            print(f"   - A low-quality or unclear image")
            print(f"   - An unauthorized user attempting access")
        
        if show_probabilities:
            print(f"\nProbability Distribution:")
            for user, prob in result['probabilities'].items():
                bar = '#' * int(prob / 2)
                marker = ' <-' if user == self.user_mapping[prediction_label] else ''
                print(f"  {user:>8s}: {prob:>6.2f}% {bar}{marker}")
            
            print(f"\nUncertainty Metrics:")
            print(f"  Max Probability: {max_prob:.4f}")
            print(f"  Probability Gap (top 2): {prob_diff:.4f} {'(High = More certain)' if prob_diff > 0.3 else '(Low = Less certain)'}")
            print(f"  Entropy: {entropy:.4f} {'(Low = More certain)' if entropy < 0.5 else '(High = Less certain)'}")
        
        # Decision recommendation
        print(f"\n{'='*70}")
        print(f"AUTHENTICATION RECOMMENDATION")
        print(f"{'='*70}")
        
        if is_known_person and confidence > 0.95:
            print(f"GRANT ACCESS - High confidence identification")
        elif is_known_person and confidence > 0.85:
            print(f"GRANT ACCESS (with caution) - Acceptable confidence")
        elif is_known_person:
            print(f"VERIFY IDENTITY - Confidence meets threshold but is low")
            print(f"   Consider secondary authentication method")
        else:
            print(f"DENY ACCESS - Unknown person or insufficient confidence")
            print(f"   This person is not recognized as a trained user")
        
        print(f"\n{'='*70}")
        
        return result


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description='Enhanced Face Recognition with Unknown Person Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default threshold (80%)
  python predict_face_enhanced.py image.jpg
  
  # Stricter threshold (90%)
  python predict_face_enhanced.py image.jpg --threshold 0.90
  
  # More lenient threshold (70%)
  python predict_face_enhanced.py image.jpg --threshold 0.70
  
Confidence Threshold Guide:
  0.95+ : Very strict - Only high confidence matches
  0.85  : Strict - Recommended for security
  0.80  : Balanced - Default setting
  0.70  : Lenient - May accept more false positives
  0.60- : Very lenient - Not recommended
        '''
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file to predict'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='../models',
        help='Directory containing the trained model files (default: ../models)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.80,
        help='Confidence threshold (0.0-1.0, default: 0.80)'
    )
    
    parser.add_argument(
        '--no-probabilities',
        action='store_true',
        help='Hide probability distribution in output'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("ERROR: Threshold must be between 0.0 and 1.0")
        return 1
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"ERROR: Image file not found: {args.image_path}")
        return 1
    
    try:
        # Create predictor
        predictor = EnhancedFaceRecognitionPredictor(
            model_dir=args.model_dir,
            confidence_threshold=args.threshold
        )
        
        # Make prediction
        result = predictor.predict(
            args.image_path,
            show_probabilities=not args.no_probabilities
        )
        
        # Exit code: 0 for known person, 1 for unknown
        return 0 if result['is_known_person'] else 1
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
