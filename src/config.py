"""
Configuration Module

Centralized configuration for the face recognition system.
This module provides shared constants and utilities to ensure consistency
across all scripts, notebooks, and models.

Usage:
    from src.config import Config
    
    # Get current users
    users = Config.get_users()
    
    # Get label mapping
    labels = Config.get_label_mapping()
    
    # Get paths
    images_dir = Config.IMAGES_DIR
"""

from pathlib import Path
from typing import List, Dict, Optional


class Config:
    """
    Centralized configuration for the face recognition system.
    
    All hardcoded values should be defined here to maintain consistency
    across the entire project.
    """
    
    # ==================== PROJECT PATHS ====================
    
    # Get project root (parent of src directory)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / 'data'
    IMAGES_DIR = DATA_DIR / 'images'
    FEATURES_DIR = DATA_DIR / 'features'
    PROCESSED_DIR = DATA_DIR / 'processed'
    RAW_DIR = DATA_DIR / 'raw'
    AUDIO_DIR = DATA_DIR / 'audio'
    
    # Output directories
    MODELS_DIR = PROJECT_ROOT / 'models'
    REPORTS_DIR = PROJECT_ROOT / 'reports'
    NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
    
    # Feature files
    IMAGE_FEATURES_CSV = FEATURES_DIR / 'image_features.csv'
    
    # Model files
    FACE_MODEL_PATH = MODELS_DIR / 'face_recognition_model.pkl'
    FACE_SCALER_PATH = MODELS_DIR / 'face_recognition_scaler.pkl'
    FACE_METADATA_PATH = MODELS_DIR / 'face_recognition_metadata.json'
    
    # Report files
    SAMPLE_IMAGES_REPORT = REPORTS_DIR / 'sample_images.png'
    
    # ==================== TEAM MEMBERS ====================
    
    # Default team members (can be overridden)
    DEFAULT_USERS = ['Alice', 'Armstrong', 'cedric', 'yassin']
    
    # Expressions for facial images
    EXPRESSIONS = ['neutral', 'smiling', 'surprised']
    
    # Folders to exclude from auto-detection
    EXCLUDED_FOLDERS = ['test', 'temp', 'backup', '.git', '__pycache__']
    
    # ==================== MODEL PARAMETERS ====================
    
    # Image processing
    IMAGE_TARGET_SIZE = (224, 224)  # VGG16 input size
    VGG16_FEATURE_DIM = 512  # VGG16 output feature dimension
    
    # Augmentation types
    AUGMENTATION_TYPES = [
        'original',
        'rotation_15',
        'rotation_neg15',
        'horizontal_flip',
        'brightness_increase',
        'brightness_decrease',
        'grayscale',
        'blur'
    ]
    
    # Random Forest parameters
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 20
    RF_MIN_SAMPLES_SPLIT = 2
    RF_MIN_SAMPLES_LEAF = 1
    RF_RANDOM_STATE = 42
    
    # Train-test split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Prediction thresholds
    CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence for known person
    
    # ==================== HELPER METHODS ====================
    
    @classmethod
    def get_users(cls, auto_detect: bool = False) -> List[str]:
        """
        Get the list of users.
        
        Args:
            auto_detect: If True, automatically detect users from IMAGES_DIR.
                        If False, returns DEFAULT_USERS.
        
        Returns:
            List of user names
        """
        if auto_detect:
            return cls._auto_detect_users()
        return cls.DEFAULT_USERS.copy()
    
    @classmethod
    def _auto_detect_users(cls) -> List[str]:
        """
        Automatically detect user directories from IMAGES_DIR.
        
        Returns:
            Sorted list of user names (directory names)
        """
        try:
            if not cls.IMAGES_DIR.exists():
                print(f"Warning: Images directory {cls.IMAGES_DIR} does not exist")
                return cls.DEFAULT_USERS.copy()
            
            # Get all subdirectories, excluding test/temp folders
            users = [
                d.name for d in cls.IMAGES_DIR.iterdir()
                if d.is_dir() 
                and not d.name.startswith('.')
                and d.name.lower() not in cls.EXCLUDED_FOLDERS
            ]
            
            if not users:
                print(f"Warning: No user directories found, using defaults")
                return cls.DEFAULT_USERS.copy()
            
            return sorted(users)
        except Exception as e:
            print(f"Error detecting users: {e}, using defaults")
            return cls.DEFAULT_USERS.copy()
    
    @classmethod
    def get_label_mapping(cls, users: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Get the label mapping (user name -> numeric label).
        
        Args:
            users: List of user names. If None, uses DEFAULT_USERS.
        
        Returns:
            Dictionary mapping user names to numeric labels
        """
        if users is None:
            users = cls.DEFAULT_USERS
        
        # Sort to ensure consistency
        return {name: idx for idx, name in enumerate(sorted(users))}
    
    @classmethod
    def get_reverse_label_mapping(cls, users: Optional[List[str]] = None) -> Dict[int, str]:
        """
        Get the reverse label mapping (numeric label -> user name).
        
        Args:
            users: List of user names. If None, uses DEFAULT_USERS.
        
        Returns:
            Dictionary mapping numeric labels to user names
        """
        label_map = cls.get_label_mapping(users)
        return {v: k for k, v in label_map.items()}
    
    @classmethod
    def ensure_directories(cls):
        """
        Create all necessary directories if they don't exist.
        """
        directories = [
            cls.DATA_DIR,
            cls.IMAGES_DIR,
            cls.FEATURES_DIR,
            cls.PROCESSED_DIR,
            cls.RAW_DIR,
            cls.AUDIO_DIR,
            cls.MODELS_DIR,
            cls.REPORTS_DIR,
            cls.NOTEBOOKS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_config_summary(cls) -> str:
        """
        Get a summary of the current configuration.
        
        Returns:
            Formatted string with configuration details
        """
        users = cls.get_users()
        label_map = cls.get_label_mapping(users)
        
        summary = [
            "=" * 60,
            "FACE RECOGNITION SYSTEM CONFIGURATION",
            "=" * 60,
            f"\nProject Root: {cls.PROJECT_ROOT}",
            f"\nTeam Members ({len(users)}):",
        ]
        
        for user, label in sorted(label_map.items(), key=lambda x: x[1]):
            summary.append(f"  {label}: {user}")
        
        summary.extend([
            f"\nExpressions: {', '.join(cls.EXPRESSIONS)}",
            f"Augmentations: {len(cls.AUGMENTATION_TYPES)} types",
            f"\nImage Size: {cls.IMAGE_TARGET_SIZE}",
            f"Feature Dimension: {cls.VGG16_FEATURE_DIM}",
            f"\nModel Type: Random Forest",
            f"  - Estimators: {cls.RF_N_ESTIMATORS}",
            f"  - Max Depth: {cls.RF_MAX_DEPTH}",
            f"  - Random State: {cls.RF_RANDOM_STATE}",
            f"\nConfidence Threshold: {cls.CONFIDENCE_THRESHOLD * 100:.0f}%",
            "=" * 60
        ])
        
        return "\n".join(summary)


# Convenience function for backwards compatibility
def get_users(auto_detect: bool = False) -> List[str]:
    """Get the list of users."""
    return Config.get_users(auto_detect)


def get_label_mapping(users: Optional[List[str]] = None) -> Dict[str, int]:
    """Get label mapping dictionary."""
    return Config.get_label_mapping(users)


if __name__ == "__main__":
    # Print configuration summary when run directly
    print(Config.get_config_summary())
    
    # Show auto-detected users
    print("\nAuto-detected users:")
    detected = Config.get_users(auto_detect=True)
    print(f"  {detected}")
