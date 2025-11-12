"""
Image Processing Module - Task 2 (Yassin)

This module handles the complete image processing pipeline for facial recognition:
1. Loading facial images from user directories (3 per member: neutral, smiling, surprised)
2. Displaying sample pictures of each member for verification
3. Applying various augmentation techniques to increase dataset diversity
4. Extracting meaningful features using deep learning (VGG16)
5. Saving extracted features to CSV for model training

Requirements Implementation:
- Submit at least 3 facial images: one neutral, one smiling, and one surprised
- Load and display sample pictures of each member
- Apply augmentations per image (e.g., rotation, flipping, grayscale)
- Extract and save image features (e.g., embeddings, histograms) into image_features.csv

Key Functions:
- load_images(): Load all facial images from data/images directory
- display_sample_images(): Display sample pictures of each member (required step)
- augment_image(): Apply transformations (rotation, flipping, brightness, etc.)
- extract_features(): Use pre-trained VGG16 to extract feature embeddings
- process_all_images(): Main pipeline function
- save_features_to_csv(): Export features to image_features.csv

Usage Examples:
    # Auto-detect users from data/images directory
    processor = ImageProcessor()
    
    # Specify custom users list
    processor = ImageProcessor(users=['Alice', 'Bob', 'Charlie'])
    
    # Specify custom directory and users
    processor = ImageProcessor(base_dir='/path/to/images', users=['User1', 'User2'])
    
    # Process images
    features = processor.process_all_images()
    processor.save_features_to_csv(features)
"""

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from pathlib import Path


class ImageProcessor:
    """
    Main class for processing facial images and extracting features.
    Uses VGG16 pre-trained on ImageNet for feature extraction.
    """
    
    def __init__(self, base_dir=None, target_size=(224, 224), users=None):
        """
        Initialize the image processor with configuration.
        
        Args:
            base_dir: Path to the images directory containing user folders
            target_size: Image dimensions required by VGG16 (224x224)
            users: List of user names to process. If None, auto-detects from base_dir
        """
        # Auto-detect base_dir if not provided
        if base_dir is None:
            # Get the project root directory (2 levels up from this file)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            base_dir = project_root / 'data' / 'images'
        
        self.base_dir = str(base_dir)
        self.target_size = target_size
        
        # Auto-detect users from directory if not provided
        if users is None:
            self.users = self._auto_detect_users()
        else:
            self.users = users
        
        # Load pre-trained VGG16 model without the top classification layer
        # This gives us a 512-dimensional feature vector for each image
        base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
        
        print(f"Image processor initialized with VGG16 feature extractor")
        print(f"Users to process: {self.users}")
    
    def _auto_detect_users(self):
        """
        Automatically detect user directories from the base_dir.
        
        Returns:
            List of user names (directory names) found in base_dir
        """
        try:
            base_path = Path(self.base_dir)
            if not base_path.exists():
                print(f"Warning: Base directory {self.base_dir} does not exist")
                return []
            
            # Get all subdirectories (user folders)
            users = [d.name for d in base_path.iterdir() 
                    if d.is_dir() 
                    and not d.name.startswith('.') 
                    and d.name.lower() not in ['test', 'temp', 'backup']]
            
            if not users:
                print(f"Warning: No user directories found in {self.base_dir}")
            else:
                print(f"Auto-detected {len(users)} user(s): {users}")
            
            return sorted(users)  # Sort for consistency
        except Exception as e:
            print(f"Error detecting users: {e}")
            return []
    
    def load_image(self, image_path):
        """
        Load and preprocess a single image file.
        
        Args:
            image_path: Full path to the image file
            
        Returns:
            img_array: Preprocessed image as numpy array ready for VGG16
            original_img: Original image for visualization
        """
        # Load image and resize to target dimensions
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        original_img = img.copy()
        
        # Resize to VGG16 input size
        img_resized = cv2.resize(img, self.target_size)
        
        # Convert to array and add batch dimension
        img_array = np.expand_dims(img_resized, axis=0)
        
        # Apply VGG16 preprocessing (mean subtraction, etc.)
        img_array = preprocess_input(img_array)
        
        return img_array, original_img
    
    def augment_image(self, img, augmentation_type):
        """
        Apply various augmentation techniques to increase dataset diversity.
        
        Args:
            img: Input image as numpy array
            augmentation_type: Type of augmentation to apply
            
        Returns:
            augmented_img: Transformed image
        """
        if augmentation_type == 'original':
            return img
        
        elif augmentation_type == 'rotation_15':
            # Rotate image by 15 degrees clockwise
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
            return cv2.warpAffine(img, rotation_matrix, (width, height))
        
        elif augmentation_type == 'rotation_neg15':
            # Rotate image by 15 degrees counter-clockwise
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -15, 1.0)
            return cv2.warpAffine(img, rotation_matrix, (width, height))
        
        elif augmentation_type == 'horizontal_flip':
            # Mirror image horizontally
            return cv2.flip(img, 1)
        
        elif augmentation_type == 'brightness_increase':
            # Increase brightness by adding constant value
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        elif augmentation_type == 'brightness_decrease':
            # Decrease brightness
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.7, 0, 255)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        elif augmentation_type == 'grayscale':
            # Convert to grayscale and back to RGB (3 channels)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        elif augmentation_type == 'blur':
            # Apply Gaussian blur
            return cv2.GaussianBlur(img, (5, 5), 0)
        
        else:
            return img
    
    def extract_features(self, img_array):
        """
        Extract deep learning features using pre-trained VGG16 model.
        
        Args:
            img_array: Preprocessed image array
            
        Returns:
            features: 512-dimensional feature vector
        """
        # Pass image through VGG16 to get feature embeddings
        features = self.feature_extractor.predict(img_array, verbose=0)
        
        # Flatten to 1D array
        return features.flatten()
    
    def process_single_image(self, image_path, user, expression):
        """
        Process a single image with all augmentations and extract features.
        
        Args:
            image_path: Path to the image file
            user: Username (Alice, cedric, yassin)
            expression: Expression type (neutral, smiling, surprised)
            
        Returns:
            features_list: List of dictionaries containing features and metadata
        """
        print(f"Processing {user}/{expression}...")
        
        # Load original image
        _, original_img = self.load_image(image_path)
        
        # Define augmentation types to apply
        augmentation_types = [
            'original',
            'rotation_15',
            'rotation_neg15',
            'horizontal_flip',
            'brightness_increase',
            'brightness_decrease',
            'grayscale',
            'blur'
        ]
        
        features_list = []
        
        # Apply each augmentation and extract features
        for aug_type in augmentation_types:
            # Apply augmentation
            augmented_img = self.augment_image(original_img.copy(), aug_type)
            
            # Resize and preprocess for VGG16
            img_resized = cv2.resize(augmented_img, self.target_size)
            img_array = np.expand_dims(img_resized, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features
            features = self.extract_features(img_array)
            
            # Create feature dictionary with metadata
            feature_dict = {
                'user': user,
                'expression': expression,
                'image_path': image_path,
                'augmentation': aug_type
            }
            
            # Add each feature dimension as a separate column
            for i, feature_value in enumerate(features):
                feature_dict[f'color_hist_{i}'] = feature_value
            
            # Add numeric label for classification
            # Dynamically create label mapping based on sorted users list
            label_map = {name: idx for idx, name in enumerate(sorted(self.users))}
            
            # Only add label if user is in the current users list
            if user in label_map:
                feature_dict['label'] = label_map[user]
            else:
                feature_dict['label'] = -1  # Unknown user
            
            features_list.append(feature_dict)
        
        return features_list
    
    def process_all_images(self):
        """
        Main pipeline function to process all images from all users.
        
        Returns:
            df_features: DataFrame containing all extracted features
        """
        all_features = []
        
        print("Starting image processing pipeline...")
        print(f"Target users: {self.users}")
        
        # Process images for each user
        for user in self.users:
            user_dir = os.path.join(self.base_dir, user)
            
            if not os.path.exists(user_dir):
                print(f"Warning: Directory not found for user {user}")
                continue
            
            # Get all image files in user directory
            image_files = [f for f in os.listdir(user_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"\nProcessing {len(image_files)} images for {user}")
            
            # Process each image
            for img_file in image_files:
                image_path = os.path.join(user_dir, img_file)
                
                # Determine expression from filename
                filename_lower = img_file.lower()
                if 'neutral' in filename_lower:
                    expression = 'neutral'
                elif 'smil' in filename_lower:  # Covers 'smile' and 'smiling'
                    expression = 'smiling'
                elif 'surpris' in filename_lower:  # Covers 'surprised' and 'suprised'
                    expression = 'surprised'
                else:
                    expression = 'unknown'
                
                # Process image with all augmentations
                features = self.process_single_image(image_path, user, expression)
                all_features.extend(features)
        
        # Convert to DataFrame
        df_features = pd.DataFrame(all_features)
        
        print(f"\nProcessing complete!")
        print(f"Total images processed: {len(df_features)}")
        print(f"Feature dimensions: {len([col for col in df_features.columns if col.startswith('color_hist_')])}")
        
        return df_features
    
    def save_features_to_csv(self, df_features, output_path=None):
        """
        Save extracted features to CSV file for model training.
        
        Args:
            df_features: DataFrame containing features and metadata
            output_path: Path to save the CSV file. If None, uses default location
                        in project's data/features/image_features.csv
        """
        # Use default path if not provided
        if output_path is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).resolve().parent.parent.parent
            output_path = project_root / 'data' / 'features' / 'image_features.csv'
        else:
            output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df_features.to_csv(output_path, index=False)
        
        print(f"\nFeatures saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"Shape: {df_features.shape}")
    
    def display_sample_images(self, save_output=True):
        """
        Load and display sample pictures of each member.
        This fulfills the requirement: "Load and display sample pictures of each member"
        
        Shows all 3 facial images per member (neutral, smiling, surprised) in a grid.
        
        Args:
            save_output: Whether to save the output figure to reports directory
        """
        print("\n" + "="*70)
        print("DISPLAYING SAMPLE IMAGES FROM EACH MEMBER")
        print("="*70)
        
        # Define expressions to display
        expressions = ['neutral', 'smiling', 'surprised']
        
        # Create a figure to display all images (4 users x 3 expressions)
        num_users = len(self.users)
        fig, axes = plt.subplots(num_users, 3, figsize=(15, num_users * 4))
        fig.suptitle('Sample Images from Each Team Member', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        images_found = 0
        images_missing = 0
        
        # Loop through each member and display their images
        for row_idx, user in enumerate(self.users):
            user_dir = os.path.join(self.base_dir, user)
            print(f"\n{user}:")
            
            for col_idx, expression in enumerate(expressions):
                img_path = None
                
                # Try to find image with various extensions
                if os.path.exists(user_dir):
                    for ext in ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']:
                        test_path = os.path.join(user_dir, f"{expression}{ext}")
                        if os.path.exists(test_path):
                            img_path = test_path
                            break
                
                # Read and display the image
                if img_path and os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Display the image
                        axes[row_idx, col_idx].imshow(img_rgb)
                        axes[row_idx, col_idx].axis('off')
                        
                        # Add title with member name and expression
                        if col_idx == 0:
                            title = f'{user}\n{expression.capitalize()}'
                        else:
                            title = expression.capitalize()
                        axes[row_idx, col_idx].set_title(title, fontsize=12, fontweight='bold')
                        
                        print(f"  * {expression}: {os.path.basename(img_path)} ({img.shape[1]}x{img.shape[0]})")
                        images_found += 1
                    else:
                        print(f"  - {expression}: Could not read image")
                        axes[row_idx, col_idx].text(0.5, 0.5, 'Error\nReading Image', 
                                                    ha='center', va='center', fontsize=12, color='red')
                        axes[row_idx, col_idx].axis('off')
                        images_missing += 1
                else:
                    # If image doesn't exist, show placeholder
                    print(f"  - {expression}: Image not found")
                    axes[row_idx, col_idx].text(0.5, 0.5, 'Image\nNot Found', 
                                                ha='center', va='center', fontsize=12, color='gray')
                    axes[row_idx, col_idx].axis('off')
                    axes[row_idx, col_idx].set_title(f'{user} - {expression}', fontsize=10)
                    images_missing += 1
        
        plt.tight_layout()
        
        # Save if requested
        if save_output:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).resolve().parent.parent.parent
            output_path = project_root / 'reports' / 'sample_images.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to: {output_path}")
        
        # Only show plot in interactive mode
        try:
            plt.show(block=False)
            plt.pause(0.1)  # Brief pause to render
        except Exception:
            pass  # Skip if not in interactive environment
        
        plt.close()  # Close to free memory
        
        # Print summary
        print("\n" + "="*70)
        print("SAMPLE IMAGES SUMMARY")
        print("="*70)
        print(f"Images found: {images_found}")
        print(f"Images missing: {images_missing}")
        print(f"Total expected: {len(self.users) * len(expressions)}")
        print(f"Members: {', '.join(self.users)}")
        print(f"Expressions: {', '.join([e.capitalize() for e in expressions])}")
        print("="*70 + "\n")
        
        return images_found, images_missing


def main():
    """
    Main function to run the complete image processing pipeline.
    
    This implements the Image Data Collection and Processing requirements:
    1. Submit at least 3 facial images (neutral, smiling, surprised)
    2. Load and display sample pictures of each member
    3. Apply augmentations per image (rotation, flipping, grayscale, etc.)
    4. Extract and save image features to image_features.csv
    """
    # Initialize processor
    processor = ImageProcessor()
    
    # Step 1 & 2: Load and display sample pictures of each member
    print("\n" + "="*70)
    print("STEP 1 & 2: LOAD AND DISPLAY SAMPLE PICTURES OF EACH MEMBER")
    print("="*70)
    processor.display_sample_images(save_output=True)
    
    # Step 3 & 4: Process all images, apply augmentations, and extract features
    print("\n" + "="*70)
    print("STEP 3 & 4: APPLY AUGMENTATIONS AND EXTRACT FEATURES")
    print("="*70)
    df_features = processor.process_all_images()
    
    # Display summary statistics
    print("\n" + "="*50)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*50)
    print(f"\nDataset Statistics:")
    print(f"- Total samples: {len(df_features)}")
    print(f"- Users: {df_features['user'].nunique()}")
    print(f"- Expressions: {df_features['expression'].nunique()}")
    print(f"- Augmentation types: {df_features['augmentation'].nunique()}")
    print(f"\nSamples per user:")
    print(df_features['user'].value_counts())
    print(f"\nSamples per expression:")
    print(df_features['expression'].value_counts())
    
    # Save features to CSV
    processor.save_features_to_csv(df_features)
    
    print("\nImage processing pipeline completed successfully!")
    return df_features


if __name__ == "__main__":
    main()
