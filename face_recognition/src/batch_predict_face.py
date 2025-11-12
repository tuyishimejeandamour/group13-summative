"""
Batch Face Recognition Script - Predict multiple images at once

This script allows you to pass multiple image paths or a directory
and get face recognition predictions for all images.

Usage:
    python batch_predict_face.py <image_path1> <image_path2> ...
    python batch_predict_face.py --directory <dir_path>
    
Example:
    python batch_predict_face.py img1.jpg img2.jpg img3.jpg
    python batch_predict_face.py --directory ../data/images/test/
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path
from predict_face import EnhancedFaceRecognitionPredictor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import Config


def find_images_in_directory(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Find all image files in a directory
    
    Args:
        directory: Directory path to search
        extensions: Tuple of valid image extensions
        
    Returns:
        List of image file paths
    """
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).rglob(f'*{ext}'))
        image_files.extend(Path(directory).rglob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def batch_predict(image_paths, model_dir='../models', save_results=False, output_file='predictions.csv'):
    """
    Predict user identity for multiple images
    
    Args:
        image_paths: List of image file paths
        model_dir: Directory containing trained model
        save_results: Whether to save results to CSV
        output_file: Output CSV filename
        
    Returns:
        DataFrame with prediction results
    """
    print(f"\n{'='*80}")
    print(f"BATCH FACE RECOGNITION")
    print(f"{'='*80}")
    print(f"\nTotal images to process: {len(image_paths)}")
    
    # Initialize predictor
    predictor = EnhancedFaceRecognitionPredictor(model_dir=model_dir)
    
    # Get users dynamically from Config
    users = Config.get_users()
    
    # Store results
    results = []
    
    # Process each image
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n{'─'*80}")
        print(f"Processing image {idx}/{len(image_paths)}: {os.path.basename(image_path)}")
        print(f"{'─'*80}")
        
        try:
            # Make prediction
            result = predictor.predict(str(image_path), show_probabilities=False)
            
            # Store result - dynamically create probability fields for each user
            result_dict = {
                'image_path': str(image_path),
                'image_name': os.path.basename(image_path),
                'predicted_user': result['predicted_user'],
                'confidence': result['confidence']
            }
            
            # Add probability for each user dynamically
            for user in users:
                result_dict[f'{user.lower()}_prob'] = result['probabilities'].get(user, 0.0)
            
            results.append(result_dict)
            
            print(f"Success: {result['predicted_user']} ({result['confidence']:.1f}% confidence)")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            
            # Store error result - dynamically create probability fields for each user
            error_dict = {
                'image_path': str(image_path),
                'image_name': os.path.basename(image_path),
                'predicted_user': 'ERROR',
                'confidence': 0.0
            }
            
            # Add zero probability for each user dynamically
            for user in users:
                error_dict[f'{user.lower()}_prob'] = 0.0
            
            error_dict['error'] = str(e)
            results.append(error_dict)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Display summary
    print(f"\n{'='*80}")
    print(f"BATCH PREDICTION SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal images processed: {len(results)}")
    
    successful = len(df_results[df_results['predicted_user'] != 'ERROR'])
    failed = len(df_results[df_results['predicted_user'] == 'ERROR'])
    
    print(f"Successful predictions: {successful}")
    print(f"Failed predictions: {failed}")
    
    if successful > 0:
        print(f"\nPrediction Distribution:")
        for user in users:
            count = len(df_results[df_results['predicted_user'] == user])
            print(f"  {user}: {count} images")
        
        print(f"\nAverage Confidence: {df_results[df_results['predicted_user'] != 'ERROR']['confidence'].mean():.2f}%")
    
    # Save to CSV if requested
    if save_results:
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    # Display results table
    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS TABLE")
    print(f"{'='*80}\n")
    
    display_df = df_results[['image_name', 'predicted_user', 'confidence']].copy()
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2f}%")
    print(display_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    
    return df_results


def main():
    """Main function for batch face recognition"""
    
    parser = argparse.ArgumentParser(
        description='Batch Face Recognition - Predict multiple images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Predict multiple images
  python batch_predict_face.py img1.jpg img2.jpg img3.jpg
  
  # Predict all images in a directory
  python batch_predict_face.py --directory ../data/images/test/
  
  # Save results to CSV
  python batch_predict_face.py --directory test_images/ --save results.csv
        '''
    )
    
    parser.add_argument(
        'images',
        nargs='*',
        help='Image file paths to predict'
    )
    
    parser.add_argument(
        '--directory', '-d',
        type=str,
        help='Directory containing images to predict'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help=f'Directory containing trained model files (default: {Config.MODELS_DIR})'
    )
    
    parser.add_argument(
        '--save', '-s',
        type=str,
        metavar='OUTPUT_FILE',
        help='Save results to CSV file'
    )
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    
    if args.directory:
        # Get all images from directory
        print(f"Searching for images in: {args.directory}")
        image_paths = find_images_in_directory(args.directory)
        if not image_paths:
            print(f"ERROR: No image files found in {args.directory}")
            return 1
        print(f"Found {len(image_paths)} images")
    
    if args.images:
        # Add individual image paths
        for img_path in args.images:
            if os.path.exists(img_path):
                image_paths.append(Path(img_path))
            else:
                print(f"WARNING: Image not found: {img_path}")
    
    if not image_paths:
        print("ERROR: No image paths provided. Use --help for usage information.")
        parser.print_help()
        return 1
    
    try:
        # Use Config default if model_dir not specified
        model_dir = args.model_dir if args.model_dir else str(Config.MODELS_DIR)
        
        # Run batch prediction
        batch_predict(
            image_paths,
            model_dir=model_dir,
            save_results=bool(args.save),
            output_file=args.save if args.save else 'predictions.csv'
        )
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
