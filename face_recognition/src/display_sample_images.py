"""
Display Sample Images of Each Team Member

This script loads and displays sample images from each team member
showing their different facial expressions.

Usage:
    python display_sample_images.py
"""

import cv2
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import Config


def display_member_images(base_dir=None, save_output=False, output_path=None):
    """
    Display sample images of each team member
    
    Args:
        base_dir: Directory containing member image folders (default: Config.IMAGES_DIR)
        save_output: Whether to save the output figure
        output_path: Output file path for saved figure (default: Config.SAMPLE_IMAGES_REPORT)
    """
    # Use configuration with optional overrides
    base_image_dir = Path(base_dir) if base_dir else Config.IMAGES_DIR
    members = Config.get_users()
    expressions = Config.EXPRESSIONS
    output_file_path = Path(output_path) if output_path else Config.SAMPLE_IMAGES_REPORT
    
    print("="*70)
    print("SAMPLE IMAGES VIEWER")
    print("="*70)
    print(f"Image directory: {base_image_dir}")
    print(f"Team members: {', '.join(members)}\n")
    
    # Create a figure to display all images (4 members x 3 expressions)
    num_members = len(members)
    fig, axes = plt.subplots(num_members, 3, figsize=(15, num_members * 4))
    fig.suptitle('Sample Images from Each Team Member', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    images_found = 0
    images_missing = 0
    
    # Loop through each member and display their images
    for row_idx, member in enumerate(members):
        member_dir = base_image_dir / member
        print(f"Processing {member}:")
        
        for col_idx, expression in enumerate(expressions):
            # Try different extensions
            img_path = None
            for ext in ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']:
                test_path = member_dir / f"{expression}{ext}"
                if test_path.exists():
                    img_path = test_path
                    break
            
            # Read and display the image
            if img_path and img_path.exists():
                img = cv2.imread(str(img_path))
                
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Display the image
                    axes[row_idx, col_idx].imshow(img_rgb)
                    axes[row_idx, col_idx].axis('off')
                    
                    # Add title with member name and expression
                    if col_idx == 0:
                        title = f'{member}\n{expression.capitalize()}'
                    else:
                        title = expression.capitalize()
                    axes[row_idx, col_idx].set_title(title, fontsize=12, fontweight='bold')
                    
                    print(f"  * {expression}: {img_path.name} ({img.shape[1]}x{img.shape[0]})")
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
                axes[row_idx, col_idx].set_title(f'{member} - {expression}', fontsize=10)
                images_missing += 1
        
        print()
    
    plt.tight_layout()
    
    # Save if requested
    if save_output:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_file_path}\n")
    
    plt.show()
    
    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Images found: {images_found}")
    print(f"Images missing: {images_missing}")
    print(f"Total expected: {len(members) * len(expressions)}")
    print(f"Members: {', '.join(members)}")
    print(f"Expressions: {', '.join([e.capitalize() for e in expressions])}")
    print("="*70)
    
    return images_found, images_missing


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Display sample images of each team member',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Display images (default)
  python display_sample_images.py

  # Display and save to file
  python display_sample_images.py --save
  
  # Custom image directory
  python display_sample_images.py --image-dir /path/to/images
  
  # Save to custom location
  python display_sample_images.py --save --output /path/to/output.png
        '''
    )
    
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help=f'Directory containing member image folders (default: {Config.IMAGES_DIR})'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save the output figure to a file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=f'Output file path for saved figure (default: {Config.SAMPLE_IMAGES_REPORT})'
    )
    
    args = parser.parse_args()
    
    # Use Config defaults if not specified
    base_dir = Path(args.image_dir) if args.image_dir else Config.IMAGES_DIR
    output_path = Path(args.output) if args.output else Config.SAMPLE_IMAGES_REPORT
    
    try:
        display_member_images(
            base_dir=base_dir,
            save_output=args.save,
            output_path=output_path
        )
        return 0
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
