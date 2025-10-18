"""
Utility module to load sample images for testing chicken disease classification.
Supports loading images from the test sample folder organized by disease class.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

# Try multiple possible locations for sample images
# 1. Production: relative to app directory
# 2. Development: local Downloads folder
SAMPLE_PATHS = [
    Path("sample_images"),  # Production: ./sample_images/
    Path("/app/sample_images"),  # Production Docker: /app/sample_images/
    Path(r"c:\Users\royan\Downloads\Dataset Feces\chicken_disease\test\Sample"),  # Development Windows
]

# Find the first existing path
SAMPLE_ROOT = None
for path in SAMPLE_PATHS:
    if path.exists():
        SAMPLE_ROOT = path
        break

# If none found, use the first one as default (for potential creation)
if SAMPLE_ROOT is None:
    SAMPLE_ROOT = SAMPLE_PATHS[0]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def list_samples() -> Dict[str, List[str]]:
    """
    Return dict: {class_name: [absolute_file_path, ...]} for available sample images.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping class names to lists of image file paths
    """
    out: Dict[str, List[str]] = {}
    
    if not SAMPLE_ROOT.exists():
        return out
    
    for sub in sorted(SAMPLE_ROOT.iterdir()):
        if sub.is_dir():
            files = [
                str(p) for p in sorted(sub.iterdir()) 
                if p.is_file() and p.suffix.lower() in VALID_EXTS
            ]
            if files:
                out[sub.name] = files
    
    return out


def get_sample_by_class_and_index(class_name: str, index: int = 0) -> Optional[str]:
    """
    Get a specific sample image by class name and index.
    
    Args:
        class_name: Name of the disease class (e.g., 'Coccidiosis', 'Healthy')
        index: Index of the image in the class folder (default: 0)
        
    Returns:
        Optional[str]: Path to the image file, or None if not found
    """
    samples = list_samples()
    
    if class_name in samples and 0 <= index < len(samples[class_name]):
        return samples[class_name][index]
    
    return None


def count_samples_by_class() -> Dict[str, int]:
    """
    Count the number of samples per class.
    
    Returns:
        Dict[str, int]: Dictionary mapping class names to sample counts
    """
    samples = list_samples()
    return {class_name: len(files) for class_name, files in samples.items()}


def get_all_classes() -> List[str]:
    """
    Get list of all available disease classes.
    
    Returns:
        List[str]: Sorted list of class names
    """
    samples = list_samples()
    return sorted(samples.keys())


if __name__ == "__main__":
    # Test the functions
    print("=" * 60)
    print("Sample Images Available:")
    print("=" * 60)
    
    samples = list_samples()
    if samples:
        for class_name, files in samples.items():
            print(f"\n📁 {class_name}: {len(files)} images")
            for i, file_path in enumerate(files[:3], 1):  # Show first 3 files
                print(f"   {i}. {Path(file_path).name}")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more")
    else:
        print(f"\n❌ No sample images found at: {SAMPLE_ROOT}")
        print(f"   Please ensure the folder structure exists with subdirectories:")
        print(f"   - Coccidiosis")
        print(f"   - Healthy")
        print(f"   - New Castle Disease")
        print(f"   - Salmonella")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    counts = count_samples_by_class()
    total = sum(counts.values())
    for class_name, count in counts.items():
        print(f"  {class_name}: {count} images")
    print(f"\n  Total: {total} images across {len(counts)} classes")
