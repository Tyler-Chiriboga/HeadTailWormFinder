#!/usr/bin/env python3
"""
HeadTailWormFinder - Worm Annotation Tool

A PyQt5 application for annotating worm head and tail regions
using YOLOv7 for detection and SAM for segmentation.

Usage:
    python main.py [--folder PATH]

Requirements:
    - PyQt5
    - OpenCV
    - PyTorch with CUDA
    - segment-anything
    - openpyxl
"""
import os
# Fix Qt plugin conflict between OpenCV and PyQt5
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add YOLOv7 repo to path for model definitions (MUST be before importing models)
YOLOV7_REPO = PROJECT_ROOT / "yolov7_repo"
if YOLOV7_REPO.exists():
    sys.path.insert(0, str(YOLOV7_REPO))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HeadTailWormFinder - Worm Annotation Tool"
    )
    parser.add_argument(
        "--folder", "-f",
        type=str,
        help="Initial folder to load videos from"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU mode (disable GPU)"
    )
    return parser.parse_args()


def check_dependencies():
    """Check that required dependencies are installed."""
    missing = []
    
    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5")
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import openpyxl
    except ImportError:
        missing.append("openpyxl")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        return False
    
    return True


def check_gpu():
    """Check GPU availability and print info."""
    import torch
    
    print("\n=== GPU Information ===")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available: Yes")
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"         Memory: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA available: No (using CPU)")
    print("=" * 25 + "\n")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Import after dependency check
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Check GPU
    if not args.no_gpu:
        check_gpu()
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("HeadTailWormFinder")
    app.setOrganizationName("WormLab")
    
    # Set style
    app.setStyle("Fusion")
    
    # Import main window
    from ui.main_window import MainWindow
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Load initial folder if specified
    if args.folder:
        folder_path = Path(args.folder)
        if folder_path.exists():
            window._load_folder(str(folder_path))
        else:
            print(f"Warning: Folder not found: {args.folder}")
    
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
