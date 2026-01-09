# HeadTailWormFinder 🪱

A comprehensive worm annotation tool for scientific research, featuring automated detection using YOLOv7 and precise segmentation using SAM (Segment Anything Model). Available as both a desktop PyQt5 application and a web-based interface.

## Features

- **Automated Worm Detection**: Uses custom-trained YOLOv7 model to detect worms in video frames
- **Precise Segmentation**: SAM (Segment Anything Model) for pixel-perfect worm segmentation
- **Health Classification**: CNN-based health scoring with automatic Healthy/Leaky classification
- **Head/Tail Annotation**: Manual annotation tools for marking worm head and tail positions (boxes or lines)
- **Dual Interface**: Desktop (PyQt5) and Web (FastAPI) versions
- **Multi-GPU Support**: Utilizes multiple GPUs for parallel model inference
- **Batch Processing**: Process entire video folders or projects with skip options for QC'd/detected videos
- **Quality Control (QC) Tracking**: Mark videos as reviewed and track QC progress across project
- **Auto-Detection**: Automatically runs detection when navigating to new folders
- **Export Options**:
  - **Excel/CSV**: Spreadsheets with annotations and health statistics (project-level and per-chip)
  - **Training Datasets**: YOLO format exports for detection and segmentation model training
    - Configurable train/val split (by video to prevent data leakage)
    - YOLO Detection format (bounding boxes)
    - YOLOv8 Segmentation format (polygon masks)
    - Generates ready-to-use dataset with `data.yaml`

## System Requirements

- **OS**: Linux (Ubuntu 20.04+), Windows 10+, or macOS
- **Python**: 3.9+
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3080+ or RTX 4090)
- **VRAM**: Minimum 8GB (16GB+ recommended for SAM vit_h model)
- **RAM**: 16GB minimum, 32GB recommended

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/HeadTailWormFinder.git
cd HeadTailWormFinder
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

#### YOLOv7 Custom Weights
Place your custom YOLOv7 weights at the configured path (see \`config/settings.py\`).

#### SAM Weights
SAM weights will be automatically downloaded on first run, or you can manually download:

```bash
mkdir -p weights
# SAM ViT-H (best quality, ~2.5GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O weights/sam_vit_h_4b8939.pth

# SAM ViT-L (balanced, ~1.2GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O weights/sam_vit_l_0b3195.pth

# SAM ViT-B (fastest, ~375MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O weights/sam_vit_b_01ec64.pth
```

## Usage

### Desktop Application (PyQt5)

```bash
python main.py
```

**Keyboard Shortcuts:**
| Key | Action |
|-----|--------|
| \`D\` | Run YOLO detection on current frame |
| \`S\` | Run SAM segmentation on selected worm |
| \`H\` | Switch to Head drawing mode |
| \`T\` | Switch to Tail drawing mode |
| \`Ctrl+S\` | Save annotations |
| \`←/→\` | Navigate between videos |
| \`F\` | Fit image to window |
| \`+/-\` | Zoom in/out |

### Web Application (FastAPI)

```bash
# Start the server
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 in your browser.

**Default Login:**
- Username: \`admin\`
- Password: \`wormadmin123\`

**Web Keyboard Shortcuts:**
| Key | Action |
|-----|--------|
| \`D\` | Run detection |
| \`S\` | Select mode |
| \`H\` | Head drawing mode |
| \`T\` | Tail drawing mode |
| \`Ctrl+S\` | Save |
| \`←/→\` | Navigate videos |
| \`+/-\` | Zoom |

## Project Structure

```
HeadTailWormFinder/
├── main.py                 # Desktop app entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── config/
│   └── settings.py        # Configuration settings
│
├── core/
│   ├── video_handler.py   # Video loading and navigation
│   ├── annotation_manager.py  # Annotation storage and management
│   └── excel_exporter.py  # Export functionality
│
├── ml_models/
│   ├── yolo_detector.py   # YOLOv7 detection wrapper
│   └── sam_segmenter.py   # SAM segmentation wrapper
│
├── ui/
│   ├── main_window.py     # PyQt5 main window
│   ├── frame_widget.py    # Video frame display widget
│   └── annotation_canvas.py  # Drawing canvas for annotations
│
├── web/
│   ├── app.py             # FastAPI application
│   ├── auth.py            # Authentication system
│   ├── users.json         # User database
│   └── static/
│       ├── index.html     # Main web interface
│       └── login.html     # Login page
│
├── weights/               # Model weights directory
│   └── sam_vit_h_4b8939.pth
│
└── annotations/           # Saved annotations directory
```

## Configuration

Edit \`config/settings.py\` to customize:

```python
# Model paths
YOLO_MODEL_PATH = "/path/to/your/yolov7/weights.pt"
SAM_CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"

# Detection settings
YOLO_CONFIDENCE_THRESHOLD = 0.5

# GPU settings
YOLO_DEVICE = "cuda:0"
SAM_DEVICE = "cuda:1"  # Use second GPU if available
```

For web app, edit \`web/app.py\`:

```python
# Default dataset path (auto-loads on startup)
DEFAULT_DATASET_PATH = "/path/to/your/videos"
```

## API Reference (Web Version)

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/api/auth/login\` | POST | Login with username/password |
| \`/api/auth/logout\` | POST | Logout current session |
| \`/api/auth/me\` | GET | Get current user info |

### Video Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/api/open_folder\` | POST | Open folder with videos |
| \`/api/videos\` | GET | List all videos |
| \`/api/video/{index}\` | GET | Select and load video |
| \`/api/frame\` | GET | Get current frame with annotations |
| \`/api/navigate/{direction}\` | POST | Navigate next/prev video |

### Detection & Segmentation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect` | POST | Run YOLO detection on current video |
| `/api/detect/batch` | POST | Batch detection (folder or project scope) |
| `/api/detect/preview` | GET | Preview stats for batch detection |
| `/api/segment` | POST | Run SAM segmentation on selected worm |

### Annotations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/annotation/head` | POST | Set head box for worm |
| `/api/annotation/tail` | POST | Set tail box for worm |
| `/api/annotation/{worm_id}` | DELETE | Delete worm annotation |
| `/api/annotation/{worm_id}/censor` | POST | Toggle censored status |
| `/api/qc/toggle` | POST | Toggle QC complete status for video |
| `/api/save` | POST | Save all annotations |

### Export

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/export/summary` | GET | Get export statistics for modal display |
| `/api/export/excel` | GET | Export to Excel (project/chip/both modes) |
| `/api/export/training` | GET | Export YOLO training dataset (detect/segment/both) |
| `/api/export/training/preview` | GET | Preview training export statistics |

### Admin (requires admin role)

| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/api/admin/users\` | GET | List all users |
| \`/api/admin/users\` | POST | Create new user |
| \`/api/admin/users/{username}\` | DELETE | Delete user |

## Annotation Workflow

### 1. Load Videos
- Desktop: File → Open Folder, or click "📁 Open Folder"
- Web: Click "📁 Open Folder" and enter path
- Auto-detection runs automatically when navigating to new folders

### 2. Run Detection
- Click "🔍 Detect" or press \`D\` for single video
- Click "🚀 Batch Detect" to process multiple videos at once
- Options to skip already QC'd videos or those with existing detections
- YOLO will detect all worms in the frame
- Each worm gets a blue bounding box with ID

### 3. Select a Worm
- Click on a worm's bounding box
- Or click on the worm in the annotation tree

### 4. Annotate Head and Tail
- Press \`H\` or click "🟢 Head" to enter head mode
- Draw a box around the worm's head (or line with \`L\` toggle)
- Press \`T\` or click "🔴 Tail" to enter tail mode
- Draw a box around the worm's tail (or line with \`L\` toggle)

### 5. Run Segmentation (Optional)
- Select a worm and click "✂️ Segment"
- SAM will generate a pixel-perfect mask
- Masks are shown as colored overlays
- Use brush tools to refine masks if needed

### 6. Review Health Classification
- Each worm is automatically scored by CNN classifier
- **Healthy**: Score < 0.5 (shown in green)
- **Leaky**: Score ≥ 0.5 (shown in red)
- Mark problematic worms as **Censored** to exclude from analysis

### 7. Mark as QC'd
- Click "✅ Mark as QC'd" when you've verified all annotations
- QC'd videos show a green checkmark in the sidebar
- QC status is tracked in exports and can filter batch operations

### 8. Save Annotations
- Press \`Ctrl+S\` or click "💾 Save"
- Annotations are saved as JSON in the \`annotations/\` folder

### 9. Navigate to Next Video
- Press \`→\` or click "Next ▶"
- Annotations are auto-saved when navigating

### 10. Export Data
- Click "📊 Export" to open export modal
- **Excel/CSV Export**: Spreadsheet with all annotations and statistics
  - Project-level: Single file with all data
  - Per-chip: Separate file for each chip/folder
  - Includes summary sheets with health statistics
- **Training Dataset Export**: YOLO format for AI training
  - YOLO Detection: Bounding boxes (class x_center y_center width height)
  - YOLO Segmentation: Polygon masks (class x1 y1 x2 y2 ... xn yn)
  - Configurable train/val split (80/20 default)
  - Filter options: QC'd only, skip censored, require masks
  - Generates \`data.yaml\` and folder structure for training

## Annotation File Format

Annotations are stored as JSON files:

```json
{
  "video_path": "/path/to/video.avi",
  "frame_number": 0,
  "qc_complete": true,
  "annotations": [
    {
      "worm_id": 1,
      "detection_box": [x1, y1, x2, y2],
      "head_box": [x1, y1, x2, y2],
      "tail_box": [x1, y1, x2, y2],
      "head_line": [x1, y1, x2, y2],
      "tail_line": [x1, y1, x2, y2],
      "confidence": 0.95,
      "health_score": 0.23,
      "censored": false,
      "segmentation_mask_path": "masks/video_worm1_worm.png",
      "head_mask_path": "masks/video_worm1_head.png",
      "tail_mask_path": "masks/video_worm1_tail.png"
    }
  ]
}
```

### Field Descriptions:
- **qc_complete**: Whether the video has been reviewed and marked as QC'd
- **detection_box**: YOLO-detected bounding box [x1, y1, x2, y2]
- **head_box/tail_box**: Manual head/tail bounding boxes
- **head_line/tail_line**: Manual head/tail line annotations [x1, y1, x2, y2]
- **confidence**: YOLO detection confidence (0-1)
- **health_score**: CNN health classification score (0-1, ≥0.5 = Leaky, <0.5 = Healthy)
- **censored**: Whether worm should be excluded from analysis
- **segmentation_mask_path**: Path to SAM-generated worm mask PNG

## Training Dataset Format

When exporting for YOLO training, the following structure is generated:

```
training_dataset/
├── detect/                    # Detection format
│   ├── images/
│   │   ├── train/            # Training images
│   │   └── val/              # Validation images
│   ├── labels/
│   │   ├── train/            # Training labels (YOLO format)
│   │   └── val/              # Validation labels
│   └── data.yaml             # Dataset configuration
│
├── segment/                   # Segmentation format
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/               # Polygon annotations
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
│
└── README.md                  # Usage instructions
```

**Detection Label Format** (per line): `class_id x_center y_center width height`
- All values normalized to 0-1
- Example: `0 0.523 0.412 0.156 0.089`

**Segmentation Label Format** (per line): `class_id x1 y1 x2 y2 ... xn yn`
- Polygon points, all normalized to 0-1
- Example: `0 0.45 0.38 0.52 0.35 0.58 0.41 ...`

## Troubleshooting

### CUDA Out of Memory
- Use a smaller SAM model (vit_b instead of vit_h)
- Reduce batch size in settings
- Close other GPU applications

### Models Not Loading
- Check that weight files exist at configured paths
- Verify CUDA is properly installed: \`nvidia-smi\`
- Check Python has access to GPU: \`python -c "import torch; print(torch.cuda.is_available())"\`

### Web App Authentication Issues
- Clear browser cookies and localStorage
- Check that \`web/users.json\` exists and is readable
- Default credentials: admin / wormadmin123

### Slow Performance
- Enable multi-GPU if available
- Use SSD storage for video files
- Reduce video resolution if possible

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black .

# Lint
flake8 .
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{headtailwormfinder,
  title = {HeadTailWormFinder: Automated Worm Annotation Tool},
  year = {2026},
  url = {https://github.com/yourusername/HeadTailWormFinder}
}
```

## Acknowledgments

- [YOLOv7](https://github.com/WongKinYiu/yolov7) - Real-time object detection
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) - Image segmentation
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - Desktop UI framework
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
