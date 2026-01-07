"""
Main application window for HeadTailWormFinder.
"""
import sys
import json
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QAction, QToolBar, QStatusBar,
    QFileDialog, QMessageBox, QProgressDialog, QApplication,
    QDockWidget, QListWidget, QListWidgetItem, QLabel,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QSplitter, QPushButton, QComboBox, QCheckBox,
    QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QKeySequence, QIcon

from config.settings import (
    UI_SETTINGS, YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD,
    DEFAULT_VIDEO_FOLDER, SAM_DOWNLOAD_URL, SAM_CHECKPOINT_PATH,
    get_available_gpus, ensure_directories, CACHE_FILE, AUTO_SETTINGS,
    SAM_MODELS, WEIGHTS_DIR
)
from core.video_handler import VideoHandler
from core.annotation_manager import AnnotationManager, WormAnnotation
from core.excel_exporter import ExcelExporter, export_for_training
from ml_models.yolo_detector import YOLODetector, Detection
from ml_models.sam_segmenter import SAMSegmenter
from ui.frame_widget import FrameWidget
from ui.annotation_canvas import BoxType
from config.settings import get_best_sam_model


class ModelLoaderThread(QThread):
    """Thread for loading models in background."""
    
    progress = pyqtSignal(str, int)  # message, percent
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, load_yolo=True, load_sam=True, sam_model_type=None):
        super().__init__()
        self.load_yolo = load_yolo
        self.load_sam = load_sam
        self.sam_model_type = sam_model_type or get_best_sam_model()
        self.yolo_detector = None
        self.sam_segmenter = None
    
    def run(self):
        try:
            if self.load_yolo:
                self.progress.emit("Loading YOLO model...", 10)
                self.yolo_detector = YOLODetector(
                    weights_path=YOLO_MODEL_PATH,
                    conf_threshold=YOLO_CONFIDENCE_THRESHOLD
                )
                self.progress.emit("YOLO loaded", 50)
            
            if self.load_sam:
                self.progress.emit(f"Loading SAM model ({self.sam_model_type})...", 60)
                self.sam_segmenter = SAMSegmenter(model_type=self.sam_model_type)
                
                def download_progress(downloaded, total):
                    if total > 0:
                        percent = int(60 + (downloaded / total) * 30)
                        self.progress.emit(
                            f"Downloading SAM {self.sam_model_type}... {downloaded/1024/1024:.1f}MB",
                            percent
                        )
                
                self.sam_segmenter.load_model(download_progress)
                self.progress.emit(f"SAM ({self.sam_model_type}) loaded", 100)
            
            self.finished.emit(True, "Models loaded successfully")
            
        except Exception as e:
            self.finished.emit(False, str(e))


class MainWindow(QMainWindow):
    """
    Main application window.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.video_handler = VideoHandler()
        self.annotation_manager = AnnotationManager()
        self.yolo_detector: Optional[YOLODetector] = None
        self.sam_segmenter: Optional[SAMSegmenter] = None
        
        # Current state
        self._current_video_path: Optional[str] = None
        self._current_frame = None
        self._current_worm_id: Optional[int] = None
        self._models_loaded = False  # Track when models are ready
        self._pending_detection = False  # Track if we need to run detection after models load
        self._current_sam_type = 'vit_b'  # Default SAM model type
        
        # Ensure directories exist
        ensure_directories()
        
        # Setup UI
        self._setup_ui()
        self._setup_menus()
        self._setup_statusbar()
        self._connect_signals()
        
        # Load models in background
        QTimer.singleShot(100, self._load_models)
        
        # Auto-load last folder after a brief delay (to allow models to start loading)
        if AUTO_SETTINGS.get('auto_load_last_folder', True):
            QTimer.singleShot(200, self._auto_load_last_folder)
    
    def _load_cache(self) -> dict:
        """Load application cache."""
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
        return {}
    
    def _save_cache(self, cache: dict):
        """Save application cache."""
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _auto_load_last_folder(self):
        """Auto-load the last used folder from cache."""
        cache = self._load_cache()
        last_folder = cache.get('last_folder')
        
        if last_folder and Path(last_folder).exists():
            print(f"Auto-loading last folder from cache: {last_folder}")
            self._load_folder(last_folder)
    
    def _setup_ui(self):
        """Set up the main UI layout."""
        self.setWindowTitle(UI_SETTINGS['window_title'])
        self.resize(UI_SETTINGS['default_width'], UI_SETTINGS['default_height'])
        
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Video list
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center - Frame widget
        self.frame_widget = FrameWidget()
        splitter.addWidget(self.frame_widget)
        
        # Right panel - Annotation details
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([200, 900, 250])
    
    def _create_left_panel(self) -> QWidget:
        """Create left panel with video list."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Folder label
        self.folder_label = QLabel("No folder loaded")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)
        
        # Open folder button
        open_btn = QPushButton("📁 Open Folder...")
        open_btn.clicked.connect(self._open_folder)
        layout.addWidget(open_btn)
        
        # Video list
        layout.addWidget(QLabel("Videos:"))
        self.video_list = QListWidget()
        self.video_list.itemClicked.connect(self._on_video_selected)
        layout.addWidget(self.video_list)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create right panel with annotation details."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Current video info
        info_group = QGroupBox("Current Video")
        info_layout = QFormLayout(info_group)
        
        self.video_name_label = QLabel("-")
        info_layout.addRow("File:", self.video_name_label)
        
        self.video_resolution_label = QLabel("-")
        info_layout.addRow("Resolution:", self.video_resolution_label)
        
        layout.addWidget(info_group)
        
        # Annotations tree (hierarchical)
        annot_group = QGroupBox("Annotations")
        annot_layout = QVBoxLayout(annot_group)
        
        self.annotation_tree = QTreeWidget()
        self.annotation_tree.setHeaderLabels(["Item", "Status"])
        self.annotation_tree.setColumnWidth(0, 150)
        self.annotation_tree.itemClicked.connect(self._on_annotation_tree_clicked)
        self.annotation_tree.itemSelectionChanged.connect(self._on_annotation_selection_changed)
        annot_layout.addWidget(self.annotation_tree)
        
        # Show/hide controls
        show_layout = QHBoxLayout()
        self.show_all_btn = QPushButton("Show All")
        self.show_all_btn.clicked.connect(self._show_all_annotations)
        show_layout.addWidget(self.show_all_btn)
        
        self.show_selected_btn = QPushButton("Show Selected Only")
        self.show_selected_btn.clicked.connect(self._show_selected_only)
        show_layout.addWidget(self.show_selected_btn)
        annot_layout.addLayout(show_layout)
        
        # Batch segmentation controls
        seg_layout = QHBoxLayout()
        
        self.segment_worm_btn = QPushButton("✂️ Segment Worm")
        self.segment_worm_btn.setToolTip("Segment all boxes for selected worm")
        self.segment_worm_btn.clicked.connect(self._segment_selected_worm)
        self.segment_worm_btn.setEnabled(False)
        seg_layout.addWidget(self.segment_worm_btn)
        
        self.segment_all_btn = QPushButton("✂️ Segment All")
        self.segment_all_btn.setToolTip("Segment all boxes for all worms")
        self.segment_all_btn.clicked.connect(self._segment_all_worms)
        seg_layout.addWidget(self.segment_all_btn)
        
        annot_layout.addLayout(seg_layout)
        
        # Annotation actions
        btn_layout = QHBoxLayout()
        
        self.delete_annot_btn = QPushButton("Delete")
        self.delete_annot_btn.clicked.connect(self._delete_selected_annotation)
        self.delete_annot_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_annot_btn)
        
        annot_layout.addLayout(btn_layout)
        
        layout.addWidget(annot_group)
        
        # Detection settings
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.1, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(YOLO_CONFIDENCE_THRESHOLD)
        self.conf_spinbox.valueChanged.connect(self._on_confidence_changed)
        settings_layout.addRow("YOLO Confidence:", self.conf_spinbox)
        
        # Auto-detection toggle
        self.auto_detect_checkbox = QCheckBox("Auto-run detection")
        self.auto_detect_checkbox.setChecked(AUTO_SETTINGS.get('auto_run_detection', True))
        self.auto_detect_checkbox.setToolTip("Automatically run YOLO detection when loading new videos")
        self.auto_detect_checkbox.stateChanged.connect(self._on_auto_detect_changed)
        settings_layout.addRow(self.auto_detect_checkbox)
        
        layout.addWidget(settings_group)
        
        # SAM Model settings
        sam_group = QGroupBox("SAM Segmentation Model")
        sam_layout = QVBoxLayout(sam_group)
        
        self.sam_combo = QComboBox()
        for model_type, model_info in SAM_MODELS.items():
            self.sam_combo.addItem(model_info['name'], model_type)
        self.sam_combo.setCurrentIndex(0)  # Default to vit_b
        self.sam_combo.currentIndexChanged.connect(self._on_sam_model_changed)
        sam_layout.addWidget(self.sam_combo)
        
        self.sam_status_label = QLabel("Current: vit_b (loaded)")
        self.sam_status_label.setStyleSheet("color: green;")
        sam_layout.addWidget(self.sam_status_label)
        
        self.load_sam_btn = QPushButton("Load Selected SAM Model")
        self.load_sam_btn.clicked.connect(self._load_selected_sam_model)
        self.load_sam_btn.setEnabled(False)  # Disabled until different model selected
        sam_layout.addWidget(self.load_sam_btn)
        
        layout.addWidget(sam_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)
        
        self.total_worms_label = QLabel("0")
        stats_layout.addRow("Total Worms:", self.total_worms_label)
        
        self.complete_label = QLabel("0")
        stats_layout.addRow("Complete:", self.complete_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        # Export buttons
        export_btn = QPushButton("📊 Export to Excel")
        export_btn.clicked.connect(self._export_to_excel)
        layout.addWidget(export_btn)
        
        training_btn = QPushButton("🎓 Export Training Data")
        training_btn.clicked.connect(self._export_training_data)
        layout.addWidget(training_btn)
        
        return panel
    
    def _setup_menus(self):
        """Set up menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Folder...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save Annotations", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_annotations)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export to &Excel...", self)
        export_action.triggered.connect(self._export_to_excel)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        delete_action = QAction("&Delete Selected", self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self._delete_selected_annotation)
        edit_menu.addAction(delete_action)
        
        clear_action = QAction("&Clear All", self)
        clear_action.triggered.connect(self._clear_current_annotations)
        edit_menu.addAction(clear_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self._fit_to_window)
        view_menu.addAction(fit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        detect_action = QAction("Run &Detection", self)
        detect_action.setShortcut("D")
        detect_action.triggered.connect(self._run_detection)
        tools_menu.addAction(detect_action)
        
        batch_detect_action = QAction("Run Detection on All &Videos", self)
        batch_detect_action.setShortcut("Ctrl+D")
        batch_detect_action.triggered.connect(self._run_batch_detection)
        tools_menu.addAction(batch_detect_action)
        
        tools_menu.addSeparator()
        
        segment_action = QAction("Run &Segmentation", self)
        segment_action.setShortcut("S")
        segment_action.triggered.connect(self._run_segmentation_on_selected)
        tools_menu.addAction(segment_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)
    
    def _setup_statusbar(self):
        """Set up status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Model status labels
        self.yolo_status = QLabel("YOLO: Loading...")
        self.sam_status = QLabel("SAM: Loading...")
        self.gpu_status = QLabel(f"GPU: {', '.join(get_available_gpus())}")
        
        self.statusbar.addPermanentWidget(self.yolo_status)
        self.statusbar.addPermanentWidget(self.sam_status)
        self.statusbar.addPermanentWidget(self.gpu_status)
    
    def _connect_signals(self):
        """Connect signals between components."""
        # Frame widget signals
        self.frame_widget.next_video_requested.connect(self._next_video)
        self.frame_widget.prev_video_requested.connect(self._prev_video)
        self.frame_widget.next_folder_requested.connect(self._next_folder)
        self.frame_widget.prev_folder_requested.connect(self._prev_folder)
        self.frame_widget.run_detection_requested.connect(self._run_detection)
        self.frame_widget.segment_requested.connect(self._run_segmentation)
        self.frame_widget.box_drawn.connect(self._on_box_drawn)
        self.frame_widget.save_requested.connect(self._save_annotations)
        self.frame_widget.mask_accepted.connect(self._on_mask_accepted)
    
    def _load_models(self):
        """Load ML models in background."""
        self.statusbar.showMessage("Loading models...")
        
        # Create progress dialog
        progress = QProgressDialog("Loading models...", None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Auto-select best SAM model based on GPU memory
        best_sam = get_best_sam_model()
        self._current_sam_type = best_sam
        print(f"Auto-selected SAM model: {best_sam}")
        
        # Create loader thread with auto-selected model
        self.loader_thread = ModelLoaderThread(load_yolo=True, load_sam=True, sam_model_type=best_sam)
        
        def on_progress(msg, percent):
            progress.setLabelText(msg)
            progress.setValue(percent)
        
        def on_finished(success, message):
            progress.close()
            
            if success:
                self.yolo_detector = self.loader_thread.yolo_detector
                self.sam_segmenter = self.loader_thread.sam_segmenter
                
                self.yolo_status.setText("YOLO: ✓")
                self.yolo_status.setStyleSheet("color: green;")
                
                # Show which SAM model was loaded
                sam_type = self.loader_thread.sam_model_type
                self.sam_status.setText(f"SAM: ✓ ({sam_type})")
                self.sam_status.setStyleSheet("color: green;")
                
                # Update the SAM model selector to reflect loaded model
                for i in range(self.sam_combo.count()):
                    if self.sam_combo.itemData(i) == sam_type:
                        self.sam_combo.setCurrentIndex(i)
                        break
                
                self.statusbar.showMessage(f"Models loaded (SAM: {sam_type})", 3000)
                
                self._models_loaded = True
                print("Models loaded, checking for pending detection...")
                
                # Run pending detection if we loaded a video before models were ready
                if self._pending_detection and AUTO_SETTINGS.get('auto_run_detection', True):
                    print("Running pending detection!")
                    self._pending_detection = False
                    QTimer.singleShot(100, self._run_detection)
            else:
                self.yolo_status.setText("YOLO: ✗")
                self.yolo_status.setStyleSheet("color: red;")
                
                self.sam_status.setText("SAM: ✗")
                self.sam_status.setStyleSheet("color: red;")
                
                QMessageBox.warning(
                    self,
                    "Model Loading Error",
                    f"Failed to load models:\n{message}"
                )
        
        self.loader_thread.progress.connect(on_progress)
        self.loader_thread.finished.connect(on_finished)
        self.loader_thread.start()
    
    def _open_folder(self):
        """Open folder dialog to select video folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Video Folder",
            DEFAULT_VIDEO_FOLDER if Path(DEFAULT_VIDEO_FOLDER).exists() else str(Path.home())
        )
        
        if folder:
            self._load_folder(folder)
    
    def _load_folder(self, folder_path: str):
        """Load videos from a folder."""
        try:
            print(f"Loading folder: {folder_path}")
            count = self.video_handler.load_folder(folder_path, recursive=True)
            total_videos = self.video_handler.get_total_video_count()
            folder_count = self.video_handler.get_folder_count()
            print(f"Found {total_videos} videos in {folder_count} folders")
            
            if count == 0:
                QMessageBox.warning(
                    self,
                    "No Videos Found",
                    f"No AVI files found in:\n{folder_path}\n(searched recursively)"
                )
                return
            
            # Save folder to cache for next time
            cache = self._load_cache()
            cache['last_folder'] = folder_path
            self._save_cache(cache)
            
            # Update annotation manager
            self.annotation_manager.set_folder(folder_path)
            
            # Update UI - show root folder name
            self.folder_label.setText(f"📁 {Path(folder_path).name}")
            self.folder_label.setToolTip(folder_path)
            
            # Update folder position
            folder_info = self.video_handler.get_current_folder_info()
            if folder_info:
                self.frame_widget.set_folder_position(
                    self.video_handler.get_current_folder_index() + 1,
                    self.video_handler.get_folder_count(),
                    folder_info.relative_path or folder_info.name
                )
            
            # Populate video list for current folder
            self.video_list.clear()
            for idx, name in self.video_handler.get_video_list():
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, idx)
                self.video_list.addItem(item)
            
            # Load first video
            self._load_current_video()
            
            total_videos = self.video_handler.get_total_video_count()
            folder_count = self.video_handler.get_folder_count()
            self.statusbar.showMessage(
                f"Loaded {total_videos} videos in {folder_count} folders from {folder_path}", 
                5000
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Folder",
                f"Failed to load folder:\n{str(e)}"
            )
    
    def _load_current_video(self):
        """Load the current video frame."""
        video_path = self.video_handler.get_current_video_path()
        print(f"Loading video: {video_path}")
        if video_path is None:
            print("No video path!")
            return
        
        self._current_video_path = str(video_path)
        
        # Get video info
        info = self.video_handler.get_video_info()
        print(f"Video info: {info}")
        if info:
            self.video_name_label.setText(info.filename)
            self.video_resolution_label.setText(info.resolution)
        
        # Get first frame
        frame = self.video_handler.get_first_frame()
        print(f"Frame: {frame.shape if frame is not None else None}")
        if frame is not None:
            self._current_frame = frame
            self.frame_widget.set_frame(frame)
            print("Frame set on widget")
            
            # Update video position
            self.frame_widget.set_position(
                self.video_handler.get_current_index() + 1,
                self.video_handler.get_video_count()
            )
            
            # Update folder position
            folder_info = self.video_handler.get_current_folder_info()
            if folder_info:
                self.frame_widget.set_folder_position(
                    self.video_handler.get_current_folder_index() + 1,
                    self.video_handler.get_folder_count(),
                    folder_info.relative_path or folder_info.name
                )
            
            # Highlight in list
            self.video_list.setCurrentRow(self.video_handler.get_current_index())
            
            # Load existing annotations
            self._load_annotations_for_current_video()
            
            # Initialize annotations entry
            if info:
                self.annotation_manager.get_or_create_video_annotations(
                    self._current_video_path,
                    info.width,
                    info.height
                )
            
            # Auto-run detection if enabled
            existing_annotations = self.annotation_manager.get_all_worm_annotations(
                self._current_video_path
            )
            print(f"Existing annotations: {len(existing_annotations)}, models_loaded: {self._models_loaded}")
            if AUTO_SETTINGS.get('auto_run_detection', True) and len(existing_annotations) == 0:
                # Only auto-detect if there are no existing annotations
                if self._models_loaded:
                    print("Running detection immediately...")
                    QTimer.singleShot(100, self._run_detection)
                else:
                    # Mark that we need to run detection once models are loaded
                    print("Queuing pending detection...")
                    self._pending_detection = True
        
        self._update_statistics()
    
    def _load_annotations_for_current_video(self):
        """Load and display existing annotations for current video."""
        self.frame_widget.clear_annotations()
        self.annotation_tree.clear()
        
        if self._current_video_path is None:
            return
        
        annotations = self.annotation_manager.get_all_worm_annotations(
            self._current_video_path
        )
        
        for annot in annotations:
            # Add detection box
            if annot.detection_box:
                self.frame_widget.add_detection_box(
                    *annot.detection_box, annot.worm_id
                )
            
            # Add head box
            if annot.head_box:
                self.frame_widget.add_head_box(
                    *annot.head_box, annot.worm_id
                )
            
            # Add tail box
            if annot.tail_box:
                self.frame_widget.add_tail_box(
                    *annot.tail_box, annot.worm_id
                )
            
            # Load and display saved masks
            self._load_masks_for_annotation(annot)
            
            # Add to tree
            self._add_worm_to_tree(annot)
    
    def _load_masks_for_annotation(self, annot):
        """Load and display saved masks for an annotation."""
        import cv2
        import numpy as np
        
        # Load worm mask
        if annot.segmentation_mask_path:
            try:
                mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = (mask > 127).astype(np.uint8)
                    self.frame_widget.add_segmentation_mask(mask, worm_id=annot.worm_id, mask_type="worm")
            except Exception as e:
                print(f"Error loading worm mask: {e}")
        
        # Load head mask
        if annot.head_mask_path:
            try:
                mask = cv2.imread(annot.head_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = (mask > 127).astype(np.uint8)
                    self.frame_widget.add_segmentation_mask(mask, worm_id=annot.worm_id, mask_type="head")
            except Exception as e:
                print(f"Error loading head mask: {e}")
        
        # Load tail mask
        if annot.tail_mask_path:
            try:
                mask = cv2.imread(annot.tail_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = (mask > 127).astype(np.uint8)
                    self.frame_widget.add_segmentation_mask(mask, worm_id=annot.worm_id, mask_type="tail")
            except Exception as e:
                print(f"Error loading tail mask: {e}")

    def _next_video(self):
        """Navigate to next video."""
        self._autosave()
        if self.video_handler.next_video():
            self._load_current_video()
    
    def _prev_video(self):
        """Navigate to previous video."""
        self._autosave()
        if self.video_handler.previous_video():
            self._load_current_video()
    
    def _next_folder(self):
        """Navigate to next folder."""
        self._autosave()
        if self.video_handler.next_folder():
            self._on_folder_changed()
    
    def _prev_folder(self):
        """Navigate to previous folder."""
        self._autosave()
        if self.video_handler.previous_folder():
            self._on_folder_changed()
    
    def _on_folder_changed(self):
        """Handle folder navigation - update UI and load first video."""
        folder_info = self.video_handler.get_current_folder_info()
        if folder_info:
            self.folder_label.setText(f"📁 {folder_info.name}")
            self.folder_label.setToolTip(str(folder_info.path))
            
            # Update folder position
            self.frame_widget.set_folder_position(
                self.video_handler.get_current_folder_index() + 1,
                self.video_handler.get_folder_count(),
                folder_info.relative_path or folder_info.name
            )
            
            # Update video list for new folder
            self.video_list.clear()
            for idx, name in self.video_handler.get_video_list():
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, idx)
                self.video_list.addItem(item)
            
            # Load first video in folder
            self._load_current_video()
    
    def _autosave(self):
        """Auto-save annotations when navigating."""
        # First sync any moved/resized boxes back to annotation manager
        self._sync_box_positions()
        
        # Always try to save if we have any annotations for the current video
        if self._current_video_path and self._current_video_path in self.annotation_manager.annotations:
            # Force save
            if self.annotation_manager.save_annotations():
                self.statusbar.showMessage("Auto-saved annotations", 2000)
        elif self.annotation_manager.has_unsaved_changes():
            if self.annotation_manager.save_annotations():
                self.statusbar.showMessage("Auto-saved annotations", 2000)
    
    def _sync_box_positions(self):
        """Sync current box positions from canvas to annotation manager."""
        if self._current_video_path is None:
            return
        
        from ui.annotation_canvas import BoxType
        
        try:
            box_data = self.frame_widget.canvas.get_all_box_data()
            for data in box_data:
                worm_id = data['worm_id']
                box_type = data['box_type']
                coords = data['coords']
                
                if worm_id is None:
                    continue
                
                # Convert coords to tuple if needed
                if isinstance(coords, list):
                    coords = tuple(coords)
                
                if box_type == BoxType.HEAD:
                    self.annotation_manager.set_head_box(
                        self._current_video_path, worm_id, coords
                    )
                elif box_type == BoxType.TAIL:
                    self.annotation_manager.set_tail_box(
                        self._current_video_path, worm_id, coords
                    )
                elif box_type == BoxType.YOLO_DETECTION:
                    # Update detection box if moved
                    annot = self.annotation_manager.get_worm_annotation(
                        self._current_video_path, worm_id
                    )
                    if annot:
                        annot.detection_box = coords
                        self.annotation_manager._unsaved_changes = True
        except Exception as e:
            print(f"Error syncing box positions: {e}")
    
    def _on_video_selected(self, item: QListWidgetItem):
        """Handle video selection from list."""
        self._autosave()
        idx = item.data(Qt.UserRole)
        if self.video_handler.navigate_to(idx):
            self._load_current_video()
    
    def _run_detection(self):
        """Run YOLO detection on current frame."""
        if self.yolo_detector is None:
            QMessageBox.warning(
                self,
                "Model Not Loaded",
                "YOLO model is not loaded yet. Please wait."
            )
            return
        
        if self._current_frame is None:
            return
        
        self.statusbar.showMessage("Running detection...")
        QApplication.processEvents()
        
        try:
            detections = self.yolo_detector.detect(self._current_frame)
            
            # Clear existing detection boxes (keep head/tail)
            self.frame_widget.canvas.clear_annotations()
            
            first_annot = None
            first_box = None
            
            # Add detections
            for det in detections:
                # Create annotation
                annot = self.annotation_manager.add_worm_annotation(
                    self._current_video_path,
                    detection_box=det.bbox,
                    confidence=det.confidence
                )
                
                # Add to canvas
                box = self.frame_widget.add_detection_box(
                    det.x1, det.y1, det.x2, det.y2,
                    annot.worm_id
                )
                
                # Keep track of first worm
                if first_annot is None:
                    first_annot = annot
                    first_box = box
                
                # Add to tree
                self._add_worm_to_tree(annot)
            
            # Auto-select first worm and expand it in tree
            if first_annot and first_box:
                self._select_and_expand_worm(first_annot.worm_id, first_box)
            
            self.statusbar.showMessage(
                f"Detected {len(detections)} worms - Worm 1 selected", 3000
            )
            self._update_statistics()
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Detection Error",
                f"Detection failed:\n{str(e)}"
            )
    
    def _run_batch_detection(self):
        """Run YOLO detection on all videos in folder that don't have annotations."""
        if self.yolo_detector is None:
            QMessageBox.warning(
                self,
                "Model Not Loaded",
                "YOLO model is not loaded yet. Please wait."
            )
            return
        
        if not self._video_list:
            QMessageBox.information(
                self,
                "No Videos",
                "No videos loaded. Please open a folder first."
            )
            return
        
        # Find videos without annotations
        videos_to_process = []
        for video_path in self._video_list:
            annotations = self.annotation_manager.get_all_worm_annotations(video_path)
            if not annotations:
                videos_to_process.append(video_path)
        
        if not videos_to_process:
            QMessageBox.information(
                self,
                "All Videos Processed",
                f"All {len(self._video_list)} videos already have annotations."
            )
            return
        
        # Confirm
        result = QMessageBox.question(
            self,
            "Batch Detection",
            f"Run YOLO detection on {len(videos_to_process)} videos without annotations?\n\n"
            f"({len(self._video_list) - len(videos_to_process)} videos already have annotations)",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result != QMessageBox.Yes:
            return
        
        # Create progress dialog
        progress = QProgressDialog(
            "Running batch detection...",
            "Cancel",
            0,
            len(videos_to_process),
            self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        total_detections = 0
        processed = 0
        
        for i, video_path in enumerate(videos_to_process):
            if progress.wasCanceled():
                break
            
            progress.setValue(i)
            progress.setLabelText(f"Processing: {Path(video_path).name}")
            QApplication.processEvents()
            
            try:
                # Load first frame
                handler = VideoHandler()
                if handler.load_video(video_path):
                    frame = handler.get_frame(0)
                    if frame is not None:
                        # Run detection
                        detections = self.yolo_detector.detect(frame)
                        
                        # Add annotations
                        for det in detections:
                            self.annotation_manager.add_worm_annotation(
                                video_path,
                                detection_box=det.bbox,
                                confidence=det.confidence
                            )
                            total_detections += 1
                        
                        processed += 1
                    handler.release()
                    
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        
        progress.setValue(len(videos_to_process))
        
        # Save annotations
        self.annotation_manager.save_all()
        
        # Refresh current view
        if self._current_video_path:
            self._load_annotations_for_current_video()
        
        self._update_statistics()
        
        QMessageBox.information(
            self,
            "Batch Detection Complete",
            f"Processed {processed} videos.\n"
            f"Found {total_detections} total worms."
        )

    def _run_segmentation(self, x1: float, y1: float, x2: float, y2: float):
        """Run SAM segmentation on a bounding box."""
        if self.sam_segmenter is None:
            QMessageBox.warning(
                self,
                "Model Not Loaded",
                "SAM model is not loaded yet. Please wait."
            )
            return
        
        if self._current_frame is None:
            return
        
        self.statusbar.showMessage("Running segmentation...")
        QApplication.processEvents()
        
        try:
            # Set image
            self.sam_segmenter.set_image(self._current_frame)
            
            # Segment
            result = self.sam_segmenter.segment((x1, y1, x2, y2))
            
            if result:
                # Determine mask type from box type
                selected = self.frame_widget.canvas.get_selected_box()
                worm_id = selected.worm_id if selected else None
                
                if selected:
                    if selected.box_type == BoxType.HEAD:
                        mask_type = "head"
                    elif selected.box_type == BoxType.TAIL:
                        mask_type = "tail"
                    else:
                        mask_type = "worm"
                else:
                    mask_type = "worm"
                
                # Remove any existing mask for this worm/type before adding new one
                if worm_id:
                    self.frame_widget.canvas.remove_masks_by_worm_id(worm_id, mask_type)
                
                # Add mask to canvas as overlay with worm_id tracking
                self.frame_widget.add_segmentation_mask(result.mask, worm_id=worm_id, mask_type=mask_type)
                
                # Also set as editable mask for refinement
                self.frame_widget.set_editable_mask(result.mask)
                
                # Save mask for training
                if worm_id:
                    self.annotation_manager.save_segmentation_mask(
                        self._current_video_path,
                        worm_id,
                        result.mask,
                        mask_type=mask_type
                    )
                    
                    # Update tree to show the mask
                    self._update_annotation_tree()
                
                self.statusbar.showMessage(
                    f"Segmentation complete (score: {result.score:.2f}) - Use Paint/Erase to refine", 5000
                )
            else:
                self.statusbar.showMessage("Segmentation failed", 3000)
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Segmentation Error",
                f"Segmentation failed:\n{str(e)}"
            )
    
    def _run_segmentation_on_selected(self):
        """Run segmentation on the selected box."""
        selected = self.frame_widget.canvas.get_selected_box()
        if selected:
            coords = selected.get_coordinates()
            self._run_segmentation(*coords)
    
    def _segment_selected_worm(self):
        """Segment all boxes for the selected worm."""
        if self._current_worm_id is None or self._current_video_path is None:
            QMessageBox.information(self, "No Selection", "Please select a worm first.")
            return
        
        if self.sam_segmenter is None:
            QMessageBox.warning(self, "Model Not Loaded", "SAM model is not loaded yet.")
            return
        
        annot = self.annotation_manager.get_worm_annotation(
            self._current_video_path, self._current_worm_id
        )
        if annot is None:
            return
        
        self._segment_annotation_boxes(annot)
        self._update_annotation_tree()
        self.statusbar.showMessage(f"Segmented all boxes for Worm {self._current_worm_id}", 3000)
    
    def _segment_all_worms(self):
        """Segment all boxes for all worms in current video."""
        if self._current_video_path is None:
            return
        
        if self.sam_segmenter is None:
            QMessageBox.warning(self, "Model Not Loaded", "SAM model is not loaded yet.")
            return
        
        annotations = self.annotation_manager.get_all_worm_annotations(self._current_video_path)
        if not annotations:
            QMessageBox.information(self, "No Annotations", "No worm annotations to segment.")
            return
        
        # Count boxes to segment
        total_boxes = sum(
            (1 if a.detection_box else 0) + (1 if a.head_box else 0) + (1 if a.tail_box else 0)
            for a in annotations
        )
        
        if total_boxes == 0:
            QMessageBox.information(self, "No Boxes", "No boxes to segment.")
            return
        
        result = QMessageBox.question(
            self, "Segment All",
            f"Segment {total_boxes} boxes for {len(annotations)} worms?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result != QMessageBox.Yes:
            return
        
        progress = QProgressDialog("Segmenting...", "Cancel", 0, total_boxes, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        count = 0
        for annot in annotations:
            if progress.wasCanceled():
                break
            self._segment_annotation_boxes(annot, progress, count)
            count += (1 if annot.detection_box else 0) + (1 if annot.head_box else 0) + (1 if annot.tail_box else 0)
        
        progress.setValue(total_boxes)
        self._update_annotation_tree()
        self.statusbar.showMessage(f"Segmented all boxes for {len(annotations)} worms", 3000)
    
    def _segment_annotation_boxes(self, annot, progress=None, progress_offset=0):
        """Segment all boxes for a single annotation."""
        if self._current_frame is None or self.sam_segmenter is None:
            return
        
        self.sam_segmenter.set_image(self._current_frame)
        
        boxes_to_segment = []
        if annot.detection_box:
            boxes_to_segment.append(('worm', annot.detection_box))
        if annot.head_box:
            boxes_to_segment.append(('head', annot.head_box))
        if annot.tail_box:
            boxes_to_segment.append(('tail', annot.tail_box))
        
        for i, (mask_type, box) in enumerate(boxes_to_segment):
            if progress:
                if progress.wasCanceled():
                    return
                progress.setValue(progress_offset + i)
                progress.setLabelText(f"Segmenting {mask_type} for Worm {annot.worm_id}...")
                QApplication.processEvents()
            
            try:
                result = self.sam_segmenter.segment(box)
                if result:
                    # Remove existing mask overlay for this type
                    self.frame_widget.canvas.remove_masks_by_worm_id(annot.worm_id, mask_type)
                    
                    # Add new mask overlay
                    self.frame_widget.add_segmentation_mask(
                        result.mask, worm_id=annot.worm_id, mask_type=mask_type
                    )
                    
                    # Save mask
                    self.annotation_manager.save_segmentation_mask(
                        self._current_video_path,
                        annot.worm_id,
                        result.mask,
                        mask_type=mask_type
                    )
            except Exception as e:
                print(f"Error segmenting {mask_type} for worm {annot.worm_id}: {e}")

    def _on_mask_accepted(self, mask):
        """Handle mask accepted after editing."""
        import numpy as np
        
        if self._current_video_path is None:
            return
        
        # Get selected worm ID and box type
        selected = self.frame_widget.canvas.get_selected_box()
        worm_id = selected.worm_id if selected else None
        
        # Determine mask type from box type
        if selected:
            if selected.box_type == BoxType.HEAD:
                mask_type = "head"
            elif selected.box_type == BoxType.TAIL:
                mask_type = "tail"
            else:
                mask_type = "worm"
        else:
            mask_type = "worm"
        
        # Save the edited mask
        saved_worm_id = worm_id
        if worm_id:
            saved_path = self.annotation_manager.save_segmentation_mask(
                self._current_video_path,
                worm_id,
                mask,
                mask_type=mask_type
            )
            if saved_path:
                self.statusbar.showMessage(f"Saved {mask_type} mask for worm {worm_id}", 2000)
                # Update tree to show the mask
                self._update_annotation_tree()
            else:
                self.statusbar.showMessage(f"Failed to save mask for worm {worm_id}", 3000)
        else:
            # Try to find any worm ID from the annotations
            annotations = self.annotation_manager.get_all_worm_annotations(self._current_video_path)
            if annotations:
                saved_worm_id = annotations[0].worm_id
                saved_path = self.annotation_manager.save_segmentation_mask(
                    self._current_video_path,
                    saved_worm_id,
                    mask,
                    mask_type=mask_type
                )
                if saved_path:
                    self.statusbar.showMessage(f"Saved {mask_type} mask for worm {saved_worm_id}", 2000)
                    self._update_annotation_tree()
                else:
                    self.statusbar.showMessage("Failed to save mask", 3000)
            else:
                self.statusbar.showMessage("Mask accepted (not saved - no worm annotation found)", 3000)
                return
        
        # Advance to next annotation step after saving mask
        if saved_worm_id:
            self._advance_to_next_annotation_step(saved_worm_id, mask_type)
    
    def _on_box_drawn(self, box_type: str, x1: float, y1: float, x2: float, y2: float):
        """Handle new box drawn by user."""
        if self._current_video_path is None:
            return
        
        # Handle manual detection boxes (when YOLO misses a worm)
        if box_type == "detection":
            # Create new worm annotation with manual detection box
            annot = self.annotation_manager.add_worm_annotation(
                self._current_video_path,
                detection_box=(x1, y1, x2, y2),
                confidence=1.0  # Manual annotations get full confidence
            )
            
            # Add to canvas and get the box
            new_box = self.frame_widget.add_detection_box(x1, y1, x2, y2, annot.worm_id)
            
            # Select the new box so it can be segmented
            self.frame_widget.canvas._select_box(new_box)
            self.frame_widget.segment_btn.setEnabled(True)
            
            # Add to tree
            self._add_worm_to_tree(annot)
            
            # Switch back to select mode
            self.frame_widget.set_select_mode()
            
            self.statusbar.showMessage(f"Added manual worm detection (Worm {annot.worm_id}) - Click 'Segment Selected' to segment", 5000)
            self._update_statistics()
            return
        
        # Get or select worm for head/tail boxes
        if box_type == "head" or box_type == "tail":
            # Need to associate with a worm - use current worm if set
            worm_id = self._current_worm_id
            
            if worm_id is None:
                # Check if there's a selected box
                selected = self.frame_widget.canvas.get_selected_box()
                if selected and selected.worm_id:
                    worm_id = selected.worm_id
            
            if worm_id is None:
                # Create new worm annotation as fallback
                annot = self.annotation_manager.add_worm_annotation(
                    self._current_video_path
                )
                worm_id = annot.worm_id
                self._current_worm_id = worm_id
                
                # Add to tree
                self._add_worm_to_tree(annot)
            
            # Set head/tail box
            if box_type == "head":
                self.annotation_manager.set_head_box(
                    self._current_video_path, worm_id, (x1, y1, x2, y2)
                )
                new_box = self.frame_widget.add_head_box(x1, y1, x2, y2, worm_id)
                # Auto-advance to tail after drawing head
                self.frame_widget.set_tail_mode()
                self.statusbar.showMessage(f"Head box added for Worm {worm_id} - Now draw TAIL box (red)", 5000)
            else:
                self.annotation_manager.set_tail_box(
                    self._current_video_path, worm_id, (x1, y1, x2, y2)
                )
                new_box = self.frame_widget.add_tail_box(x1, y1, x2, y2, worm_id)
                # After tail, switch to select mode and move to next worm
                self.frame_widget.set_select_mode()
                self._advance_to_next_worm(worm_id)
            
            # Update tree
            self._update_annotation_tree()
            self._update_statistics()
            
            # Save cropped regions for training
            annot = self.annotation_manager.get_worm_annotation(
                self._current_video_path, worm_id
            )
            if annot and self._current_frame is not None:
                self.annotation_manager.save_cropped_regions(
                    self._current_video_path,
                    worm_id,
                    self._current_frame,
                    annot
                )
    
    def _on_annotation_tree_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle annotation tree item click."""
        data = item.data(0, Qt.UserRole)
        if data is None:
            return
        
        worm_id = data.get('worm_id')
        item_type = data.get('type', 'worm')
        
        self._current_worm_id = worm_id
        self.delete_annot_btn.setEnabled(True)
        self.segment_worm_btn.setEnabled(True)
        
        # Show only the selected item
        self._show_selected_item(worm_id, item_type)
        
        # If a mask item is clicked, load it for editing
        if item_type in ('worm_mask', 'head_mask', 'tail_mask'):
            self._load_mask_for_editing(worm_id, item_type)
        
        # Also select the corresponding box
        self._select_box_for_worm(worm_id, item_type)
    
    def _select_box_for_worm(self, worm_id: int, item_type: str):
        """Select the box corresponding to the worm and type."""
        target_box_type = None
        if item_type in ('head', 'head_mask'):
            target_box_type = BoxType.HEAD
        elif item_type in ('tail', 'tail_mask'):
            target_box_type = BoxType.TAIL
        elif item_type in ('detection', 'worm_mask', 'worm'):
            target_box_type = BoxType.YOLO_DETECTION
        
        for box in self.frame_widget.canvas._boxes:
            if box.worm_id == worm_id:
                if target_box_type is None or box.box_type == target_box_type:
                    self.frame_widget.canvas._select_box(box)
                    return
    
    def _select_and_expand_worm(self, worm_id: int, box=None):
        """Select a worm, expand it in tree, and prepare for annotation workflow."""
        self._current_worm_id = worm_id
        self.delete_annot_btn.setEnabled(True)
        self.segment_worm_btn.setEnabled(True)
        
        # Select the box if provided
        if box:
            self.frame_widget.canvas._select_box(box)
            self.frame_widget.segment_btn.setEnabled(True)
        
        # Find and expand the worm item in tree
        root = self.annotation_tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            data = item.data(0, Qt.UserRole)
            if data and data.get('worm_id') == worm_id:
                # Expand the worm item
                item.setExpanded(True)
                # Select the worm item
                self.annotation_tree.setCurrentItem(item)
                # Show all items for this worm
                self._show_all_for_worm(worm_id)
                break
    
    def _show_all_for_worm(self, worm_id: int):
        """Show all boxes and masks for a worm."""
        for box in self.frame_widget.canvas._boxes:
            if box.worm_id == worm_id:
                box.setVisible(True)
        for mask in self.frame_widget.canvas._masks:
            if mask.worm_id == worm_id:
                mask.setVisible(True)
    
    def _advance_to_next_annotation_step(self, worm_id: int, current_step: str):
        """
        Advance to the next annotation step after accepting a mask.
        Flow: worm mask -> head box -> tail box
        """
        annot = self.annotation_manager.get_worm_annotation(
            self._current_video_path, worm_id
        )
        if annot is None:
            return
        
        if current_step == "worm":
            # Next: draw head box
            self.frame_widget.set_head_mode()
            self.statusbar.showMessage(
                f"Worm {worm_id}: Draw HEAD box (green)", 5000
            )
        elif current_step == "head":
            # Next: draw tail box
            self.frame_widget.set_tail_mode()
            self.statusbar.showMessage(
                f"Worm {worm_id}: Draw TAIL box (red)", 5000
            )
        elif current_step == "tail":
            # Annotation complete for this worm - move to next worm
            self._advance_to_next_worm(worm_id)
    
    def _advance_to_next_worm(self, current_worm_id: int):
        """Move to the next worm after completing annotation for current one."""
        if self._current_video_path is None:
            return
        
        annotations = self.annotation_manager.get_all_worm_annotations(
            self._current_video_path
        )
        
        # Find next worm ID
        worm_ids = sorted([a.worm_id for a in annotations])
        try:
            current_idx = worm_ids.index(current_worm_id)
            if current_idx < len(worm_ids) - 1:
                next_worm_id = worm_ids[current_idx + 1]
                # Find the box for next worm
                for box in self.frame_widget.canvas._boxes:
                    if box.worm_id == next_worm_id and box.box_type == BoxType.YOLO_DETECTION:
                        self._select_and_expand_worm(next_worm_id, box)
                        self.statusbar.showMessage(
                            f"Worm {next_worm_id}: Press 'Segment Selected' or draw HEAD/TAIL boxes", 5000
                        )
                        return
            else:
                # All worms done
                self.statusbar.showMessage(
                    f"All worms annotated! Save with Ctrl+S", 5000
                )
                self.frame_widget.set_select_mode()
        except ValueError:
            pass

    def _load_mask_for_editing(self, worm_id: int, item_type: str):
        """Load a saved mask for editing."""
        import cv2
        import numpy as np
        
        if self._current_video_path is None:
            return
        
        annot = self.annotation_manager.get_worm_annotation(self._current_video_path, worm_id)
        if annot is None:
            return
        
        mask_path = None
        if item_type == 'worm_mask' and annot.segmentation_mask_path:
            mask_path = annot.segmentation_mask_path
        elif item_type == 'head_mask' and annot.head_mask_path:
            mask_path = annot.head_mask_path
        elif item_type == 'tail_mask' and annot.tail_mask_path:
            mask_path = annot.tail_mask_path
        
        if mask_path:
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = (mask > 127).astype(np.uint8)
                    self.frame_widget.set_editable_mask(mask)
                    self.statusbar.showMessage(f"Loaded {item_type.replace('_', ' ')} for editing - use Paint/Erase tools", 3000)
            except Exception as e:
                print(f"Error loading mask for editing: {e}")
    
    def _on_annotation_selection_changed(self):
        """Handle annotation tree selection change."""
        items = self.annotation_tree.selectedItems()
        if not items:
            self.delete_annot_btn.setEnabled(False)
            self.segment_worm_btn.setEnabled(False)
            self._current_worm_id = None
    
    def _show_selected_item(self, worm_id: int, item_type: str):
        """Show only the selected worm/head/tail."""
        # Hide all boxes first
        for box in self.frame_widget.canvas._boxes:
            box.setVisible(False)
        
        # Hide all masks
        for mask in self.frame_widget.canvas._masks:
            mask.setVisible(False)
        
        # Show only relevant items based on item_type
        for box in self.frame_widget.canvas._boxes:
            if box.worm_id == worm_id:
                if item_type == 'worm':
                    box.setVisible(True)
                elif item_type == 'head' and box.box_type.value == 'head':
                    box.setVisible(True)
                elif item_type == 'tail' and box.box_type.value == 'tail':
                    box.setVisible(True)
                elif item_type == 'detection' and box.box_type.value == 'yolo':
                    box.setVisible(True)
        
        # Show masks based on item_type
        for mask in self.frame_widget.canvas._masks:
            if mask.worm_id == worm_id:
                if item_type == 'worm':
                    # Show all masks for this worm
                    mask.setVisible(True)
                elif item_type == 'worm_mask' and mask.mask_type == 'worm':
                    mask.setVisible(True)
                elif item_type == 'head' and mask.mask_type == 'head':
                    mask.setVisible(True)
                elif item_type == 'head_mask' and mask.mask_type == 'head':
                    mask.setVisible(True)
                elif item_type == 'tail' and mask.mask_type == 'tail':
                    mask.setVisible(True)
                elif item_type == 'tail_mask' and mask.mask_type == 'tail':
                    mask.setVisible(True)
                elif item_type == 'detection' and mask.mask_type == 'worm':
                    mask.setVisible(True)
    
    def _show_all_annotations(self):
        """Show all annotations."""
        for box in self.frame_widget.canvas._boxes:
            box.setVisible(True)
        for mask in self.frame_widget.canvas._masks:
            mask.setVisible(True)
        self.statusbar.showMessage("Showing all annotations", 2000)
    
    def _show_selected_only(self):
        """Show only the selected annotation."""
        items = self.annotation_tree.selectedItems()
        if items:
            item = items[0]
            data = item.data(0, Qt.UserRole)
            if data:
                self._show_selected_item(data.get('worm_id'), data.get('type', 'worm'))
                self.statusbar.showMessage("Showing selected only", 2000)
    
    def _delete_selected_annotation(self):
        """Delete the selected annotation."""
        if self._current_worm_id is None or self._current_video_path is None:
            return
        
        result = QMessageBox.question(
            self,
            "Delete Annotation",
            f"Delete Worm {self._current_worm_id} annotation?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Remove from manager
            self.annotation_manager.delete_worm_annotation(
                self._current_video_path, self._current_worm_id
            )
            
            # Remove from canvas
            self.frame_widget.canvas.remove_boxes_by_worm_id(self._current_worm_id)
            
            # Update tree
            self._update_annotation_tree()
            
            self._current_worm_id = None
            self.delete_annot_btn.setEnabled(False)
            self._update_statistics()
    
    def _clear_current_annotations(self):
        """Clear all annotations for current video."""
        if self._current_video_path is None:
            return
        
        result = QMessageBox.question(
            self,
            "Clear Annotations",
            "Clear all annotations for this video?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Clear from canvas
            self.frame_widget.clear_annotations()
            
            # Clear from manager (create new empty entry)
            if self._current_video_path in self.annotation_manager.annotations:
                self.annotation_manager.annotations[self._current_video_path].annotations.clear()
            
            # Clear tree
            self.annotation_tree.clear()
            self._update_statistics()
    
    def _add_worm_to_tree(self, annot, expanded_worms=None, expanded_items=None):
        """Add a worm annotation to the tree widget."""
        if expanded_worms is None:
            expanded_worms = set()
        if expanded_items is None:
            expanded_items = {}
            
        status = "✓" if annot.is_complete() else "○"
        
        # Create worm item
        worm_item = QTreeWidgetItem([f"🐛 Worm {annot.worm_id}", status])
        worm_item.setData(0, Qt.UserRole, {'worm_id': annot.worm_id, 'type': 'worm'})
        
        # Expand by default for new items, or preserve state
        should_expand = annot.worm_id in expanded_worms or annot.worm_id not in expanded_items
        worm_item.setExpanded(should_expand)
        
        worm_expanded_children = expanded_items.get(annot.worm_id, set())
        
        # Add detection box child
        if annot.detection_box:
            det_status = "✓" if annot.segmentation_mask_path else "○"
            det_item = QTreeWidgetItem(["📦 Detection", det_status])
            det_item.setData(0, Qt.UserRole, {'worm_id': annot.worm_id, 'type': 'detection'})
            det_item.setExpanded('detection' in worm_expanded_children)
            worm_item.addChild(det_item)
            
            # Add mask info if exists
            if annot.segmentation_mask_path:
                mask_item = QTreeWidgetItem(["  🎭 Mask", "✓"])
                mask_item.setData(0, Qt.UserRole, {'worm_id': annot.worm_id, 'type': 'worm_mask'})
                det_item.addChild(mask_item)
        
        # Add head child
        head_status = "✓" if annot.head_box else "○"
        head_item = QTreeWidgetItem(["🟢 Head", head_status])
        head_item.setData(0, Qt.UserRole, {'worm_id': annot.worm_id, 'type': 'head'})
        head_item.setExpanded('head' in worm_expanded_children)
        worm_item.addChild(head_item)
        
        if annot.head_mask_path:
            mask_item = QTreeWidgetItem(["  🎭 Mask", "✓"])
            mask_item.setData(0, Qt.UserRole, {'worm_id': annot.worm_id, 'type': 'head_mask'})
            head_item.addChild(mask_item)
        
        # Add tail child
        tail_status = "✓" if annot.tail_box else "○"
        tail_item = QTreeWidgetItem(["🔴 Tail", tail_status])
        tail_item.setData(0, Qt.UserRole, {'worm_id': annot.worm_id, 'type': 'tail'})
        tail_item.setExpanded('tail' in worm_expanded_children)
        worm_item.addChild(tail_item)
        
        if annot.tail_mask_path:
            mask_item = QTreeWidgetItem(["  🎭 Mask", "✓"])
            mask_item.setData(0, Qt.UserRole, {'worm_id': annot.worm_id, 'type': 'tail_mask'})
            tail_item.addChild(mask_item)
        
        self.annotation_tree.addTopLevelItem(worm_item)
    
    def _update_annotation_tree(self):
        """Update annotation tree display while preserving expanded state."""
        # Save expanded state
        expanded_worms = set()
        expanded_items = {}  # worm_id -> set of expanded child types
        
        for i in range(self.annotation_tree.topLevelItemCount()):
            item = self.annotation_tree.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            if data:
                worm_id = data.get('worm_id')
                if item.isExpanded():
                    expanded_worms.add(worm_id)
                # Check children
                expanded_items[worm_id] = set()
                for j in range(item.childCount()):
                    child = item.child(j)
                    if child.isExpanded():
                        child_data = child.data(0, Qt.UserRole)
                        if child_data:
                            expanded_items[worm_id].add(child_data.get('type'))
        
        # Clear and rebuild
        self.annotation_tree.clear()
        
        if self._current_video_path is None:
            return
        
        annotations = self.annotation_manager.get_all_worm_annotations(
            self._current_video_path
        )
        
        for annot in annotations:
            self._add_worm_to_tree(annot, expanded_worms, expanded_items)
    
    def _update_statistics(self):
        """Update statistics display."""
        stats = self.annotation_manager.get_statistics()
        self.total_worms_label.setText(str(stats['total_worms']))
        self.complete_label.setText(str(stats['complete_annotations']))
    
    def _on_confidence_changed(self, value: float):
        """Handle confidence threshold change."""
        if self.yolo_detector:
            self.yolo_detector.set_confidence_threshold(value)
    
    def _on_auto_detect_changed(self, state: int):
        """Handle auto-detection toggle change."""
        AUTO_SETTINGS['auto_run_detection'] = (state == Qt.Checked)
        status = "enabled" if state == Qt.Checked else "disabled"
        self.statusbar.showMessage(f"Auto-detection {status}", 2000)
    
    def _on_sam_model_changed(self, index: int):
        """Handle SAM model selection change."""
        selected_type = self.sam_combo.currentData()
        current_type = getattr(self, '_current_sam_type', 'vit_b')
        
        # Enable load button if different from current
        self.load_sam_btn.setEnabled(selected_type != current_type)
    
    def _load_selected_sam_model(self):
        """Load the selected SAM model."""
        selected_type = self.sam_combo.currentData()
        model_info = SAM_MODELS[selected_type]
        
        checkpoint_path = WEIGHTS_DIR / model_info['checkpoint']
        
        # Check if we need to download
        if not checkpoint_path.exists():
            result = QMessageBox.question(
                self,
                "Download SAM Model",
                f"The {model_info['name']} needs to be downloaded.\n"
                f"Size: ~{model_info['size_mb']}MB\n\n"
                f"Download now?",
                QMessageBox.Yes | QMessageBox.No
            )
            if result != QMessageBox.Yes:
                return
        
        # Load the model
        self.statusbar.showMessage(f"Loading {model_info['name']}...")
        self.sam_status_label.setText(f"Loading {selected_type}...")
        self.sam_status_label.setStyleSheet("color: orange;")
        self.load_sam_btn.setEnabled(False)
        QApplication.processEvents()
        
        try:
            # Create new SAM segmenter with selected model type
            # Let the segmenter handle checkpoint path and download URL based on model type
            self.sam_segmenter = SAMSegmenter(
                model_type=selected_type
            )
            
            def download_progress(downloaded, total):
                if total > 0:
                    percent = int(downloaded / total * 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    self.statusbar.showMessage(
                        f"Downloading {selected_type}... {mb_downloaded:.0f}/{mb_total:.0f}MB ({percent}%)"
                    )
                    QApplication.processEvents()
            
            self.sam_segmenter.load_model(download_progress)
            
            self._current_sam_type = selected_type
            self.sam_status_label.setText(f"Current: {selected_type} (loaded)")
            self.sam_status_label.setStyleSheet("color: green;")
            self.statusbar.showMessage(f"Loaded {model_info['name']}", 3000)
            
            # Update SAM status in status bar
            self.sam_status.setText(f"SAM: ✓ ({selected_type})")
            self.sam_status.setStyleSheet("color: green;")
            
            # Save preference to cache
            cache = self._load_cache()
            cache['sam_model'] = selected_type
            self._save_cache(cache)
            
        except Exception as e:
            self.sam_status_label.setText(f"Error loading {selected_type}")
            self.sam_status_label.setStyleSheet("color: red;")
            self.load_sam_btn.setEnabled(True)
            QMessageBox.warning(
                self,
                "SAM Model Error",
                f"Failed to load SAM model:\n{str(e)}"
            )
    
    def _save_annotations(self):
        """Save annotations to JSON file."""
        # First sync any moved/resized boxes back to annotation manager
        self._sync_box_positions()
        
        if self.annotation_manager.save_annotations():
            self.statusbar.showMessage("Annotations saved", 3000)
        else:
            QMessageBox.warning(
                self,
                "Save Error",
                "Failed to save annotations"
            )
    
    def _export_to_excel(self):
        """Export annotations to Excel."""
        if not self.annotation_manager.annotations:
            QMessageBox.information(
                self,
                "No Annotations",
                "No annotations to export"
            )
            return
        
        # Get save path
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to Excel",
            str(self.annotation_manager.folder_path / "worm_annotations.xlsx"),
            "Excel Files (*.xlsx)"
        )
        
        if path:
            try:
                exporter = ExcelExporter(self.annotation_manager)
                exporter.export(path)
                
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Annotations exported to:\n{path}"
                )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Export Error",
                    f"Failed to export:\n{str(e)}"
                )
    
    def _export_training_data(self):
        """Export annotations for model training."""
        if not self.annotation_manager.annotations:
            QMessageBox.information(
                self,
                "No Annotations",
                "No annotations to export"
            )
            return
        
        try:
            stats = export_for_training(self.annotation_manager)
            
            QMessageBox.information(
                self,
                "Training Data Exported",
                f"Exported training data:\n"
                f"- Heads: {stats['heads_exported']}\n"
                f"- Tails: {stats['tails_exported']}\n"
                f"- Total annotations: {stats['total_exported']}\n\n"
                f"Data saved to: {self.annotation_manager.folder_path / 'training_data'}"
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Export Error",
                f"Failed to export training data:\n{str(e)}"
            )
    
    def _fit_to_window(self):
        """Fit view to window."""
        self.frame_widget.canvas.fitInView(
            self.frame_widget.canvas.scene.sceneRect(),
            Qt.KeepAspectRatio
        )
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About HeadTailWormFinder",
            "HeadTailWormFinder v1.0\n\n"
            "A tool for annotating worm head and tail regions\n"
            "using YOLOv7 detection and SAM segmentation.\n\n"
            "Features:\n"
            "• Load AVI videos from folders\n"
            "• YOLOv7 worm detection\n"
            "• SAM segmentation\n"
            "• Head/tail box annotation\n"
            "• Excel export\n"
            "• Training data export"
        )
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts."""
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "Navigation:\n"
            "  Left Arrow - Previous video\n"
            "  Right Arrow - Next video\n"
            "  Ctrl+Left - Previous folder\n"
            "  Ctrl+Right - Next folder\n\n"
            "Zoom:\n"
            "  Mouse wheel - Zoom in/out\n"
            "  Ctrl++ - Zoom in\n"
            "  Ctrl+- - Zoom out\n"
            "  Ctrl+0 - Reset to 100%\n"
            "  F - Fit to window\n"
            "  1 - Reset to 100%\n"
            "  Middle mouse drag - Pan\n\n"
            "Actions:\n"
            "  D - Run detection\n"
            "  Ctrl+D - Run batch detection on all videos\n"
            "  S - Run segmentation\n"
            "  Delete - Delete selected\n\n"
            "File:\n"
            "  Ctrl+O - Open folder\n"
            "  Ctrl+S - Save annotations\n"
            "  Ctrl+Q - Quit"
        )
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.annotation_manager.has_unsaved_changes():
            result = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved annotations.\nSave before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if result == QMessageBox.Save:
                self._save_annotations()
                event.accept()
            elif result == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
