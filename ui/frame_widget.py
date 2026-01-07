"""
Frame widget combining canvas with navigation and mode controls.
"""
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QButtonGroup, QRadioButton, QGroupBox,
    QSlider, QSpinBox, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal

from ui.annotation_canvas import AnnotationCanvas, AnnotationMode, BoxType


class FrameWidget(QWidget):
    """
    Widget for displaying video frame with annotation tools.
    
    Signals:
        next_video_requested: User wants next video
        prev_video_requested: User wants previous video
        next_folder_requested: User wants next folder
        prev_folder_requested: User wants previous folder
        run_detection_requested: User wants to run YOLO detection
        segment_requested: User wants to segment selection (x1, y1, x2, y2)
        box_drawn: A box was drawn (type, x1, y1, x2, y2)
        save_requested: User wants to save annotations
        mask_accepted: User accepted edited mask (mask_array)
    """
    
    next_video_requested = pyqtSignal()
    prev_video_requested = pyqtSignal()
    next_folder_requested = pyqtSignal()
    prev_folder_requested = pyqtSignal()
    run_detection_requested = pyqtSignal()
    segment_requested = pyqtSignal(float, float, float, float)
    box_drawn = pyqtSignal(str, float, float, float, float)
    save_requested = pyqtSignal()
    mask_accepted = pyqtSignal(object)  # numpy array
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Top toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # Canvas
        self.canvas = AnnotationCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, stretch=1)
        
        # Bottom navigation bar
        nav_bar = self._create_navigation_bar()
        layout.addWidget(nav_bar)
    
    def _create_toolbar(self) -> QWidget:
        """Create the annotation toolbar."""
        toolbar = QFrame()
        toolbar.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Mode selection
        mode_group = QGroupBox("Annotation Mode")
        mode_layout = QHBoxLayout(mode_group)
        
        self.mode_button_group = QButtonGroup(self)
        
        self.select_mode_btn = QRadioButton("Select")
        self.select_mode_btn.setChecked(True)
        self.mode_button_group.addButton(self.select_mode_btn, 0)
        mode_layout.addWidget(self.select_mode_btn)
        
        self.head_mode_btn = QRadioButton("Head")
        self.head_mode_btn.setStyleSheet("color: green; font-weight: bold;")
        self.mode_button_group.addButton(self.head_mode_btn, 1)
        mode_layout.addWidget(self.head_mode_btn)
        
        self.tail_mode_btn = QRadioButton("Tail")
        self.tail_mode_btn.setStyleSheet("color: red; font-weight: bold;")
        self.mode_button_group.addButton(self.tail_mode_btn, 2)
        mode_layout.addWidget(self.tail_mode_btn)
        
        self.detection_mode_btn = QRadioButton("Add Worm")
        self.detection_mode_btn.setStyleSheet("color: blue; font-weight: bold;")
        self.detection_mode_btn.setToolTip("Manually draw a worm detection box if YOLO missed it")
        self.mode_button_group.addButton(self.detection_mode_btn, 3)
        mode_layout.addWidget(self.detection_mode_btn)
        
        layout.addWidget(mode_group)
        
        layout.addSpacing(20)
        
        # Action buttons
        self.detect_btn = QPushButton("🔍 Detect Worms")
        self.detect_btn.setToolTip("Run YOLO detection on current frame")
        layout.addWidget(self.detect_btn)
        
        self.segment_btn = QPushButton("✂️ Segment Selected")
        self.segment_btn.setToolTip("Run SAM segmentation on selected box")
        self.segment_btn.setEnabled(False)
        layout.addWidget(self.segment_btn)
        
        layout.addSpacing(20)
        
        # Clear buttons
        self.clear_masks_btn = QPushButton("Clear Masks")
        self.clear_masks_btn.setToolTip("Remove segmentation masks")
        layout.addWidget(self.clear_masks_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setToolTip("Remove all annotations")
        layout.addWidget(self.clear_all_btn)
        
        layout.addSpacing(20)
        
        # Mask editing controls
        mask_group = QGroupBox("Mask Edit")
        mask_layout = QHBoxLayout(mask_group)
        
        self.paint_mask_btn = QRadioButton("Paint")
        self.paint_mask_btn.setStyleSheet("color: #FFD700; font-weight: bold;")
        self.paint_mask_btn.setToolTip("Paint on segmentation mask")
        self.mode_button_group.addButton(self.paint_mask_btn, 4)
        mask_layout.addWidget(self.paint_mask_btn)
        
        self.erase_mask_btn = QRadioButton("Erase")
        self.erase_mask_btn.setStyleSheet("color: #888; font-weight: bold;")
        self.erase_mask_btn.setToolTip("Erase from segmentation mask")
        self.mode_button_group.addButton(self.erase_mask_btn, 5)
        mask_layout.addWidget(self.erase_mask_btn)
        
        # Brush size
        mask_layout.addWidget(QLabel("Size:"))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 100)
        self.brush_size_spin.setValue(20)
        self.brush_size_spin.setToolTip("Brush size for mask editing")
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_changed)
        mask_layout.addWidget(self.brush_size_spin)
        
        layout.addWidget(mask_group)
        
        # Accept/Clear mask buttons
        self.accept_mask_btn = QPushButton("✓ Accept Mask")
        self.accept_mask_btn.setToolTip("Accept current mask edits")
        self.accept_mask_btn.setStyleSheet("background-color: #228B22; color: white;")
        layout.addWidget(self.accept_mask_btn)
        
        self.clear_edit_mask_btn = QPushButton("✗ Clear Edit")
        self.clear_edit_mask_btn.setToolTip("Clear mask being edited")
        layout.addWidget(self.clear_edit_mask_btn)
        
        layout.addStretch()
        
        # Save button
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setToolTip("Save annotations")
        layout.addWidget(self.save_btn)
        
        return toolbar
    
    def _create_navigation_bar(self) -> QWidget:
        """Create the video navigation bar with folder navigation."""
        nav_bar = QFrame()
        nav_bar.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout(nav_bar)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Zoom controls (left side)
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout(zoom_group)
        zoom_layout.setContentsMargins(5, 2, 5, 2)
        
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedSize(28, 28)
        self.zoom_out_btn.setToolTip("Zoom out (Ctrl+-)")
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_layout.addWidget(self.zoom_out_btn)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 500)  # 10% to 500%
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(100)
        self.zoom_slider.setToolTip("Zoom level")
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        zoom_layout.addWidget(self.zoom_slider)
        
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(28, 28)
        self.zoom_in_btn.setToolTip("Zoom in (Ctrl++)")
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(45)
        self.zoom_label.setStyleSheet("font-size: 11px;")
        zoom_layout.addWidget(self.zoom_label)
        
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setFixedWidth(35)
        self.fit_btn.setToolTip("Fit to window (F)")
        self.fit_btn.clicked.connect(self._fit_to_window)
        zoom_layout.addWidget(self.fit_btn)
        
        self.reset_zoom_btn = QPushButton("1:1")
        self.reset_zoom_btn.setFixedWidth(35)
        self.reset_zoom_btn.setToolTip("Reset to 100% (1)")
        self.reset_zoom_btn.clicked.connect(self._reset_zoom)
        zoom_layout.addWidget(self.reset_zoom_btn)
        
        layout.addWidget(zoom_group)
        
        layout.addSpacing(10)
        
        # Folder navigation
        self.prev_folder_btn = QPushButton("⏮ Prev Folder")
        self.prev_folder_btn.setToolTip("Go to previous folder (Ctrl+Left)")
        self.prev_folder_btn.setShortcut("Ctrl+Left")
        layout.addWidget(self.prev_folder_btn)
        
        self.folder_label = QLabel("Folder 0/0")
        self.folder_label.setStyleSheet("font-size: 12px; color: #666;")
        self.folder_label.setMinimumWidth(100)
        layout.addWidget(self.folder_label)
        
        self.next_folder_btn = QPushButton("Next Folder ⏭")
        self.next_folder_btn.setToolTip("Go to next folder (Ctrl+Right)")
        self.next_folder_btn.setShortcut("Ctrl+Right")
        layout.addWidget(self.next_folder_btn)
        
        layout.addSpacing(20)
        
        # Video navigation (center)
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.setToolTip("Go to previous video (Left Arrow)")
        self.prev_btn.setShortcut(Qt.Key_Left)
        layout.addWidget(self.prev_btn)
        
        layout.addStretch()
        
        # Video position indicator
        self.position_label = QLabel("0 / 0")
        self.position_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.position_label)
        
        layout.addStretch()
        
        # Next button
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setToolTip("Go to next video (Right Arrow)")
        self.next_btn.setShortcut(Qt.Key_Right)
        layout.addWidget(self.next_btn)
        
        return nav_bar
    
    def _connect_signals(self):
        """Connect internal signals."""
        # Mode selection
        self.mode_button_group.buttonClicked.connect(self._on_mode_changed)
        
        # Video navigation
        self.prev_btn.clicked.connect(self.prev_video_requested.emit)
        self.next_btn.clicked.connect(self.next_video_requested.emit)
        
        # Folder navigation
        self.prev_folder_btn.clicked.connect(self.prev_folder_requested.emit)
        self.next_folder_btn.clicked.connect(self.next_folder_requested.emit)
        
        # Actions
        self.detect_btn.clicked.connect(self.run_detection_requested.emit)
        self.segment_btn.clicked.connect(self._on_segment_clicked)
        self.clear_masks_btn.clicked.connect(self.canvas.clear_masks)
        self.clear_all_btn.clicked.connect(self.canvas.clear_annotations)
        self.save_btn.clicked.connect(self.save_requested.emit)
        
        # Mask editing
        self.accept_mask_btn.clicked.connect(self._on_accept_mask)
        self.clear_edit_mask_btn.clicked.connect(self.canvas.clear_editable_mask)
        
        # Canvas signals
        self.canvas.box_drawn.connect(self.box_drawn.emit)
        self.canvas.request_segmentation.connect(self.segment_requested.emit)
        self.canvas.box_selected.connect(self._on_box_selected)
        self.canvas.zoom_changed.connect(self._update_zoom_display)
    
    def _on_mode_changed(self, button):
        """Handle mode change."""
        mode_map = {
            self.select_mode_btn: AnnotationMode.SELECT,
            self.head_mode_btn: AnnotationMode.HEAD,
            self.tail_mode_btn: AnnotationMode.TAIL,
            self.detection_mode_btn: AnnotationMode.DETECTION,
            self.paint_mask_btn: AnnotationMode.MASK_PAINT,
            self.erase_mask_btn: AnnotationMode.MASK_ERASE
        }
        mode = mode_map.get(button, AnnotationMode.SELECT)
        self.canvas.set_mode(mode)
    
    def _on_box_selected(self, box_type: str, worm_id: int):
        """Handle box selection."""
        self.segment_btn.setEnabled(True)
    
    def _on_segment_clicked(self):
        """Handle segment button click."""
        selected_box = self.canvas.get_selected_box()
        if selected_box:
            coords = selected_box.get_coordinates()
            self.segment_requested.emit(*coords)
    
    def _on_brush_size_changed(self, size: int):
        """Handle brush size change."""
        self.canvas.set_brush_size(size)
    
    def _on_accept_mask(self):
        """Handle accept mask button click."""
        mask = self.canvas.get_current_mask()
        if mask is not None:
            # Get worm_id and mask_type from selected box
            selected = self.canvas.get_selected_box()
            worm_id = selected.worm_id if selected else None
            
            if selected:
                from ui.annotation_canvas import BoxType
                if selected.box_type == BoxType.HEAD:
                    mask_type = "head"
                elif selected.box_type == BoxType.TAIL:
                    mask_type = "tail"
                else:
                    mask_type = "worm"
            else:
                mask_type = "worm"
            
            # Remove existing mask for this worm/type before adding edited one
            if worm_id:
                self.canvas.remove_masks_by_worm_id(worm_id, mask_type)
            
            # Add as regular mask overlay with worm tracking
            self.canvas.add_segmentation_mask(mask, worm_id=worm_id, mask_type=mask_type)
            
            self.mask_accepted.emit(mask.copy())
            self.canvas.clear_editable_mask()
            # Switch back to select mode
            self.set_select_mode()
    
    def set_editable_mask(self, mask):
        """Set a mask for editing."""
        self.canvas.set_editable_mask(mask)
    
    def set_frame(self, frame):
        """Set the current frame to display."""
        self.canvas.set_frame(frame)
    
    def set_position(self, current: int, total: int):
        """Update position indicator."""
        self.position_label.setText(f"{current} / {total}")
        
        # Enable/disable navigation buttons
        self.prev_btn.setEnabled(current > 1)
        self.next_btn.setEnabled(current < total)
    
    def set_folder_position(self, current: int, total: int, folder_name: str = ""):
        """Update folder position indicator."""
        self.folder_label.setText(f"Folder {current}/{total}")
        if folder_name:
            self.folder_label.setToolTip(folder_name)
        
        # Enable/disable folder navigation buttons
        self.prev_folder_btn.setEnabled(current > 1)
        self.next_folder_btn.setEnabled(current < total)
    
    def set_video_info(self, filename: str, info: str = ""):
        """Set video info display."""
        # Could add a label to show this
        pass
    
    def clear_annotations(self):
        """Clear all annotations from canvas."""
        self.canvas.clear_annotations()
        self.canvas.clear_editable_mask()
        self.segment_btn.setEnabled(False)
    
    def add_detection_box(self, x1, y1, x2, y2, worm_id):
        """Add a YOLO detection box."""
        return self.canvas.add_detection_box(x1, y1, x2, y2, worm_id, BoxType.YOLO_DETECTION)
    
    def add_head_box(self, x1, y1, x2, y2, worm_id):
        """Add a head annotation box."""
        return self.canvas.add_head_box(x1, y1, x2, y2, worm_id)
    
    def add_tail_box(self, x1, y1, x2, y2, worm_id):
        """Add a tail annotation box."""
        return self.canvas.add_tail_box(x1, y1, x2, y2, worm_id)
    
    def add_segmentation_mask(self, mask, worm_id=None, mask_type="worm"):
        """Add a segmentation mask."""
        self.canvas.add_segmentation_mask(mask, worm_id=worm_id, mask_type=mask_type)
    
    def set_select_mode(self):
        """Switch to select mode."""
        self.select_mode_btn.setChecked(True)
        self.canvas.set_mode(AnnotationMode.SELECT)
    
    def set_head_mode(self):
        """Switch to head annotation mode."""
        self.head_mode_btn.setChecked(True)
        self.canvas.set_mode(AnnotationMode.HEAD)
    
    def set_tail_mode(self):
        """Switch to tail annotation mode."""
        self.tail_mode_btn.setChecked(True)
        self.canvas.set_mode(AnnotationMode.TAIL)
    
    # Zoom methods
    def _zoom_in(self):
        """Zoom in by 20%."""
        self.canvas.scale(1.2, 1.2)
        self._update_zoom_display()
    
    def _zoom_out(self):
        """Zoom out by 20%."""
        self.canvas.scale(1/1.2, 1/1.2)
        self._update_zoom_display()
    
    def _on_zoom_slider_changed(self, value):
        """Handle zoom slider change."""
        # Get current zoom level
        current_scale = self.canvas.transform().m11()
        target_scale = value / 100.0
        
        # Calculate factor needed
        if current_scale > 0:
            factor = target_scale / current_scale
            self.canvas.scale(factor, factor)
        
        self.zoom_label.setText(f"{value}%")
    
    def _fit_to_window(self):
        """Fit image to window."""
        # Fit to the image bounds, not the expanded scene rect
        if self.canvas._image_item is not None:
            self.canvas.fitInView(self.canvas._image_item, Qt.KeepAspectRatio)
        else:
            self.canvas.fitInView(self.canvas.sceneRect(), Qt.KeepAspectRatio)
        self._update_zoom_display()
    
    def _reset_zoom(self):
        """Reset zoom to 100%."""
        self.canvas.resetTransform()
        self._update_zoom_display()
    
    def _update_zoom_display(self):
        """Update zoom slider and label to match current zoom."""
        current_scale = self.canvas.transform().m11()
        zoom_percent = int(current_scale * 100)
        
        # Clamp to slider range
        zoom_percent = max(10, min(500, zoom_percent))
        
        # Block signals to prevent feedback loop
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(zoom_percent)
        self.zoom_slider.blockSignals(False)
        
        self.zoom_label.setText(f"{zoom_percent}%")
    
    def zoom_to(self, percent: int):
        """Zoom to a specific percentage."""
        self.canvas.resetTransform()
        scale = percent / 100.0
        self.canvas.scale(scale, scale)
        self._update_zoom_display()


class VideoListWidget(QWidget):
    """
    Widget showing list of videos in the folder.
    """
    
    video_selected = pyqtSignal(int)  # index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        self.label = QLabel("Videos")
        self.label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label)
        
        # Video list would go here - simplified for now
        self.info_label = QLabel("No folder loaded")
        layout.addWidget(self.info_label)
        
        layout.addStretch()
    
    def set_videos(self, videos: list):
        """Set the list of videos."""
        if videos:
            self.info_label.setText(f"{len(videos)} videos")
        else:
            self.info_label.setText("No videos found")
    
    def set_current(self, index: int):
        """Highlight current video."""
        pass
