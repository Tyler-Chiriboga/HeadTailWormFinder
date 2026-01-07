"""
Annotation canvas for drawing and editing bounding boxes.
Uses QGraphicsView/QGraphicsScene for interactive annotation.
"""
from enum import Enum
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsPixmapItem, QGraphicsPolygonItem, QMenu, QAction
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QPen, QBrush, QColor, QPixmap, QImage, QPainter,
    QPolygonF, QCursor, QBitmap
)
import numpy as np


class AnnotationMode(Enum):
    """Current annotation mode."""
    SELECT = "select"
    HEAD = "head"
    TAIL = "tail"
    DETECTION = "detection"
    MASK_PAINT = "mask_paint"    # Add to mask
    MASK_ERASE = "mask_erase"    # Remove from mask


class BoxType(Enum):
    """Type of bounding box."""
    YOLO_DETECTION = "yolo"
    HEAD = "head"
    TAIL = "tail"
    USER_SELECTION = "selection"


@dataclass
class BoxColors:
    """Colors for different box types."""
    YOLO_DETECTION = QColor(0, 120, 255, 180)      # Blue
    HEAD = QColor(0, 255, 0, 180)                   # Green
    TAIL = QColor(255, 0, 0, 180)                   # Red
    SELECTION = QColor(255, 165, 0, 180)           # Orange
    SELECTED_BORDER = QColor(255, 255, 0, 255)     # Yellow
    MASK = QColor(255, 255, 0, 100)                # Yellow transparent


class AnnotationBox(QGraphicsRectItem):
    """
    A bounding box annotation item.
    Supports selection, moving, and resizing.
    """
    
    # Resize handle size
    HANDLE_SIZE = 10
    
    def __init__(
        self,
        rect: QRectF,
        box_type: BoxType,
        worm_id: Optional[int] = None,
        parent=None
    ):
        super().__init__(rect, parent)
        
        self.box_type = box_type
        self.worm_id = worm_id
        self._is_selected = False
        self._resize_handle = None  # Which handle is being dragged
        self._drag_start = None
        
        # Set appearance based on type
        self._setup_appearance()
        
        # Make selectable and movable
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setAcceptHoverEvents(True)
    
    def _setup_appearance(self):
        """Set pen and brush based on box type."""
        color_map = {
            BoxType.YOLO_DETECTION: BoxColors.YOLO_DETECTION,
            BoxType.HEAD: BoxColors.HEAD,
            BoxType.TAIL: BoxColors.TAIL,
            BoxType.USER_SELECTION: BoxColors.SELECTION
        }
        
        color = color_map.get(self.box_type, BoxColors.SELECTION)
        
        pen = QPen(color)
        pen.setWidth(2)
        self.setPen(pen)
        
        # Semi-transparent fill
        fill_color = QColor(color)
        fill_color.setAlpha(30)
        self.setBrush(QBrush(fill_color))
    
    def set_selected_style(self, selected: bool):
        """Update appearance when selected."""
        self._is_selected = selected
        
        if selected:
            pen = QPen(BoxColors.SELECTED_BORDER)
            pen.setWidth(3)
            pen.setStyle(Qt.DashLine)
            self.setPen(pen)
        else:
            self._setup_appearance()
    
    def get_coordinates(self) -> Tuple[float, float, float, float]:
        """Get box coordinates as (x1, y1, x2, y2) in scene coordinates."""
        # Get the bounding rect in scene coordinates (accounts for item position)
        scene_rect = self.sceneBoundingRect()
        return (
            scene_rect.x(),
            scene_rect.y(),
            scene_rect.x() + scene_rect.width(),
            scene_rect.y() + scene_rect.height()
        )
    
    def hoverEnterEvent(self, event):
        """Handle hover enter."""
        if not self._is_selected:
            pen = self.pen()
            pen.setWidth(3)
            self.setPen(pen)
        super().hoverEnterEvent(event)
    
    def hoverMoveEvent(self, event):
        """Update cursor based on position for resize handles."""
        handle = self._get_resize_handle(event.pos())
        if handle:
            cursor_map = {
                'top-left': Qt.SizeFDiagCursor,
                'top-right': Qt.SizeBDiagCursor,
                'bottom-left': Qt.SizeBDiagCursor,
                'bottom-right': Qt.SizeFDiagCursor,
                'top': Qt.SizeVerCursor,
                'bottom': Qt.SizeVerCursor,
                'left': Qt.SizeHorCursor,
                'right': Qt.SizeHorCursor
            }
            self.setCursor(cursor_map.get(handle, Qt.ArrowCursor))
        else:
            self.setCursor(Qt.SizeAllCursor)  # Move cursor
        super().hoverMoveEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle hover leave."""
        if not self._is_selected:
            self._setup_appearance()
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)
    
    def _get_resize_handle(self, pos: QPointF) -> Optional[str]:
        """Determine which resize handle (if any) is at the given position."""
        rect = self.rect()
        hs = self.HANDLE_SIZE
        
        # Check corners first
        if QRectF(rect.left(), rect.top(), hs, hs).contains(pos):
            return 'top-left'
        if QRectF(rect.right() - hs, rect.top(), hs, hs).contains(pos):
            return 'top-right'
        if QRectF(rect.left(), rect.bottom() - hs, hs, hs).contains(pos):
            return 'bottom-left'
        if QRectF(rect.right() - hs, rect.bottom() - hs, hs, hs).contains(pos):
            return 'bottom-right'
        
        # Check edges
        if QRectF(rect.left() + hs, rect.top(), rect.width() - 2*hs, hs).contains(pos):
            return 'top'
        if QRectF(rect.left() + hs, rect.bottom() - hs, rect.width() - 2*hs, hs).contains(pos):
            return 'bottom'
        if QRectF(rect.left(), rect.top() + hs, hs, rect.height() - 2*hs).contains(pos):
            return 'left'
        if QRectF(rect.right() - hs, rect.top() + hs, hs, rect.height() - 2*hs).contains(pos):
            return 'right'
        
        return None
    
    def mousePressEvent(self, event):
        """Handle mouse press for resize/move."""
        if event.button() == Qt.LeftButton:
            self._resize_handle = self._get_resize_handle(event.pos())
            self._drag_start = event.pos()
            if self._resize_handle:
                # Don't call parent to prevent move during resize
                event.accept()
                return
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for resize."""
        if self._resize_handle and self._drag_start:
            rect = self.rect()
            delta = event.pos() - self._drag_start
            
            new_rect = QRectF(rect)
            
            # Apply resize based on handle
            if 'left' in self._resize_handle:
                new_rect.setLeft(rect.left() + delta.x())
            if 'right' in self._resize_handle:
                new_rect.setRight(rect.right() + delta.x())
            if 'top' in self._resize_handle:
                new_rect.setTop(rect.top() + delta.y())
            if 'bottom' in self._resize_handle:
                new_rect.setBottom(rect.bottom() + delta.y())
            
            # Ensure minimum size
            if new_rect.width() >= 10 and new_rect.height() >= 10:
                self.setRect(new_rect.normalized())
                self._drag_start = event.pos()
            
            event.accept()
            return
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self._resize_handle = None
        self._drag_start = None
        super().mouseReleaseEvent(event)


class SegmentationMaskItem(QGraphicsPolygonItem):
    """
    A segmentation mask overlay item.
    """
    
    def __init__(self, polygon: QPolygonF, worm_id: int = None, mask_type: str = "worm", parent=None):
        super().__init__(polygon, parent)
        
        self.worm_id = worm_id
        self.mask_type = mask_type  # "worm", "head", or "tail"
        
        # Set appearance based on mask type
        if mask_type == "head":
            color = BoxColors.HEAD
        elif mask_type == "tail":
            color = BoxColors.TAIL
        else:
            color = BoxColors.MASK
        
        pen = QPen(color)
        pen.setWidth(2)
        self.setPen(pen)
        
        fill = QColor(color)
        fill.setAlpha(60)
        self.setBrush(QBrush(fill))


class AnnotationCanvas(QGraphicsView):
    """
    Canvas for displaying frames and drawing annotations.
    
    Signals:
        box_drawn: Emitted when a new box is drawn (box_type, x1, y1, x2, y2)
        box_selected: Emitted when a box is selected (box_type, worm_id)
        request_segmentation: Emitted to request SAM segmentation (x1, y1, x2, y2)
        mask_edited: Emitted when mask is edited (mask_array)
        zoom_changed: Emitted when zoom level changes
    """
    
    box_drawn = pyqtSignal(str, float, float, float, float)  # type, x1, y1, x2, y2
    box_selected = pyqtSignal(str, int)  # type, worm_id
    request_segmentation = pyqtSignal(float, float, float, float)  # x1, y1, x2, y2
    mask_edited = pyqtSignal(object)  # numpy array
    zoom_changed = pyqtSignal()  # Emitted when zoom changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Image item
        self._image_item: Optional[QGraphicsPixmapItem] = None
        self._current_frame: Optional[np.ndarray] = None
        
        # Drawing state
        self._mode = AnnotationMode.SELECT
        self._drawing = False
        self._start_point: Optional[QPointF] = None
        self._current_rect: Optional[QGraphicsRectItem] = None
        
        # Annotations
        self._boxes: List[AnnotationBox] = []
        self._masks: List[SegmentationMaskItem] = []
        self._selected_box: Optional[AnnotationBox] = None
        
        # Mask editing
        self._current_mask: Optional[np.ndarray] = None  # Binary mask for editing
        self._mask_pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._brush_size = 20  # Brush size for mask editing
        self._painting = False
        self._last_paint_pos: Optional[QPointF] = None
        
        # Pan state
        self._panning = False
        self._pan_start = None
        
        # Configure view
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        # Allow view to show content beyond scene rect (important for zoom)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Enable mouse tracking for cursor updates
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
    
    def set_frame(self, frame: np.ndarray):
        """
        Set the current frame to display.
        
        Args:
            frame: RGB image as numpy array (H, W, C)
        """
        self._current_frame = frame
        
        # Convert to QPixmap - ensure contiguous array
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        # Make sure the data is contiguous
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        qimage = QImage(
            frame.data, w, h, bytes_per_line,
            QImage.Format_RGB888
        ).copy()  # Copy to ensure data persists
        pixmap = QPixmap.fromImage(qimage)
        
        # Check if this is a new image size
        is_new_size = (self._image_item is None or 
                       self._image_item.pixmap().width() != w or 
                       self._image_item.pixmap().height() != h)
        
        # Update or create image item
        if self._image_item is None:
            self._image_item = self.scene.addPixmap(pixmap)
            self._image_item.setZValue(-1)  # Behind annotations
        else:
            self._image_item.setPixmap(pixmap)
        
        # Set scene rect with some padding to allow scrolling when zoomed in
        padding = max(w, h) * 0.5
        self.scene.setSceneRect(-padding, -padding, w + padding * 2, h + padding * 2)
        
        # Only fit in view if it's a new size or first load
        if is_new_size:
            self.fitInView(0, 0, w, h, Qt.KeepAspectRatio)
            self.zoom_changed.emit()
    
    def set_mode(self, mode: AnnotationMode):
        """Set the current annotation mode."""
        self._mode = mode
        
        # Update cursor
        if mode == AnnotationMode.SELECT:
            self.setCursor(Qt.ArrowCursor)
        elif mode in (AnnotationMode.MASK_PAINT, AnnotationMode.MASK_ERASE):
            # Use circle cursor for mask editing
            self._update_brush_cursor()
        else:
            self.setCursor(Qt.CrossCursor)
    
    def _update_brush_cursor(self):
        """Create a circular cursor matching the brush size."""
        # Account for view transform scale
        transform = self.transform()
        scale = transform.m11()  # Get current zoom scale
        cursor_size = max(4, int(self._brush_size * 2 * scale))
        
        # Create cursor pixmap
        pixmap = QPixmap(cursor_size + 2, cursor_size + 2)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw circle outline
        if self._mode == AnnotationMode.MASK_PAINT:
            pen = QPen(QColor(255, 255, 0, 200))  # Yellow for paint
        else:
            pen = QPen(QColor(255, 0, 0, 200))  # Red for erase
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        center = cursor_size // 2 + 1
        radius = cursor_size // 2
        painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
        
        # Draw crosshair in center
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
        painter.drawLine(center - 3, center, center + 3, center)
        painter.drawLine(center, center - 3, center, center + 3)
        
        painter.end()
        
        cursor = QCursor(pixmap, center, center)
        self.setCursor(cursor)
    
    def set_brush_size(self, size: int):
        """Set brush size for mask editing."""
        self._brush_size = max(1, min(100, size))
        # Update cursor if in mask editing mode
        if self._mode in (AnnotationMode.MASK_PAINT, AnnotationMode.MASK_ERASE):
            self._update_brush_cursor()
    
    def get_brush_size(self) -> int:
        """Get current brush size."""
        return self._brush_size
    
    def get_mode(self) -> AnnotationMode:
        """Get current annotation mode."""
        return self._mode
    
    def clear_annotations(self):
        """Clear all annotations from the canvas."""
        for box in self._boxes:
            self.scene.removeItem(box)
        self._boxes.clear()
        
        for mask in self._masks:
            self.scene.removeItem(mask)
        self._masks.clear()
        
        self._selected_box = None
        
        # Also clear editable mask
        self.clear_editable_mask()
    
    def add_detection_box(
        self,
        x1: float, y1: float, x2: float, y2: float,
        worm_id: int,
        box_type: BoxType = BoxType.YOLO_DETECTION
    ) -> AnnotationBox:
        """
        Add a detection bounding box.
        
        Returns:
            The created AnnotationBox
        """
        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        box = AnnotationBox(rect, box_type, worm_id)
        
        # Set z-value so head/tail boxes appear above detection boxes
        if box_type == BoxType.YOLO_DETECTION:
            box.setZValue(10)
        elif box_type == BoxType.HEAD:
            box.setZValue(20)
        elif box_type == BoxType.TAIL:
            box.setZValue(20)
        
        self.scene.addItem(box)
        self._boxes.append(box)
        
        return box
    
    def add_head_box(
        self,
        x1: float, y1: float, x2: float, y2: float,
        worm_id: int
    ) -> AnnotationBox:
        """Add a head bounding box."""
        return self.add_detection_box(x1, y1, x2, y2, worm_id, BoxType.HEAD)
    
    def add_tail_box(
        self,
        x1: float, y1: float, x2: float, y2: float,
        worm_id: int
    ) -> AnnotationBox:
        """Add a tail bounding box."""
        return self.add_detection_box(x1, y1, x2, y2, worm_id, BoxType.TAIL)
    
    def add_segmentation_mask(self, mask: np.ndarray, worm_id: int = None, mask_type: str = "worm"):
        """
        Add a segmentation mask overlay.
        
        Args:
            mask: Binary mask array (H, W)
            worm_id: The worm ID this mask belongs to
            mask_type: Type of mask - "worm", "head", or "tail"
        """
        import cv2
        
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Convert to QPolygonF
            points = [QPointF(p[0][0], p[0][1]) for p in contour]
            polygon = QPolygonF(points)
            
            mask_item = SegmentationMaskItem(polygon, worm_id=worm_id, mask_type=mask_type)
            self.scene.addItem(mask_item)
            self._masks.append(mask_item)
    
    def remove_masks_by_worm_id(self, worm_id: int, mask_type: str = None):
        """Remove masks for a specific worm."""
        to_remove = []
        for mask in self._masks:
            if mask.worm_id == worm_id:
                if mask_type is None or mask.mask_type == mask_type:
                    to_remove.append(mask)
        
        for mask in to_remove:
            self.scene.removeItem(mask)
            self._masks.remove(mask)
    
    def clear_masks(self):
        """Clear segmentation masks."""
        for mask in self._masks:
            self.scene.removeItem(mask)
        self._masks.clear()
        
        # Also clear editable mask
        self.clear_editable_mask()
    
    def remove_box(self, box: AnnotationBox):
        """Remove a specific box."""
        if box in self._boxes:
            self.scene.removeItem(box)
            self._boxes.remove(box)
            if self._selected_box == box:
                self._selected_box = None
    
    def remove_boxes_by_worm_id(self, worm_id: int, box_type: Optional[BoxType] = None):
        """Remove boxes for a specific worm."""
        to_remove = []
        for box in self._boxes:
            if box.worm_id == worm_id:
                if box_type is None or box.box_type == box_type:
                    to_remove.append(box)
        
        for box in to_remove:
            self.remove_box(box)
    
    def get_selected_box(self) -> Optional[AnnotationBox]:
        """Get currently selected box."""
        return self._selected_box
    
    def set_editable_mask(self, mask: np.ndarray):
        """
        Set a mask for editing.
        
        Args:
            mask: Binary mask array (H, W) with values 0 or 1/255
        """
        # Normalize to 0-1
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        self._current_mask = mask.copy()
        self._update_mask_display()
    
    def get_current_mask(self) -> Optional[np.ndarray]:
        """Get the current editable mask."""
        return self._current_mask
    
    def clear_editable_mask(self):
        """Clear the editable mask."""
        self._current_mask = None
        if self._mask_pixmap_item is not None:
            self.scene.removeItem(self._mask_pixmap_item)
            self._mask_pixmap_item = None
    
    def _update_mask_display(self):
        """Update the mask overlay display."""
        if self._current_mask is None:
            return
        
        h, w = self._current_mask.shape
        
        # Create RGBA image for mask overlay
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[self._current_mask > 0] = [0, 255, 255, 120]  # Cyan with transparency
        
        # Make contiguous for QImage
        mask_rgba = np.ascontiguousarray(mask_rgba)
        
        # Convert to QPixmap
        qimage = QImage(
            mask_rgba.data, w, h, 4 * w,
            QImage.Format_RGBA8888
        ).copy()  # Copy to own the data
        pixmap = QPixmap.fromImage(qimage)
        
        # Update or create mask item
        if self._mask_pixmap_item is None:
            self._mask_pixmap_item = self.scene.addPixmap(pixmap)
            self._mask_pixmap_item.setZValue(0.5)  # Above image, below boxes
        else:
            self._mask_pixmap_item.setPixmap(pixmap)
    
    def _paint_on_mask(self, pos: QPointF, erase: bool = False):
        """Paint or erase on the mask at the given position."""
        if self._current_mask is None:
            # Initialize empty mask if none exists
            if self._current_frame is not None:
                h, w = self._current_frame.shape[:2]
                self._current_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                return
        
        import cv2
        
        x, y = int(pos.x()), int(pos.y())
        h, w = self._current_mask.shape
        
        # Check bounds
        if 0 <= x < w and 0 <= y < h:
            # Draw circle on mask
            value = 0 if erase else 1
            cv2.circle(self._current_mask, (x, y), self._brush_size, value, -1)
            self._update_mask_display()
    
    def _paint_line_on_mask(self, start: QPointF, end: QPointF, erase: bool = False):
        """Paint a line on the mask between two points."""
        if self._current_mask is None:
            if self._current_frame is not None:
                h, w = self._current_frame.shape[:2]
                self._current_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                return
        
        import cv2
        
        x1, y1 = int(start.x()), int(start.y())
        x2, y2 = int(end.x()), int(end.y())
        
        value = 0 if erase else 1
        cv2.line(self._current_mask, (x1, y1), (x2, y2), value, self._brush_size * 2)
        self._update_mask_display()
    
    def mousePressEvent(self, event):
        """Handle mouse press for drawing."""
        # Middle mouse button for panning
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            
            # Handle mask painting modes
            if self._mode == AnnotationMode.MASK_PAINT:
                self._painting = True
                self._last_paint_pos = scene_pos
                self._paint_on_mask(scene_pos, erase=False)
                return
            elif self._mode == AnnotationMode.MASK_ERASE:
                self._painting = True
                self._last_paint_pos = scene_pos
                self._paint_on_mask(scene_pos, erase=True)
                return
            elif self._mode == AnnotationMode.SELECT:
                # Check if clicked on a box
                item = self.itemAt(event.pos())
                if isinstance(item, AnnotationBox):
                    self._select_box(item)
                else:
                    self._deselect_all()
                # Only call super in select mode to allow box dragging
                super().mousePressEvent(event)
            else:
                # Start drawing box (HEAD, TAIL, or DETECTION mode)
                # Don't call super() to prevent existing boxes from being moved
                self._drawing = True
                self._start_point = scene_pos
                
                # Create temporary rectangle
                self._current_rect = QGraphicsRectItem()
                pen = QPen(QColor(255, 255, 255, 200))
                pen.setWidth(2)
                pen.setStyle(Qt.DashLine)
                self._current_rect.setPen(pen)
                self.scene.addItem(self._current_rect)
                event.accept()
                return
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing."""
        # Handle panning
        if self._panning and self._pan_start:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        
        scene_pos = self.mapToScene(event.pos())
        
        # Handle mask painting
        if self._painting and self._last_paint_pos:
            erase = (self._mode == AnnotationMode.MASK_ERASE)
            self._paint_line_on_mask(self._last_paint_pos, scene_pos, erase)
            self._last_paint_pos = scene_pos
            return
        
        if self._drawing and self._start_point and self._current_rect:
            current_point = self.mapToScene(event.pos())
            
            # Update rectangle
            rect = QRectF(self._start_point, current_point).normalized()
            self._current_rect.setRect(rect)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to finish drawing."""
        # Handle pan release
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start = None
            # Restore cursor based on mode
            if self._mode in (AnnotationMode.MASK_PAINT, AnnotationMode.MASK_ERASE):
                self._update_brush_cursor()
            elif self._mode == AnnotationMode.SELECT:
                self.setCursor(Qt.ArrowCursor)
            else:
                self.setCursor(Qt.CrossCursor)
            event.accept()
            return
        
        if event.button() == Qt.LeftButton:
            # Handle mask painting release
            if self._painting:
                self._painting = False
                self._last_paint_pos = None
                # Emit mask edited signal
                if self._current_mask is not None:
                    self.mask_edited.emit(self._current_mask.copy())
                return
            
            if self._drawing:
                self._drawing = False
                
                if self._start_point and self._current_rect:
                    end_point = self.mapToScene(event.pos())
                    rect = QRectF(self._start_point, end_point).normalized()
                    
                    # Remove temporary rectangle
                    self.scene.removeItem(self._current_rect)
                    self._current_rect = None
                    
                    # Check minimum size
                    if rect.width() > 5 and rect.height() > 5:
                        # Emit signal based on mode
                        box_type = {
                            AnnotationMode.HEAD: "head",
                            AnnotationMode.TAIL: "tail",
                            AnnotationMode.DETECTION: "detection"
                        }.get(self._mode, "selection")
                        
                        self.box_drawn.emit(
                            box_type,
                            rect.x(), rect.y(),
                            rect.x() + rect.width(),
                            rect.y() + rect.height()
                        )
            
            self._start_point = None
        
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        # Get current scale
        current_scale = self.transform().m11()
        
        factor = 1.15
        if event.angleDelta().y() > 0:
            # Zoom in - limit to 10x (1000%)
            if current_scale < 10.0:
                self.scale(factor, factor)
        else:
            # Zoom out - limit to 0.1x (10%)
            if current_scale > 0.1:
                self.scale(1 / factor, 1 / factor)
        
        # Update brush cursor size after zoom
        if self._mode in (AnnotationMode.MASK_PAINT, AnnotationMode.MASK_ERASE):
            self._update_brush_cursor()
        
        # Emit zoom changed signal
        self.zoom_changed.emit()
        
        event.accept()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        # Zoom shortcuts
        if event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                self.scale(1.2, 1.2)
                self.zoom_changed.emit()
                return
            elif event.key() == Qt.Key_Minus:
                self.scale(1/1.2, 1/1.2)
                self.zoom_changed.emit()
                return
            elif event.key() == Qt.Key_0:
                # Reset zoom to 100%
                self.resetTransform()
                self.zoom_changed.emit()
                return
        
        # Fit to window
        if event.key() == Qt.Key_F:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self.zoom_changed.emit()
            return
        
        # Reset to 100%
        if event.key() == Qt.Key_1:
            self.resetTransform()
            self.zoom_changed.emit()
            return
        
        super().keyPressEvent(event)

    def _select_box(self, box: AnnotationBox):
        """Select a box."""
        self._deselect_all()
        self._selected_box = box
        box.set_selected_style(True)
        
        if box.worm_id is not None:
            self.box_selected.emit(box.box_type.value, box.worm_id)
    
    def _deselect_all(self):
        """Deselect all boxes."""
        for box in self._boxes:
            box.set_selected_style(False)
        self._selected_box = None
    
    def _show_context_menu(self, pos):
        """Show context menu."""
        item = self.itemAt(pos)
        
        menu = QMenu(self)
        
        if isinstance(item, AnnotationBox):
            # Box-specific actions
            delete_action = QAction("Delete Box", self)
            delete_action.triggered.connect(lambda: self.remove_box(item))
            menu.addAction(delete_action)
            
            if item.box_type == BoxType.YOLO_DETECTION:
                segment_action = QAction("Segment with SAM", self)
                coords = item.get_coordinates()
                segment_action.triggered.connect(
                    lambda: self.request_segmentation.emit(*coords)
                )
                menu.addAction(segment_action)
        else:
            # General actions
            clear_action = QAction("Clear All Annotations", self)
            clear_action.triggered.connect(self.clear_annotations)
            menu.addAction(clear_action)
            
            clear_masks_action = QAction("Clear Masks", self)
            clear_masks_action.triggered.connect(self.clear_masks)
            menu.addAction(clear_masks_action)
        
        menu.addSeparator()
        
        # Zoom actions
        fit_action = QAction("Fit to View", self)
        fit_action.triggered.connect(
            lambda: self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        )
        menu.addAction(fit_action)
        
        reset_zoom_action = QAction("Reset Zoom (100%)", self)
        reset_zoom_action.triggered.connect(lambda: self.resetTransform())
        menu.addAction(reset_zoom_action)
        
        menu.exec_(self.mapToGlobal(pos))
    
    def resizeEvent(self, event):
        """Handle resize to maintain fit."""
        super().resizeEvent(event)
        if self._image_item:
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def enterEvent(self, event):
        """Update cursor when mouse enters canvas."""
        if self._mode in (AnnotationMode.MASK_PAINT, AnnotationMode.MASK_ERASE):
            self._update_brush_cursor()
        super().enterEvent(event)
    
    def get_boxes_by_type(self, box_type: BoxType) -> List[AnnotationBox]:
        """Get all boxes of a specific type."""
        return [b for b in self._boxes if b.box_type == box_type]
    
    def get_boxes_by_worm(self, worm_id: int) -> List[AnnotationBox]:
        """Get all boxes for a specific worm."""
        return [b for b in self._boxes if b.worm_id == worm_id]
    
    def get_all_box_data(self) -> List[dict]:
        """Get all box data for syncing back to annotations.
        
        Returns:
            List of dicts with worm_id, box_type, and coordinates
        """
        result = []
        for box in self._boxes:
            coords = box.get_coordinates()
            result.append({
                'worm_id': box.worm_id,
                'box_type': box.box_type,
                'coords': coords
            })
        return result
