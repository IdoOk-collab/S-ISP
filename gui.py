"""
Simple ISP Tuning Tool - GUI Module
A PyQt5-based graphical interface for real-time image processing with modular settings.

Author: Ido Okashi
Date: February 24, 2025
"""

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget, QSizePolicy, QScrollArea, QGroupBox, QStatusBar, QMessageBox
)
import cv2
import json
import os
import numpy as np
from isp_processing import convert_cv_qt, generate_histogram, compute_frequency_image, apply_fourier_filter, apply_sharpening, adjust_rgb_gain, apply_lens_shading, adjust_white_balance, apply_noise_reduction, apply_gamma_correction, apply_tone_mapping, apply_edge_detection, apply_orb_keypoints

class NoScrollSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()

class ProcessingThread(QThread):
    result = pyqtSignal(object)

    def __init__(self, parent, original_image, settings, controls):
        super().__init__(parent)
        self.original_image = original_image
        self.settings = settings
        self.controls = controls

    def run(self):
        import time
        start_time = time.time()
        if self.original_image is None:
            return
        processed_image = self.original_image.copy()
        keypoint_count = 0
        keypoint_strength = 0.0
        edge_density = 0.0

        for group_name, controls in self.settings.items():
            for control in controls:
                if "process" in control:
                    widget = self.controls[control["name"]]
                    value = widget.isChecked() if control["type"] == "checkbox" else widget.value()
                    if control["name"] == "keypoints_checkbox":
                        processed_image, keypoint_count, keypoint_strength = control["process"](processed_image, value)
                    elif control["name"] == "edge_detection_checkbox":
                        processed_image, edge_density = control["process"](processed_image, value)
                    else:
                        processed_image = control["process"](processed_image, value)

        elapsed = (time.time() - start_time) * 1000
        self.result.emit((processed_image, keypoint_count, keypoint_strength, edge_density, elapsed))

class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.crop_active = False
        self.crop_pos = QPoint(0, 0)
        self.crop_size = 100  # 100x100 crop
        self.pixmap_base = None  # Base pixmap without rectangle

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.crop_active = True
            self.crop_pos = event.pos()
            self.update_crop()

    def mouseMoveEvent(self, event):
        if self.crop_active:
            self.crop_pos = event.pos()
            self.update_crop()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.crop_active = False
            # Optionally keep the last crop visible or reset
            self.update_crop()

    def update_crop(self):
        if not self.pixmap_base or self.pixmap_base.isNull():
            return
        pixmap = self.pixmap_base.copy()
        painter = QPainter(pixmap)
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        x = self.crop_pos.x()
        y = self.crop_pos.y()
        half_size = self.crop_size // 2
        rect_x = x - half_size
        rect_y = y - half_size
        painter.drawRect(rect_x, rect_y, self.crop_size, self.crop_size)
        painter.end()
        self.setPixmap(pixmap)

        # Show zoomed crop in popup
        if hasattr(self.parent(), 'show_zoom'):
            self.parent().show_zoom(self, x, y)

class ImageDisplayWindow(QWidget):
    def __init__(self, original_image, processed_image):
        super().__init__()
        self.setWindowTitle("Image Comparison")
        self.setWindowState(Qt.WindowMaximized)
        
        self.original_image = original_image
        self.processed_image = processed_image
        
        self.orig_pixmap = convert_cv_qt(original_image)
        self.proc_pixmap = convert_cv_qt(processed_image)

        self.orig_label = ClickableLabel(self)
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.proc_label = ClickableLabel(self)
        self.proc_label.setAlignment(Qt.AlignCenter)
        self.proc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.orig_hist_label = QLabel(self, alignment=Qt.AlignCenter, sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.proc_hist_label = QLabel(self, alignment=Qt.AlignCenter, sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        
        self.orig_hist_label.setFixedHeight(150)
        self.proc_hist_label.setFixedHeight(150)
        self.orig_hist_label.setPixmap(generate_histogram(original_image))
        self.proc_hist_label.setPixmap(generate_histogram(processed_image))

        orig_layout = QVBoxLayout()
        orig_layout.addWidget(self.orig_label)
        orig_layout.addWidget(self.orig_hist_label)

        proc_layout = QVBoxLayout()
        proc_layout.addWidget(self.proc_label)
        proc_layout.addWidget(self.proc_hist_label)

        main_layout = QHBoxLayout()
        main_layout.addLayout(orig_layout)
        main_layout.addLayout(proc_layout)
        self.setLayout(main_layout)

        self.zoom_popup = None  # Track single popup

        self.resizeEvent(None)

    def resizeEvent(self, event):
        if hasattr(self, 'orig_pixmap'):
            available_height = self.height() - 200
            available_width = self.width() // 2 - 20
            self.orig_label.pixmap_base = self.orig_pixmap.scaled(available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.proc_label.pixmap_base = self.proc_pixmap.scaled(available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.orig_label.setPixmap(self.orig_label.pixmap_base)
            self.proc_label.setPixmap(self.proc_label.pixmap_base)

    def show_zoom(self, label, x, y):
        # Convert label coordinates to image coordinates
        if label == self.orig_label:
            img = self.original_image
            side = "Original"
        else:
            img = self.processed_image
            side = "Processed"

        img_w, img_h = img.shape[1], img.shape[0]
        label_w, label_h = label.pixmap_base.width(), label.pixmap_base.height()
        scale_x = img_w / label_w
        scale_y = img_h / label_h
        img_x = int(x * scale_x)
        img_y = int(y * scale_y)

        # Crop 100x100 region
        crop_size = 50  # Half of 100x100
        y_start = max(0, img_y - crop_size)
        y_end = min(img_h, img_y + crop_size)
        x_start = max(0, img_x - crop_size)
        x_end = min(img_w, img_x + crop_size)
        crop = img[y_start:y_end, x_start:x_end]

        # Update or create popup
        if self.zoom_popup is None or not self.zoom_popup.isVisible():
            self.zoom_popup = QWidget()
            self.zoom_popup.setWindowTitle(f"{side} Zoom")
            self.zoom_popup.setFixedSize(200, 200)
            self.zoom_label = QLabel(self.zoom_popup)
            layout = QVBoxLayout(self.zoom_popup)
            layout.addWidget(self.zoom_label)
            self.zoom_popup.show()
        else:
            self.zoom_popup.setWindowTitle(f"{side} Zoom at ({img_x}, {img_y})")

        zoom_pixmap = convert_cv_qt(crop).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.zoom_label.setPixmap(zoom_pixmap)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        
        # Position popup next to mouse
        global_pos = label.mapToGlobal(QPoint(x, y))
        self.zoom_popup.move(global_pos + QPoint(10, 10))  # Slight offset

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ido Okashi - Simple ISP")
        self.setWindowState(Qt.WindowMaximized)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        scroll_area = QScrollArea(widgetResizable=True)
        scroll_widget = QWidget()
        main_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        QVBoxLayout(central_widget).addWidget(scroll_area)

        # Top button layout
        self.load_button = QPushButton("Load Image")
        self.load_button.setFixedSize(150, 50)
        self.load_button.clicked.connect(self.load_image)
        self.save_preset_button = QPushButton("Save Preset")
        self.save_preset_button.setFixedSize(150, 50)
        self.save_preset_button.clicked.connect(self.save_preset)
        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.setFixedSize(150, 50)
        self.save_image_button.clicked.connect(self.save_image)
        self.load_preset_button = QPushButton("Load Preset")
        self.load_preset_button.setFixedSize(150, 50)
        self.load_preset_button.clicked.connect(self.load_preset)
        
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.save_preset_button)
        top_layout.addWidget(self.save_image_button)
        top_layout.addWidget(self.load_preset_button)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        # Two-column layout for group boxes
        columns_layout = QHBoxLayout()
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()
        columns_layout.addLayout(left_column)
        columns_layout.addLayout(right_column)
        main_layout.addLayout(columns_layout)

        # Define settings
        self.settings = {
            "Basic Adjustments": [
                {"type": "slider", "name": "histogram_slider", "label": "Histogram Gain", "min": -100, "max": 100, 
                 "process": lambda img, val: cv2.convertScaleAbs(img, alpha=1 + val / 100.0), 
                 "tooltip": "Adjusts overall image brightness and contrast"},
                {"type": "slider", "name": "blur_slider", "label": "Gaussian Blur", "min": 0, "max": 50, 
                 "process": lambda img, val: cv2.GaussianBlur(img, (self.controls["kernel_size_selector"].value() | 1, self.controls["kernel_size_selector"].value() | 1), val * 0.05) if val > 0 else img, 
                 "tooltip": "Smooths the image to reduce noise or detail"},
                {"type": "spinbox", "name": "kernel_size_selector", "label": "Kernel Size", "min": 1, "max": 51, "step": 2, "default": 3, 
                 "tooltip": "Sets the size of the blur kernel"}
            ],
            "Fourier Settings": [
                {"type": "checkbox", "name": "fourier_checkbox", "label": "Apply Fourier Filter", 
                 "process": lambda img, val: apply_fourier_filter(img, self.controls["fourier_inner_slider"].value(), self.controls["fourier_outer_slider"].value()) if val else img, 
                 "tooltip": "Enables frequency domain filtering"},
                {"type": "slider", "name": "fourier_inner_slider", "label": "Fourier Inner Cutoff", "min": 0, "max": 100, 
                 "tooltip": "Sets the inner frequency cutoff for filtering"},
                {"type": "slider", "name": "fourier_outer_slider", "label": "Fourier Outer Cutoff", "min": 0, "max": 100, 
                 "tooltip": "Sets the outer frequency cutoff for filtering"}
            ],
            "RGB Gain Control": [
                {"type": "slider", "name": "red_gain_slider", "label": "Red Gain", "min": 0, "max": 200, "default": 100,
                 "process": lambda img, val: adjust_rgb_gain(img, val / 100.0, 'R'), 
                 "tooltip": "Adjusts red channel intensity"},
                {"type": "slider", "name": "green_gain_slider", "label": "Green Gain", "min": 0, "max": 200, "default": 100,
                 "process": lambda img, val: adjust_rgb_gain(img, val / 100.0, 'G'), 
                 "tooltip": "Adjusts green channel intensity"},
                {"type": "slider", "name": "blue_gain_slider", "label": "Blue Gain", "min": 0, "max": 200, "default": 100,
                 "process": lambda img, val: adjust_rgb_gain(img, val / 100.0, 'B'), 
                 "tooltip": "Adjusts blue channel intensity"}
            ],
            "White Balance": [
                {"type": "slider", "name": "temperature_slider", "label": "Temperature", "min": -100, "max": 100, 
                 "process": lambda img, val: adjust_white_balance(img, val / 100.0, self.controls["tint_slider"].value() / 100.0), 
                 "tooltip": "Adjusts color temperature (cool to warm)"},
                {"type": "slider", "name": "tint_slider", "label": "Tint", "min": -100, "max": 100, 
                 "process": lambda img, val: adjust_white_balance(img, self.controls["temperature_slider"].value() / 100.0, val / 100.0), 
                 "tooltip": "Adjusts tint (magenta to green)"}
            ],
            "Noise Reduction": [
                {"type": "slider", "name": "noise_reduction_slider", "label": "Noise Reduction Strength", "min": 0, "max": 100, 
                 "process": lambda img, val: apply_noise_reduction(val / 100.0, img), 
                 "tooltip": "Reduces image noise while preserving edges"}
            ],
            "Gamma Correction": [
                {"type": "slider", "name": "gamma_slider", "label": "Gamma", "min": 10, "max": 500, "default": 100,
                 "process": lambda img, val: apply_gamma_correction(val / 100.0, img), 
                 "tooltip": "Adjusts brightness/contrast non-linearly (0.1 darkens, >1 brightens)"}
            ],
            "Lens Shading Correction": [
                {"type": "slider", "name": "lens_shading_slider", "label": "Lens Shading Strength", "min": -100, "max": 100, 
                 "process": lambda img, val: apply_lens_shading(val / 100.0, img), 
                 "tooltip": "Adjusts edge intensity (darken or brighten)"}
            ],
            "Sharpening Settings": [
                {"type": "slider", "name": "sharpening_slider", "label": "Sharpening Strength", "min": 0, "max": 100, 
                 "process": lambda img, val: apply_sharpening(val, img) if val > 0 else img, 
                 "tooltip": "Enhances image edges and details"}
            ],
            "Tone Mapping": [
                {"type": "slider", "name": "tone_mapping_slider", "label": "Tone Mapping Strength", "min": 0, "max": 100,
                 "process": lambda img, val: apply_tone_mapping(val / 10.0, img),
                 "tooltip": "Simulates HDR by compressing dynamic range"}
            ],
            "Feature Extraction": [
                {"type": "checkbox", "name": "edge_detection_checkbox", "label": "Show Edges", 
                 "process": lambda img, val: apply_edge_detection(img, self.controls["edge_low_slider"].value(), self.controls["edge_high_slider"].value(), val), 
                 "tooltip": "Overlays Canny edges to highlight structure"},
                {"type": "slider", "name": "edge_low_slider", "label": "Edge Low Threshold", "min": 0, "max": 255, "default": 50,
                 "tooltip": "Lower threshold for edge detection sensitivity"},
                {"type": "slider", "name": "edge_high_slider", "label": "Edge High Threshold", "min": 0, "max": 255, "default": 150,
                 "tooltip": "Upper threshold for edge detection strength"},
                {"type": "checkbox", "name": "keypoints_checkbox", "label": "Show Keypoints", 
                 "process": lambda img, val: apply_orb_keypoints(img, self.controls["keypoints_slider"].value(), val), 
                 "tooltip": "Overlays ORB keypoints to highlight distinctive features"},
                {"type": "slider", "name": "keypoints_slider", "label": "Max Keypoints", "min": 100, "max": 1000, "default": 500,
                 "tooltip": "Controls the maximum number of keypoints detected"}
            ]
        }

        self.groups = {}
        self.controls = {}
        
        # Assign group boxes to columns
        left_groups = ["Basic Adjustments", "White Balance", "Noise Reduction", "Sharpening Settings", "Tone Mapping"]
        right_groups = ["Fourier Settings", "RGB Gain Control", "Gamma Correction", "Lens Shading Correction", "Feature Extraction"]

        for group_name, controls in self.settings.items():
            group = QGroupBox(group_name)
            group.setStyleSheet("""
                QGroupBox { 
                    border: 1px solid gray; 
                    border-radius: 5px; 
                    margin-top: 1ex; 
                } 
                QGroupBox::title { 
                    subcontrol-origin: margin; 
                    subcontrol-position: top left; 
                    padding: 0 3px; 
                }
            """)
            layout = QVBoxLayout(group)
            layout.setSpacing(10)
            layout.setContentsMargins(10, 10, 10, 10)
            for control in controls:
                if control["type"] == "slider":
                    initial_value = control.get("default", 0)
                    widget = self.add_slider(layout, control["label"], control["min"], control["max"], initial_value=initial_value)
                    widget.setToolTip(control.get("tooltip", ""))
                elif control["type"] == "spinbox":
                    widget = QSpinBox(minimum=control["min"], maximum=control["max"], singleStep=control["step"], value=control["default"], enabled=False)
                    widget.setFixedWidth(60)
                    widget.setToolTip(control.get("tooltip", ""))
                    spin_layout = QHBoxLayout()
                    label = QLabel(control["label"])
                    label.setFixedWidth(150)
                    spin_layout.addWidget(label)
                    spin_layout.addWidget(widget)
                    spin_layout.addStretch()
                    layout.addLayout(spin_layout)
                    widget.valueChanged.connect(self.queue_processing)
                elif control["type"] == "checkbox":
                    widget = QCheckBox(control["label"], enabled=False)
                    widget.stateChanged.connect(self.queue_processing)
                    widget.setToolTip(control.get("tooltip", ""))
                    layout.addWidget(widget)
                self.controls[control["name"]] = widget
            self.groups[group_name] = group
            if group_name in left_groups:
                left_column.addWidget(group)
            else:
                right_column.addWidget(group)

        # Frequency layout
        self.freq_layout = QHBoxLayout()
        self.original_freq_vbox = QVBoxLayout()
        self.processed_freq_vbox = QVBoxLayout()
        
        self.original_freq_title = QLabel("Original Frequency Domain", alignment=Qt.AlignCenter)
        self.processed_freq_title = QLabel("Processed Frequency Domain", alignment=Qt.AlignCenter)
        self.original_freq_label = QLabel(alignment=Qt.AlignCenter, sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.processed_freq_label = QLabel(alignment=Qt.AlignCenter, sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        
        self.original_freq_vbox.addWidget(self.original_freq_title)
        self.original_freq_vbox.addWidget(self.original_freq_label)
        self.processed_freq_vbox.addWidget(self.processed_freq_title)
        self.processed_freq_vbox.addWidget(self.processed_freq_label)
        
        self.freq_layout.addLayout(self.original_freq_vbox)
        self.freq_layout.addLayout(self.processed_freq_vbox)

        self.original_image = None
        self.processed_image = None
        self.image_window = None
        self.freq_layout_added = False
        self.original_freq_cache = None
        self.current_image_path = None

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        main_layout.addStretch()

    def add_slider(self, layout, text, min_val, max_val, initial_value=0, enabled=False):
        slider_layout = QHBoxLayout()
        label = QLabel(text)
        label.setFixedWidth(150)
        slider = NoScrollSlider(Qt.Horizontal, minimum=min_val, maximum=max_val, enabled=enabled)
        slider.setFixedWidth(250)
        slider.setValue(initial_value)
        value_label = QLabel(str(slider.value()))
        value_label.setFixedWidth(50)
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        slider.valueChanged.connect(self.queue_processing)
        slider_layout.addWidget(label)
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        slider_layout.addStretch()
        layout.addLayout(slider_layout)
        return slider

    def queue_processing(self):
        if not hasattr(self, 'processing_thread') or not self.processing_thread.isRunning():
            self.processing_thread = ProcessingThread(self, self.original_image, self.settings, self.controls)
            self.processing_thread.result.connect(self.update_processed_image)
            self.processing_thread.start()

    def update_processed_image(self, result):
        self.processed_image, keypoint_count, keypoint_strength, edge_density, elapsed = result
        
        # Handle Fourier display
        if self.controls["fourier_checkbox"].isChecked():
            inner = self.controls["fourier_inner_slider"].value()
            outer = self.controls["fourier_outer_slider"].value()
            if not self.freq_layout_added:
                self.groups["Fourier Settings"].layout().addLayout(self.freq_layout)
                self.freq_layout_added = True
            
            orig_freq = self.original_freq_cache
            proc_freq = compute_frequency_image(self.processed_image, inner=inner, outer=outer)
            display_size = 450
            orig_pixmap = convert_cv_qt(orig_freq).scaled(display_size, display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            proc_pixmap = convert_cv_qt(proc_freq).scaled(display_size, display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.original_freq_label.setPixmap(orig_pixmap)
            self.processed_freq_label.setPixmap(proc_pixmap)
            self.original_freq_title.setVisible(True)
            self.processed_freq_title.setVisible(True)
            self.original_freq_label.setVisible(True)
            self.processed_freq_label.setVisible(True)
            self.controls["fourier_inner_slider"].setEnabled(True)
            self.controls["fourier_outer_slider"].setEnabled(True)
        else:
            if self.freq_layout_added:
                self.groups["Fourier Settings"].layout().removeItem(self.freq_layout)
                self.original_freq_title.setVisible(False)
                self.processed_freq_title.setVisible(False)
                self.original_freq_label.setVisible(False)
                self.processed_freq_label.setVisible(False)
                self.freq_layout_added = False
            self.controls["fourier_inner_slider"].setValue(0)
            self.controls["fourier_outer_slider"].setValue(0)
            self.controls["fourier_inner_slider"].setEnabled(False)
            self.controls["fourier_outer_slider"].setEnabled(False)

        if self.image_window:
            self.image_window.proc_pixmap = convert_cv_qt(self.processed_image)
            self.image_window.processed_image = self.processed_image
            self.image_window.resizeEvent(None)
            self.image_window.proc_hist_label.setPixmap(generate_histogram(self.processed_image))
        
        self.status_bar.showMessage(
            f"Processing time: {elapsed:.1f} ms | Keypoints: {keypoint_count} | "
            f"Avg Strength: {keypoint_strength:.4f} | Edge Density: {edge_density:.1%}"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.controls["fourier_checkbox"].isChecked() and self.freq_layout_added:
            for label in [self.original_freq_label, self.processed_freq_label]:
                if label.pixmap() and not label.pixmap().isNull():
                    label.setPixmap(self.scale_frequency_image(label.pixmap()))

    def scale_frequency_image(self, pixmap):
        available_width = self.width() // 2 - 20
        available_height = self.height() - 200
        size = min(available_width, available_height)
        return pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image files (*.jpg *.png *.bmp)")
        if fname:
            self.original_image = cv2.imread(fname)
            if self.original_image is None:
                self.status_bar.showMessage("Failed to load image.", 3000)
                return
            self.current_image_path = fname
            self.processed_image = self.original_image.copy()
            for widget in self.controls.values():
                widget.setEnabled(True)
            self.original_freq_cache = compute_frequency_image(self.original_image)
            self.apply_processing()
            if self.image_window is None:
                self.image_window = ImageDisplayWindow(self.original_image, self.processed_image)
            else:
                self.image_window.orig_pixmap = convert_cv_qt(self.original_image)
                self.image_window.proc_pixmap = convert_cv_qt(self.processed_image)
                self.image_window.original_image = self.original_image
                self.image_window.processed_image = self.processed_image
                self.image_window.resizeEvent(None)
            self.image_window.show()
            self.image_window.raise_()
            self.image_window.activateWindow()

    def save_preset(self):
        if self.original_image is None:
            self.status_bar.showMessage("No image loaded to save preset for.", 3000)
            return
        
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        preset_dir = os.path.dirname(self.current_image_path)
        preset_name = f"{base_name}.json"
        counter = 2
        
        while os.path.exists(os.path.join(preset_dir, preset_name)):
            preset_name = f"{base_name}{counter}.json"
            counter += 1
        
        preset_path = os.path.join(preset_dir, preset_name)
        preset = {name: widget.value() if isinstance(widget, (QSlider, QSpinBox)) else widget.isChecked() 
                  for name, widget in self.controls.items()}
        with open(preset_path, 'w') as f:
            json.dump(preset, f)
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Preset Saved")
        msg.setText(f"Presets Saved Successfully!\nFile: {preset_path}")
        msg.setStandardButtons(QMessageBox.NoButton)
        
        from PyQt5.QtCore import QTimer
        close_timer = QTimer(self)
        close_timer.setSingleShot(True)
        close_timer.timeout.connect(msg.accept)
        close_timer.start(4000)
        
        msg.exec_()

    def save_image(self):
        if self.processed_image is None:
            self.status_bar.showMessage("No processed image to save.", 3000)
            return
        
        fname, _ = QFileDialog.getSaveFileName(self, "Save Processed Image", "", "Image files (*.jpg *.png *.bmp)")
        if fname:
            try:
                success = cv2.imwrite(fname, self.processed_image)
                if success:
                    self.status_bar.showMessage(f"Image saved successfully to {fname}", 3000)
                else:
                    self.status_bar.showMessage("Failed to save image.", 3000)   
            except Exception as e:
                self.status_bar.showMessage(f"Error saving image: {str(e)}", 3000)

    def load_preset(self):
        if self.original_image is None:
            self.status_bar.showMessage("Load an image before applying a preset.", 3000)
            return
        fname, _ = QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON files (*.json)")
        if fname and os.path.exists(fname):
            with open(fname, 'r') as f:
                preset = json.load(f)
            for name, value in preset.items():
                if name in self.controls:
                    widget = self.controls[name]
                    if isinstance(widget, QSlider) or isinstance(widget, QSpinBox):
                        widget.setValue(value)
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(value)
            self.apply_processing()
            self.status_bar.showMessage(f"Preset loaded from {fname}", 3000)

    def apply_processing(self):
        self.queue_processing()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())