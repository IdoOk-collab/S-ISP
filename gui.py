"""
Simple ISP Tuning Tool - GUI Module
A PyQt5-based graphical interface for real-time image and video processing with modular settings.

Author: Ido Okashi
Date: February 24, 2025
"""

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget, QSizePolicy, QScrollArea, QGroupBox, QStatusBar, QMessageBox
)
import cv2
import json
import os
import numpy as np
from isp_processing import (
    convert_cv_qt, generate_histogram, compute_frequency_image, apply_fourier_filter, apply_sharpening,
    adjust_rgb_gain, apply_lens_shading, adjust_white_balance, apply_noise_reduction, apply_gamma_correction,
    apply_tone_mapping, apply_edge_detection, apply_orb_keypoints, resize_image_720p
)

# Custom slider to disable scrolling
class NoScrollSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()

# Thread for capturing video frames (live stream or recorded video)
class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, source=None):
        super().__init__()
        self.source = source
        self.running = False

    def run(self):
        print(f"Starting VideoCaptureThread with source: {self.source}")
        if self.source is None:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW, [])
        else:
            if not isinstance(self.source, str):
                print(f"Error: Invalid video source type: {type(self.source)}. Expected str.")
                return
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [])

        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.source if self.source else 'webcam'}")
            return
        
        self.running = True
        frame_count = 0
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from video source")
                break
            frame, _, _ = resize_image_720p(frame)
            frame_count += 1
            print(f"Emitting frame {frame_count}")
            self.frame_ready.emit(frame)
            self.msleep(33)  # ~30 FPS
        print(f"VideoCaptureThread stopped. Total frames: {frame_count}")
        cap.release()

    def stop(self):
        print("Stopping VideoCaptureThread")
        self.running = False
        self.wait()

# Thread for processing frames (supports both image and video modes)
class ProcessingThread(QThread):
    result = pyqtSignal(object)

    def __init__(self, parent, settings, controls, is_video=False):
        super().__init__(parent)
        self.settings = settings
        self.controls = controls
        self.is_video = is_video
        self.frame = None
        self.running = False

    def set_frame(self, frame):
        self.frame = frame

    def run(self):
        if not self.is_video:
            # Single-image mode
            if self.frame is None:
                return
            processed = self.process_frame(self.frame)
            self.result.emit(processed)
        else:
            # Video mode: continuous processing
            self.running = True
            while self.running and self.frame is not None:
                processed = self.process_frame(self.frame)
                self.result.emit(processed)
                self.msleep(10)  # Control processing rate

    def process_frame(self, frame):
        import time
        start_time = time.time()
        processed_image = frame.copy()
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
        elapsed = (time.time() - start_time) * 1000  # Calculate elapsed time in milliseconds
        return (processed_image, keypoint_count, keypoint_strength, edge_density, elapsed)

    def stop(self):
        self.running = False
        self.wait()

# Custom label for clickable zoom
class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.crop_active = False
        self.crop_pos = QPoint(0, 0)
        self.crop_size = 100
        self.pixmap_base = None

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
            self.update_crop()

    def update_crop(self):
        if not self.pixmap_base or self.pixmap_base.isNull():
            return
        pixmap = self.pixmap_base.copy()
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        x, y = self.crop_pos.x(), self.crop_pos.y()
        half_size = self.crop_size // 2
        painter.drawRect(x - half_size, y - half_size, self.crop_size, self.crop_size)
        painter.end()
        self.setPixmap(pixmap)
        if hasattr(self.parent(), 'show_zoom'):
            self.parent().show_zoom(self, x, y)

# Window to display original and processed frames
class ImageDisplayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Comparison")
        self.setWindowState(Qt.WindowMaximized)
        
        self.original_image = None
        self.processed_image = None
        self.orig_pixmap = None
        self.proc_pixmap = None

        self.orig_label = ClickableLabel(self)
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.proc_label = ClickableLabel(self)
        self.proc_label.setAlignment(Qt.AlignCenter)
        self.proc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.orig_hist_label = QLabel(self)
        self.orig_hist_label.setAlignment(Qt.AlignCenter)
        self.orig_hist_label.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.proc_hist_label = QLabel(self)
        self.proc_hist_label.setAlignment(Qt.AlignCenter)
        self.proc_hist_label.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.orig_hist_label.setFixedHeight(150)
        self.proc_hist_label.setFixedHeight(150)

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

        self.zoom_popup = None

    def update_frame(self, result):
        if isinstance(result, tuple):
            self.processed_image, _, _, _, _ = result
            self.original_image = self.original_image if self.original_image is not None else self.processed_image
            self.orig_pixmap = convert_cv_qt(self.original_image)
            self.proc_pixmap = convert_cv_qt(self.processed_image)
            self.resizeEvent(None)
            self.proc_hist_label.setPixmap(generate_histogram(self.processed_image))

    def resizeEvent(self, event):
        if self.orig_pixmap and self.proc_pixmap:
            available_width = self.width() // 2 - 20
            available_height = self.height() - 200
            self.orig_label.pixmap_base = self.orig_pixmap.scaled(available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.proc_label.pixmap_base = self.proc_pixmap.scaled(available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.orig_label.setPixmap(self.orig_label.pixmap_base)
            self.proc_label.setPixmap(self.proc_label.pixmap_base)

    def show_zoom(self, label, x, y):
        img = self.original_image if label == self.orig_label else self.processed_image
        side = "Original" if label == self.orig_label else "Processed"
        img_w, img_h = img.shape[1], img.shape[0]
        label_w, label_h = label.pixmap_base.width(), label.pixmap_base.height()
        scale_x, scale_y = img_w / label_w, img_h / label_h
        img_x, img_y = int(x * scale_x), int(y * scale_y)

        crop_size = 50
        crop = img[max(0, img_y - crop_size):min(img_h, img_y + crop_size), 
                   max(0, img_x - crop_size):min(img_w, img_x + crop_size)]

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

        self.zoom_label.setPixmap(convert_cv_qt(crop).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_popup.move(label.mapToGlobal(QPoint(x, y)) + QPoint(10, 10))

# Main application window
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

        # Top controls
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Image", "Live Stream", "Video File"])
        self.mode_selector.currentTextChanged.connect(self.switch_mode)
        top_layout.addWidget(self.mode_selector)

        self.load_button = QPushButton("Load Image")
        self.load_button.setFixedSize(150, 50)
        self.load_button.clicked.connect(self.load_image)
        top_layout.addWidget(self.load_button)

        self.save_preset_button = QPushButton("Save Preset")
        self.save_preset_button.setFixedSize(150, 50)
        self.save_preset_button.clicked.connect(self.save_preset)
        top_layout.addWidget(self.save_preset_button)

        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.setFixedSize(150, 50)
        self.save_image_button.clicked.connect(self.save_image)
        top_layout.addWidget(self.save_image_button)

        self.load_preset_button = QPushButton("Load Preset")
        self.load_preset_button.setFixedSize(150, 50)
        self.load_preset_button.clicked.connect(self.load_preset)
        top_layout.addWidget(self.load_preset_button)

        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        # Two-column layout
        columns_layout = QHBoxLayout()
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()
        columns_layout.addLayout(left_column)
        columns_layout.addLayout(right_column)
        main_layout.addLayout(columns_layout)

        # Settings definition
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
                 "tooltip": "Sets the inner frequency cutoff"},
                {"type": "slider", "name": "fourier_outer_slider", "label": "Fourier Outer Cutoff", "min": 0, "max": 100, 
                 "tooltip": "Sets the outer frequency cutoff"}
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
                 "tooltip": "Adjusts brightness/contrast non-linearly"}
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
                 "tooltip": "Overlays Canny edges"},
                {"type": "slider", "name": "edge_low_slider", "label": "Edge Low Threshold", "min": 0, "max": 255, "default": 50,
                 "tooltip": "Lower threshold for edge detection"},
                {"type": "slider", "name": "edge_high_slider", "label": "Edge High Threshold", "min": 0, "max": 255, "default": 150,
                 "tooltip": "Upper threshold for edge detection"},
                {"type": "checkbox", "name": "keypoints_checkbox", "label": "Show Keypoints", 
                 "process": lambda img, val: apply_orb_keypoints(img, self.controls["keypoints_slider"].value(), val), 
                 "tooltip": "Overlays ORB keypoints"},
                {"type": "slider", "name": "keypoints_slider", "label": "Max Keypoints", "min": 100, "max": 1000, "default": 500,
                 "tooltip": "Controls maximum keypoints detected"}
            ]
        }

        self.groups = {}
        self.controls = {}
        left_groups = ["Basic Adjustments", "White Balance", "Noise Reduction", "Sharpening Settings", "Tone Mapping"]
        
        for group_name, controls in self.settings.items():
            group = QGroupBox(group_name)
            group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 5px; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }")
            layout = QVBoxLayout(group)
            layout.setSpacing(10)
            layout.setContentsMargins(10, 10, 10, 10)
            for control in controls:
                if control["type"] == "slider":
                    widget = self.add_slider(layout, control["label"], control["min"], control["max"], control.get("default", 0))
                    widget.setToolTip(control.get("tooltip", ""))
                elif control["type"] == "spinbox":
                    widget = QSpinBox(minimum=control["min"], maximum=control["max"], singleStep=control["step"], value=control["default"], enabled=False)
                    widget.setFixedWidth(60)
                    widget.setToolTip(control.get("tooltip", ""))
                    sub_layout = QHBoxLayout()
                    label = QLabel(control["label"])
                    label.setFixedWidth(150)
                    sub_layout.addWidget(label)
                    sub_layout.addWidget(widget)
                    sub_layout.addStretch()
                    layout.addLayout(sub_layout)
                    widget.valueChanged.connect(self.queue_processing)
                elif control["type"] == "checkbox":
                    widget = QCheckBox(control["label"], enabled=False)
                    widget.stateChanged.connect(self.queue_processing)
                    widget.setToolTip(control.get("tooltip", ""))
                    layout.addWidget(widget)
                self.controls[control["name"]] = widget
            self.groups[group_name] = group
            (left_column if group_name in left_groups else right_column).addWidget(group)

        # Frequency display
        self.freq_layout = QHBoxLayout()
        for side, title in [("original", "Original Frequency Domain"), ("processed", "Processed Frequency Domain")]:
            vbox = QVBoxLayout()
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
            setattr(self, f"{side}_freq_title", title_label)
            setattr(self, f"{side}_freq_label", label)
            vbox.addWidget(title_label)
            vbox.addWidget(label)
            self.freq_layout.addLayout(vbox)

        self.original_image = None
        self.processed_image = None
        self.image_window = None
        self.current_image_path = None
        self.freq_layout_added = False
        self.original_freq_cache = None
        self.video_thread = None
        self.is_video_mode = False

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        main_layout.addStretch()

    def add_slider(self, layout, text, min_val, max_val, initial_value=0, enabled=False):
        slider_layout = QHBoxLayout()
        slider = NoScrollSlider(Qt.Horizontal, minimum=min_val, maximum=max_val, enabled=enabled)
        slider.setFixedWidth(250)
        slider.setValue(initial_value)
        value_label = QLabel(str(initial_value))
        value_label.setFixedWidth(50)
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        slider.valueChanged.connect(self.queue_processing)
        label = QLabel(text)
        label.setFixedWidth(150)
        slider_layout.addWidget(label)
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        slider_layout.addStretch()
        layout.addLayout(slider_layout)
        return slider

    def switch_mode(self, mode):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        self.is_video_mode = mode in ["Live Stream", "Video File"]
        if mode == "Image":
            self.load_button.setText("Load Image")
            self.load_button.clicked.disconnect()
            self.load_button.clicked.connect(self.load_image)
        elif mode == "Live Stream":
            self.load_button.setText("Start Stream")
            self.load_button.clicked.disconnect()
            self.load_button.clicked.connect(lambda: self.start_video())  # Ignore clicked arg
        elif mode == "Video File":
            self.load_button.setText("Load Video")
            self.load_button.clicked.disconnect()
            self.load_button.clicked.connect(self.load_video)

    def start_video(self, source=None):
        if self.video_thread:
            self.video_thread.stop()
        self.video_thread = VideoCaptureThread(source)
        self.video_thread.frame_ready.connect(self.handle_frame)
        self.video_thread.start()
        self.queue_processing()

    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video files (*.mp4 *.avi *.mov)")
        if fname:
            self.current_image_path = fname  # For preset saving
            self.start_video(fname)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image files (*.jpg *.png *.bmp)")
        if fname:
            self.original_image = cv2.imread(fname)
            if self.original_image is None:
                self.status_bar.showMessage("Failed to load image.", 3000)
                return
            self.current_image_path = fname
            self.processed_image = self.original_image.copy()
            self.is_video_mode = False
            for widget in self.controls.values():
                widget.setEnabled(True)
            self.original_freq_cache = compute_frequency_image(self.original_image)
            self.apply_processing()
            if self.image_window is None:
                self.image_window = ImageDisplayWindow()
            self.image_window.update_frame((self.processed_image, 0, 0.0, 0.0, 0))
            self.image_window.show()
            self.image_window.raise_()
            self.image_window.activateWindow()

    def handle_frame(self, frame):
        self.original_image = frame
        if self.image_window is None:
            self.image_window = ImageDisplayWindow()
            self.image_window.show()
        if not self.is_video_mode:
            self.original_freq_cache = compute_frequency_image(self.original_image)
        if hasattr(self, 'processing_thread') and not self.processing_thread.isRunning():
            self.processing_thread.set_frame(self.original_image)
            self.processing_thread.start()
        else:
            self.processing_thread = ProcessingThread(self, self.settings, self.controls, self.is_video_mode)
            self.processing_thread.set_frame(self.original_image)
            self.processing_thread.result.connect(self.update_processed_image)
            self.processing_thread.start()
        for widget in self.controls.values():
            widget.setEnabled(True)

    def queue_processing(self):
        if not self.is_video_mode:
            self.processing_thread = ProcessingThread(self, self.settings, self.controls)
            self.processing_thread.set_frame(self.original_image)
            self.processing_thread.result.connect(self.update_processed_image)
            self.processing_thread.start()

    def update_processed_image(self, result):
        self.processed_image, keypoint_count, keypoint_strength, edge_density, elapsed = result
        
        if self.controls["fourier_checkbox"].isChecked():
            inner, outer = self.controls["fourier_inner_slider"].value(), self.controls["fourier_outer_slider"].value()
            if not self.freq_layout_added:
                self.groups["Fourier Settings"].layout().addLayout(self.freq_layout)
                self.freq_layout_added = True
            self.original_freq_label.setPixmap(convert_cv_qt(self.original_freq_cache).scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.processed_freq_label.setPixmap(convert_cv_qt(compute_frequency_image(self.processed_image, inner, outer)).scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            for w in ("original_freq_title", "processed_freq_title", "original_freq_label", "processed_freq_label"):
                getattr(self, w).setVisible(True)
            self.controls["fourier_inner_slider"].setEnabled(True)
            self.controls["fourier_outer_slider"].setEnabled(True)
        else:
            if self.freq_layout_added:
                self.groups["Fourier Settings"].layout().removeItem(self.freq_layout)
                self.freq_layout_added = False
            for w in ("original_freq_title", "processed_freq_title", "original_freq_label", "processed_freq_label"):
                getattr(self, w).setVisible(False)
            for w in ("fourier_inner_slider", "fourier_outer_slider"):
                self.controls[w].setValue(0)
                self.controls[w].setEnabled(False)

        if self.image_window:
            self.image_window.update_frame(result)
        
        self.status_bar.showMessage(
            f"Processing time: {elapsed:.1f} ms | Keypoints: {keypoint_count} | "
            f"Avg Strength: {keypoint_strength:.4f} | Edge Density: {edge_density:.1%}"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.controls["fourier_checkbox"].isChecked() and self.freq_layout_added:
            size = min(self.width() // 2 - 20, self.height() - 200)
            for label in (self.original_freq_label, self.processed_freq_label):
                if label.pixmap() and not label.pixmap().isNull():
                    label.setPixmap(label.pixmap().scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_preset(self):
        if self.original_image is None:
            self.status_bar.showMessage("No image or video loaded to save preset for.", 3000)
            return
        base = os.path.splitext(os.path.basename(self.current_image_path))[0]
        preset_path = os.path.join(os.path.dirname(self.current_image_path), f"{base}.json")
        counter = 2
        while os.path.exists(preset_path):
            preset_path = os.path.join(os.path.dirname(self.current_image_path), f"{base}{counter}.json")
            counter += 1
        preset = {name: w.value() if isinstance(w, (QSlider, QSpinBox)) else w.isChecked() for name, w in self.controls.items()}
        with open(preset_path, 'w') as f:
            json.dump(preset, f)
        msg = QMessageBox(self)
        msg.setWindowTitle("Preset Saved")
        msg.setText(f"Presets Saved!\nFile: {preset_path}")
        msg.setStandardButtons(QMessageBox.NoButton)
        QTimer.singleShot(4000, msg.accept)
        msg.exec_()

    def save_image(self):
        if self.processed_image is None:
            self.status_bar.showMessage("No processed image to save.", 3000)
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Processed Image", "", "Image files (*.jpg *.png *.bmp)")
        if fname:
            try:
                cv2.imwrite(fname, self.processed_image)
                self.status_bar.showMessage(f"Image saved to {fname}", 3000)
            except Exception as e:
                self.status_bar.showMessage(f"Error: {str(e)}", 3000)

    def load_preset(self):
        if self.original_image is None:
            self.status_bar.showMessage("Load an image or video first.", 3000)
            return
        fname, _ = QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON files (*.json)")
        if fname and os.path.exists(fname):
            with open(fname, 'r') as f:
                preset = json.load(f)
            for name, val in preset.items():
                if name in self.controls:
                    widget = self.controls[name]
                    if isinstance(widget, (QSlider, QSpinBox)):
                        widget.setValue(val)
                    else:
                        widget.setChecked(val)
            self.apply_processing()
            self.status_bar.showMessage(f"Preset loaded from {fname}", 3000)

    def apply_processing(self):
        self.queue_processing()

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        if hasattr(self, 'processing_thread'):
            self.processing_thread.stop()
        super().closeEvent(event)