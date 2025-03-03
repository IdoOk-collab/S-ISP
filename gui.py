"""
Simple ISP Tuning Tool - GUI Module
A PyQt5-based graphical interface for real-time image processing with modular settings.

Author: Ido Okashi
Date: February 24, 2025
"""

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget, QSizePolicy, QScrollArea, QGroupBox, QStatusBar, QMessageBox
)
import cv2
import json
import os
from isp_processing import convert_cv_qt, generate_histogram, compute_frequency_image, apply_fourier_filter, apply_sharpening, adjust_rgb_gain, apply_lens_shading, adjust_white_balance, apply_noise_reduction, apply_gamma_correction

class NoScrollSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()

class ImageDisplayWindow(QWidget):
    def __init__(self, original_image, processed_image):
        super().__init__()
        self.setWindowTitle("Image Comparison")
        self.setWindowState(Qt.WindowMaximized)

        print("Initializing ImageDisplayWindow")  # Debugging
        
        self.orig_pixmap = convert_cv_qt(original_image)
        self.proc_pixmap = convert_cv_qt(processed_image)

        self.orig_label = QLabel(self, alignment=Qt.AlignCenter, sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.proc_label = QLabel(self, alignment=Qt.AlignCenter, sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
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

        self.resizeEvent(None)

    def resizeEvent(self, event):
        if hasattr(self, 'orig_pixmap'):
            available_height = self.height() - 200
            available_width = self.width() // 2 - 20
            for label, pixmap in [(self.orig_label, self.orig_pixmap), (self.proc_label, self.proc_pixmap)]:
                label.setPixmap(pixmap.scaled(available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ido Okashi - Simple ISP")
        self.setWindowState(Qt.WindowMaximized)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        scroll_area = QScrollArea(widgetResizable=True)
        scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        QVBoxLayout(central_widget).addWidget(scroll_area)

        self.load_button = QPushButton("Load Image")
        self.load_button.setFixedSize(150, 50)
        self.load_button.clicked.connect(self.load_image)
        self.save_preset_button = QPushButton("Save Preset")
        self.save_preset_button.setFixedSize(150, 50)
        self.save_preset_button.clicked.connect(self.save_preset)
        self.load_preset_button = QPushButton("Load Preset")
        self.load_preset_button.setFixedSize(150, 50)
        self.load_preset_button.clicked.connect(self.load_preset)
        
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.save_preset_button)
        top_layout.addWidget(self.load_preset_button)
        top_layout.addStretch()
        self.scroll_layout.addLayout(top_layout)

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
            ]
        }

        self.groups = {}
        self.controls = {}
        for group_name, controls in self.settings.items():
            group = QGroupBox(group_name)
            group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
            layout = QVBoxLayout(group)
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
                    spin_layout.addWidget(QLabel(control["label"]))
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
            self.scroll_layout.addWidget(group)

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

        # Debounce timer
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.apply_processing)

        # Status bar for performance metrics
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.scroll_layout.addStretch()

    def add_slider(self, layout, text, min_val, max_val, initial_value=0, enabled=False):
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(QLabel(text))
        slider = NoScrollSlider(Qt.Horizontal, minimum=min_val, maximum=max_val, enabled=enabled)
        slider.setValue(initial_value)
        value_label = QLabel(str(slider.value()))
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        slider.valueChanged.connect(self.queue_processing)
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        layout.addLayout(slider_layout)
        return slider

    def queue_processing(self):
        """Queue processing with a debounce delay."""
        self.timer.start(5)  # 50ms debounce delay

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
                print("Creating new ImageDisplayWindow")
                self.image_window = ImageDisplayWindow(self.original_image, self.processed_image)
            else:
                print("Reusing existing ImageDisplayWindow")
                self.image_window.orig_pixmap = convert_cv_qt(self.original_image)
                self.image_window.proc_pixmap = convert_cv_qt(self.processed_image)
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
        
        # Show pop-up notification with auto-close
        msg = QMessageBox(self)
        msg.setWindowTitle("Preset Saved")
        msg.setText(f"Presets Saved Successfully!\nFile: {preset_path}")
        msg.setStandardButtons(QMessageBox.NoButton)
        
        close_timer = QTimer(self)
        close_timer.setSingleShot(True)
        close_timer.timeout.connect(msg.accept)
        close_timer.start(4000)  # 4000 ms = 4 seconds
        
        print("Showing QMessageBox")
        msg.exec_()
        print("QMessageBox should have closed")

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
        import time
        start_time = time.time()
        
        if self.original_image is None:
            return
        self.processed_image = self.original_image.copy()

        for group_name, controls in self.settings.items():
            for control in controls:
                if "process" in control:
                    widget = self.controls[control["name"]]
                    value = widget.isChecked() if control["type"] == "checkbox" else widget.value()
                    self.processed_image = control["process"](self.processed_image, value)

        if self.controls["fourier_checkbox"].isChecked():
            inner = self.controls["fourier_inner_slider"].value()
            outer = self.controls["fourier_outer_slider"].value()
            if not self.freq_layout_added:
                self.groups["Fourier Settings"].layout().addLayout(self.freq_layout)
                self.freq_layout_added = True
            
            orig_freq = self.original_freq_cache
            proc_freq = compute_frequency_image(self.processed_image, inner=inner, outer=outer)
            display_size = 500
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
            self.image_window.resizeEvent(None)
            self.image_window.proc_hist_label.setPixmap(generate_histogram(self.processed_image))
        
        elapsed = (time.time() - start_time) * 1000  # ms
        self.status_bar.showMessage(f"Processing time: {elapsed:.1f} ms")