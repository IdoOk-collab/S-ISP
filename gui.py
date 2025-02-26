"""
Simple ISP Tuning Tool - GUI Module
A PyQt5-based graphical interface for real-time image processing. Provides controls for
adjusting histogram gain, Gaussian blur, and Fourier filtering, displaying original and
processed images with histograms and optional frequency domain views.

Author: Ido Okashi
Date: February 24, 2025
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget, QSizePolicy, QScrollArea, QGroupBox
)
import cv2
from isp_processing import convert_cv_qt, generate_histogram, compute_frequency_image, apply_fourier_filter

class NoScrollSlider(QSlider):
    """A QSlider subclass that disables mouse wheel interaction to prevent unintended changes."""
    def wheelEvent(self, event):
        """Ignore mouse wheel events, passing them to the parent scroll area."""
        event.ignore()

class ImageDisplayWindow(QWidget):
    """A window displaying original and processed images with their histograms."""
    def __init__(self, original_image, processed_image):
        """Initialize the image comparison window.

        Args:
            original_image (np.ndarray): The original image in BGR format.
            processed_image (np.ndarray): The processed image in BGR format.
        """
        super().__init__()
        self.setWindowTitle("Image Comparison")
        self.setWindowState(Qt.WindowMaximized)

        self.orig_pixmap = convert_cv_qt(original_image)
        self.proc_pixmap = convert_cv_qt(processed_image)

        # Setup image and histogram labels
        self.orig_label = QLabel(self, alignment=Qt.AlignCenter, sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.proc_label = QLabel(self, alignment=Qt.AlignCenter, sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.orig_hist_label = QLabel(self, alignment=Qt.AlignCenter, sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.proc_hist_label = QLabel(self, alignment=Qt.AlignCenter, sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        
        self.orig_hist_label.setFixedHeight(150)
        self.proc_hist_label.setFixedHeight(150)

        self.orig_hist_label.setPixmap(generate_histogram(original_image))
        self.proc_hist_label.setPixmap(generate_histogram(processed_image))

        # Layout for original image and histogram
        orig_layout = QVBoxLayout()
        orig_layout.addWidget(self.orig_label)
        orig_layout.addWidget(self.orig_hist_label)

        # Layout for processed image and histogram
        proc_layout = QVBoxLayout()
        proc_layout.addWidget(self.proc_label)
        proc_layout.addWidget(self.proc_hist_label)

        # Main side-by-side layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(orig_layout)
        main_layout.addLayout(proc_layout)
        self.setLayout(main_layout)

        self.resizeEvent(None)  # Trigger initial scaling

    def resizeEvent(self, event):
        """Scale displayed images when the window is resized.

        Args:
            event (QResizeEvent): The resize event object (unused, can be None for initial call).
        """
        if hasattr(self, 'orig_pixmap'):
            available_height = self.height() - 200  # Account for histogram height
            available_width = self.width() // 2 - 20  # Split width, with padding
            for label, pixmap in [(self.orig_label, self.orig_pixmap), (self.proc_label, self.proc_pixmap)]:
                label.setPixmap(pixmap.scaled(available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))

class MainWindow(QMainWindow):
    """Main application window with image processing controls and display."""
    def __init__(self):
        """Initialize the main window with controls and layout."""
        super().__init__()
        self.setWindowTitle("Ido Okashi - Simple ISP")
        self.setWindowState(Qt.WindowMaximized)

        # Central widget with scrollable content
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        scroll_area = QScrollArea(widgetResizable=True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        QVBoxLayout(central_widget).addWidget(scroll_area)

        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.setFixedSize(150, 50)
        self.load_button.clicked.connect(self.load_image)
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        top_layout.addWidget(self.load_button)
        top_layout.addStretch()
        scroll_layout.addLayout(top_layout)

        # Basic adjustments group
        self.basic_group = QGroupBox("Basic Adjustments")
        self.fourier_group = QGroupBox("Fourier Settings")
        for group in [self.basic_group, self.fourier_group]:
            group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
            scroll_layout.addWidget(group)

        basic_layout = QVBoxLayout(self.basic_group)
        self.histogram_slider = self.add_slider(basic_layout, "Histogram Gain", -100, 100)
        self.blur_slider = self.add_slider(basic_layout, "Gaussian Blur", 0, 50)
        self.kernel_size_selector = QSpinBox(minimum=1, maximum=51, singleStep=2, value=3, enabled=False)
        self.kernel_size_selector.setFixedWidth(60)
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Kernel Size"))
        kernel_layout.addWidget(self.kernel_size_selector)
        kernel_layout.addStretch()
        basic_layout.addLayout(kernel_layout)

        # Fourier settings group
        fourier_layout = QVBoxLayout(self.fourier_group)
        self.fourier_checkbox = QCheckBox("Apply Fourier Filter", enabled=False, stateChanged=self.apply_processing)
        fourier_layout.addWidget(self.fourier_checkbox)
        self.fourier_inner_slider = self.add_slider(fourier_layout, "Fourier Inner Cutoff", 0, 100)
        self.fourier_outer_slider = self.add_slider(fourier_layout, "Fourier Outer Cutoff", 0, 100)

        # Frequency display layout (added dynamically)
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

        # Connect control signals
        for widget in [self.histogram_slider, self.blur_slider, self.kernel_size_selector, self.fourier_inner_slider, self.fourier_outer_slider]:
            widget.valueChanged.connect(self.apply_processing)

        # Initialize image storage
        self.original_image = None
        self.processed_image = None
        self.image_window = None
        self.freq_layout_added = False

        scroll_layout.addStretch()  # Keep content compact at the top

    def add_slider(self, layout, text, min_val, max_val):
        """Add a slider with label and value display to a layout.

        Args:
            layout (QVBoxLayout): The layout to add the slider to.
            text (str): Label text for the slider.
            min_val (int): Minimum value of the slider.
            max_val (int): Maximum value of the slider.

        Returns:
            NoScrollSlider: The created slider widget.
        """
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(QLabel(text))
        slider = NoScrollSlider(Qt.Horizontal, minimum=min_val, maximum=max_val, enabled=False)
        value_label = QLabel(str(slider.value()))
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        layout.addLayout(slider_layout)
        return slider

    def resizeEvent(self, event):
        """Resize frequency domain images if visible when the window size changes.

        Args:
            event (QResizeEvent): The resize event object.
        """
        super().resizeEvent(event)
        if self.fourier_checkbox.isChecked() and self.freq_layout_added:
            for label in [self.original_freq_label, self.processed_freq_label]:
                if label.pixmap() and not label.pixmap().isNull():
                    label.setPixmap(self.scale_frequency_image(label.pixmap()))

    def scale_frequency_image(self, pixmap):
        """Scale frequency domain images to fit the window.

        Args:
            pixmap (QPixmap): The pixmap to scale.

        Returns:
            QPixmap: The scaled pixmap.
        """
        return pixmap.scaled(self.width() // 2 - 20, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def load_image(self):
        """Load an image file and enable processing controls."""
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image files (*.jpg *.png *.bmp)")
        if fname:
            self.original_image = cv2.imread(fname)
            self.processed_image = self.original_image.copy()
            for widget in [self.histogram_slider, self.blur_slider, self.kernel_size_selector,
                           self.fourier_checkbox, self.fourier_inner_slider, self.fourier_outer_slider]:
                widget.setEnabled(True)
            self.apply_processing()
            self.image_window = ImageDisplayWindow(self.original_image, self.processed_image)
            self.image_window.show()

    def apply_processing(self):
        """Apply image processing based on current control values."""
        if self.original_image is None:
            return
        self.processed_image = self.original_image.copy()
        
        # Apply histogram gain
        alpha = 1 + self.histogram_slider.value() / 100.0
        self.processed_image = cv2.convertScaleAbs(self.processed_image, alpha=alpha)

        # Apply Gaussian blur if enabled
        if self.blur_slider.value() > 0:
            kernel_size = self.kernel_size_selector.value() | 1  # Ensure odd kernel size
            self.processed_image = cv2.GaussianBlur(self.processed_image, (kernel_size, kernel_size), self.blur_slider.value() * 0.05)

        # Handle Fourier filter and frequency display
        fourier_layout = self.fourier_group.layout()
        if self.fourier_checkbox.isChecked():
            self.processed_image = apply_fourier_filter(self.processed_image,
                                                        self.fourier_inner_slider.value(),
                                                        self.fourier_outer_slider.value())
            if not self.freq_layout_added:
                fourier_layout.addLayout(self.freq_layout)
                self.freq_layout_added = True
            orig_freq = compute_frequency_image(self.original_image)  # Original spectrum
            proc_freq = compute_frequency_image(self.processed_image, 
                                                inner=self.fourier_inner_slider.value(), 
                                                outer=self.fourier_outer_slider.value())  # Spectrum + mask
            self.original_freq_label.setPixmap(convert_cv_qt(orig_freq))
            self.processed_freq_label.setPixmap(convert_cv_qt(proc_freq))
            self.original_freq_title.setVisible(True)
            self.processed_freq_title.setVisible(True)
            self.original_freq_label.setVisible(True)
            self.processed_freq_label.setVisible(True)
            # Ensure sliders are enabled when filter is active
            self.fourier_inner_slider.setEnabled(True)
            self.fourier_outer_slider.setEnabled(True)
        else:
            if self.freq_layout_added:
                fourier_layout.removeItem(self.freq_layout)
                self.original_freq_title.setVisible(False)
                self.processed_freq_title.setVisible(False)
                self.original_freq_label.setVisible(False)
                self.processed_freq_label.setVisible(False)
                self.freq_layout_added = False
            # Disable and reset frequency sliders when filter is off
            self.fourier_inner_slider.setEnabled(False)
            self.fourier_inner_slider.setValue(0)  # Reset to minimum
            self.fourier_outer_slider.setEnabled(False)
            self.fourier_outer_slider.setValue(0)  # Reset to minimum
            self.fourier_group.update()

        # Update the display window if open
        if self.image_window:
            self.image_window.proc_pixmap = convert_cv_qt(self.processed_image)
            self.image_window.resizeEvent(None)
            self.image_window.proc_hist_label.setPixmap(generate_histogram(self.processed_image))