from PyQt6.QtWidgets import QMainWindow
from PyQt6 import uic
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from pubsub import pub
import logging
import cv2
import numpy as np

from message_types import Topics

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # "w" overwrites the file; use "a" to append
)

class MainWindowUI(QMainWindow):
    def __init__(self):
        super(MainWindowUI, self).__init__()
        self.ui = uic.loadUi("main.ui", self)
        self.ui.showMaximized()
        self.ui.setWindowTitle("Image Thresholding and Segmentation")
        
        # Initialize image variables
        self.original_image = None
        self.current_tab = "threshold"  # Default tab
        self.seed_points = []
        self.seg_original_img.mousePressEvent = self.on_image_clicked
        # Setup button connections
        self.setup_connections()
        
        # Subscribe to PubSub events
        self.setup_subscriptions()
    
    def setup_connections(self):
        # Tab change connection
        self.tabWidget.currentChanged.connect(self.on_tab_changed)
        
        # Thresholding tab connections
        self.load_image_btn.clicked.connect(self.on_load_image_threshold)
        self.apply_btn.clicked.connect(self.on_apply_threshold)
        self.thresholding_method.currentIndexChanged.connect(self.on_threshold_method_changed)
        self.local_thresholding.toggled.connect(self.on_threshold_type_changed)
        self.global_thresholding.toggled.connect(self.on_threshold_type_changed)
        
        # Segmentation tab connections
        self.load_btn.clicked.connect(self.on_load_image_segmentation)
        self.apply_btn_2.clicked.connect(self.on_apply_segmentation)
        self.seg_method_comboBox.currentIndexChanged.connect(self.on_segmentation_method_changed)
    
    def setup_subscriptions(self):
        # Subscribe to processing results
        pub.subscribe(self.on_threshold_result, Topics.THRESHOLD_RESULT)
        pub.subscribe(self.on_segmentation_result, Topics.SEGMENTATION_RESULT)
    
    def on_tab_changed(self, index):
        tab_name = self.tabWidget.tabText(index).lower()
        self.current_tab = tab_name
        pub.sendMessage(Topics.TAB_CHANGED, tab=tab_name)
        logging.info(f"Tab changed to {tab_name}")
    
    def on_load_image_threshold(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            logging.info(f"Loading image for thresholding: {file_path}")
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                # Display the original image
                self.display_image(self.thresh_original_img, self.original_image)
                # Publish image loaded event
                pub.sendMessage(Topics.IMAGE_LOADED, image=self.original_image, tab="threshold")
            else:
                logging.error(f"Failed to load image: {file_path}")
    
    def on_load_image_segmentation(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            logging.info(f"Loading image for segmentation: {file_path}")
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.seed_points.clear()  # Clear any existing seed points
                # Display the original image
                self.display_image(self.seg_original_img, self.original_image)
                # Publish image loaded event
                pub.sendMessage(Topics.IMAGE_LOADED, image=self.original_image, tab="segmentation")
            else:
                logging.error(f"Failed to load image: {file_path}")
    
    def on_apply_threshold(self):
        if self.original_image is None:
            logging.warning("No image loaded for thresholding")
            return
        
        # Get threshold settings
        method = self.thresholding_method.currentText().lower()
        is_local = self.local_thresholding.isChecked()
        threshold_type = "local" if is_local else "global"
        
        logging.info(f"Applying thresholding with method: {method}, type: {threshold_type}")
        pub.sendMessage(Topics.APPLY_THRESHOLD, 
                        image=self.original_image, 
                        method=method,
                        threshold_type=threshold_type)
    
    def on_apply_segmentation(self):
        if self.original_image is None:
            logging.warning("No image loaded for segmentation")
            return
        
        # Get segmentation method
        method = self.seg_method_comboBox.currentText().lower()
        
        logging.info(f"Applying segmentation with method: {method}")
        pub.sendMessage(Topics.APPLY_SEGMENTATION, 
                        image=self.original_image, 
                        method=method,
                        seed_points=self.seed_points.copy())  # Pass seed points as a named parameter
        
        self.seed_points.clear()  # Clear seed points after applying
        
        # Reset the original image display to remove seed points
        self.display_image(self.seg_original_img, self.original_image)
        
        logging.info("Cleared seed points after applying segmentation")
    
    def on_threshold_method_changed(self, index):
        method = self.thresholding_method.currentText().lower()
        logging.info(f"Threshold method changed to: {method}")
        pub.sendMessage(Topics.THRESHOLD_METHOD_CHANGED, method=method)
    
    def on_threshold_type_changed(self):
        threshold_type = "local" if self.local_thresholding.isChecked() else "global"
        logging.info(f"Threshold type changed to: {threshold_type}")
        pub.sendMessage(Topics.THRESHOLD_TYPE_CHANGED, threshold_type=threshold_type)
    
    def on_segmentation_method_changed(self, index):
        method = self.seg_method_comboBox.currentText().lower()
        self.seed_points.clear()  # Clear seed points when method changes
        logging.info(f"Segmentation method changed to: {method}, seed points cleared")
        # Reset the display to original image
        if self.original_image is not None:
            self.display_image(self.seg_original_img, self.original_image)
        pub.sendMessage(Topics.SEGMENTATION_METHOD_CHANGED, method=method)
    
    def on_threshold_result(self, result_image):
        self.display_image(self.thresh_output_img, result_image)
    
    def on_segmentation_result(self, result_image):
        self.display_image(self.seg_output_img, result_image)
    
    def display_image(self, label, image):
        """Display an image on a QLabel"""
        if image is None:
            return
        # Clear any existing image first
        label.clear()

        # Convert OpenCV image (BGR) to RGB for Qt
        if len(image.shape) == 3:  # Color image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:  # Grayscale image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        height, width = rgb_image.shape[:2]
        
        # Create QImage from numpy array
        q_img = QImage(rgb_image.data, width, height, width * 3, QImage.Format.Format_RGB888)
        
        # Create and set pixmap
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale pixmap to fit the label while maintaining aspect ratio
        pixmap = pixmap.scaled(label.width(), label.height(), 
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        
        # Set pixmap to label
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def update_image_with_seeds(self):
        """Update the displayed image with seed points"""
        if self.original_image is None or not self.seed_points:
            return
            
        logging.info(f"Updating image with {len(self.seed_points)} seed points")
        
        # Create a fresh copy of the original image each time
        display_image = self.original_image.copy()
        
        # Make sure we're drawing with the right color format (BGR for OpenCV)
        for i, (x, y) in enumerate(self.seed_points):
            # Draw a larger, more visible circle for each seed point
            cv2.circle(display_image, (x, y), 10, (0, 255, 0), -1)  # Green filled circle
            
            # Add number label with better visibility
            cv2.putText(display_image, str(i+1), (x+15, y+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Force redraw of the image with seed points
        self.display_image(self.seg_original_img, display_image)

    def on_image_clicked(self, event):
        """Handle mouse clicks on the image label"""
        if (self.current_tab == "segmentation" and 
            self.seg_method_comboBox.currentText().lower() == "region growing"):
            
            logging.info("Processing click for seed point")
            
            # Get click position relative to the label
            pos = event.pos()
            label = self.seg_original_img
            pixmap = label.pixmap()
            
            if pixmap and self.original_image is not None:
                # Convert click coordinates to image coordinates
                label_size = label.size()
                scaled_size = pixmap.size()
                
                # Calculate margins
                x_margin = (label_size.width() - scaled_size.width()) // 2
                y_margin = (label_size.height() - scaled_size.height()) // 2
                
                # Adjust click position by margins
                img_x = pos.x() - x_margin
                img_y = pos.y() - y_margin
                
                # Convert to original image coordinates
                orig_height, orig_width = self.original_image.shape[:2]
                scale_x = orig_width / scaled_size.width()
                scale_y = orig_height / scaled_size.height()
                
                orig_x = int(img_x * scale_x)
                orig_y = int(img_y * scale_y)
                
                # Add point if it's within image bounds
                if 0 <= orig_x < orig_width and 0 <= orig_y < orig_height:
                    self.seed_points.append((orig_x, orig_y))
                    logging.info(f"Added seed point at ({orig_x}, {orig_y})")
                    # Force update with the new seed point
                    self.update_image_with_seeds()
                else:
                    logging.warning(f"Click outside image bounds: ({orig_x}, {orig_y})")