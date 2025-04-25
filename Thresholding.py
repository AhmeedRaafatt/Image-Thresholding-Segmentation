from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np
import logging


class Thresholding:
    def __init__(self):
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_threshold, Topics.APPLY_THRESHOLD)
    
    def on_apply_threshold(self, image, method, threshold_type):
        logging.info(f"Applying {threshold_type} thresholding with method: {method}")
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_in_executor(executor, self.apply, image, method, threshold_type)
    
    def apply(self, image, method, threshold_type):
        try:
            ###
            result = image.copy()
            if threshold_type == "Otsu":
                result = self.otsu_threshold(image)
            elif threshold_type == "Optimal":
                result = self.optimal_threshold(image)
            elif threshold_type == "Spectral":
                result = self.spectral_threshold(image)
            elif threshold_type == "Local":
                result = self.local_threshold(image)
            else:
                raise ValueError(f"Unknown thresholding method: {method}")
                

            result_image = result

            ###
            pub.sendMessage(
                Topics.THRESHOLD_RESULT,
                result_image=result_image
                )
        except Exception as e:
            logging.error(f"Error in {threshold_type} thresholding with method {method}: {str(e)}")

    def otsu_threshold(self, image):
        """Apply Otsu's thresholding method"""
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def optimal_threshold(self, image):
        """Apply optimal thresholding using iterative method"""
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Initial threshold guess (mean of image)
        threshold = np.mean(image)
        max_iter = 100
        eps = 0.01
        
        for _ in range(max_iter):
            old_threshold = threshold
            # Split image into two groups
            foreground = image[image >= threshold]
            background = image[image < threshold]
            
            # Calculate mean of each group
            mean_fg = np.mean(foreground) if len(foreground) > 0 else 0
            mean_bg = np.mean(background) if len(background) > 0 else 0
            
            # Update threshold
            threshold = (mean_fg + mean_bg) / 2
            
            if abs(threshold - old_threshold) < eps:
                break
                
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def spectral_threshold(self, image):
        """Apply spectral thresholding using k-means clustering"""
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Reshape image for k-means
        pixels = image.reshape((-1, 1))
        pixels = np.float32(pixels)
        
        # Define criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        binary = segmented.reshape(image.shape)
        return binary

    def local_threshold(self, image):
        """Apply adaptive local thresholding"""
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 
            11,  # Block size
            2    # C constant
        )
        return binary