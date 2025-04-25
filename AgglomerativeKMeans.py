from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np
import logging


class AgglomerativeKMeans:
    def __init__(self):
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_segmentation, Topics.APPLY_SEGMENTATION)
    
    def on_apply_segmentation(self, image, method):
        if method.lower() not in ['agglomerative', 'k-means']:
            return
            
        logging.info(f"Applying {method} segmentation")
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_in_executor(executor, self.apply, image, method)
    
    def apply(self, image, method):
        try:
            ###
            result_image = image
            if method == "k-means":
                result_image = self.kmeans_segmentation(image, k=3)
            elif method == "agglomerative":
                pass


            print("image is returned")
            
            
            ###
            pub.sendMessage(
                Topics.SEGMENTATION_RESULT,
                result_image=result_image
                )
            
            return result_image
        except Exception as e:
            logging.error(f"Error in {method} segmentation: {str(e)}")

    def kmeans_segmentation(self, image, k=3):
        # Reshape the image to a 2D array of pixels
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back to uint8 and reshape to original image
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)
        print('Segmenting Kmeans Done')


        return segmented_image