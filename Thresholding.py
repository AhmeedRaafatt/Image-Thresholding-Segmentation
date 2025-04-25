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
            result_image = image

            
            ###
            pub.sendMessage(
                Topics.THRESHOLD_RESULT,
                result_image=result_image
                )
        except Exception as e:
            logging.error(f"Error in {threshold_type} thresholding with method {method}: {str(e)}")