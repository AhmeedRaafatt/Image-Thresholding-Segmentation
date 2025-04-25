from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np
import logging


class RegionGrowingKMeans:
    def __init__(self):
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_segmentation, Topics.APPLY_SEGMENTATION)
    
    def on_apply_segmentation(self, image, method):
        if method.lower() not in ['region growing', 'k-means']:
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

            
            ###
            pub.sendMessage(
                Topics.SEGMENTATION_RESULT,
                result_image=result_image
                )
        except Exception as e:
            logging.error(f"Error in {method} segmentation: {str(e)}")