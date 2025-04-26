from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np
import logging

# Import Cython modules
try:
    import mean_shift_cy
    import region_growing_cy
    CYTHON_AVAILABLE = True
    logging.info("Cython implementation available for segmentation algorithms")
except ImportError:
    CYTHON_AVAILABLE = False
    logging.error("Cython implementation not available - these algorithms require Cython")


class AgglomerativeMeanShift:
    def __init__(self):
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_segmentation, Topics.APPLY_SEGMENTATION)
    
    def on_apply_segmentation(self, image, method, seed_points, parameters=None):
        if method.lower() not in ['region growing', 'mean-shift']:
            return
            
        # Use default parameters if none provided
        if parameters is None:
            parameters = {}
            
        # Check if Cython is available
        if not CYTHON_AVAILABLE:
            logging.error("Cannot run segmentation - Cython modules not found")
            return
            
        logging.info(f"Applying {method} segmentation with parameters: {parameters}")
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_in_executor(executor, self.apply, image, method, seed_points, parameters)
    
    def apply(self, image, method, seed_points, parameters):
        try:
            if method == "mean-shift":
                # Extract parameters or use defaults
                spatial_radius = parameters.get('spatial_radius', 20)
                color_radius = parameters.get('color_radius', 40)
                max_iterations = parameters.get('max_iterations', 20)
                
                result_image = self.mean_shift_segmentation(
                    image, 
                    spatial_radius=spatial_radius,
                    color_radius=color_radius,
                    max_iterations=max_iterations
                )
                
            elif method == "region growing":
                if not seed_points:
                    seed_points = [(image.shape[0] // 2, image.shape[1] // 2)] 
                    logging.info("No seed points provided, using center of the image as seed point")
                    
                # Extract parameters or use defaults
                threshold = parameters.get('threshold', 10)
                
                # Process all seed points
                seed_points = [(int(y), int(x)) for x, y in seed_points]
                result_image = self.region_growing_segmentation(image, seed_points, threshold=threshold)
                
            pub.sendMessage(
                Topics.SEGMENTATION_RESULT,
                result_image=result_image
                )
                
        except Exception as e:
            logging.error(f"Error in {method} segmentation: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    def mean_shift_segmentation(self, image, spatial_radius=20, color_radius=40, step=3, max_iterations=20, min_shift=0.1):
        try:
            start_time = time.time()
            
            # Convert to LAB color space
            img = image.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Use Cython implementation
            logging.info(f"Using Mean Shift with parameters: spatial_radius={spatial_radius}, color_radius={color_radius}")
            result = mean_shift_cy.mean_shift_segmentation_cy(
                img, 
                spatial_radius=spatial_radius,
                color_radius=color_radius,
                step=step,
                max_iterations=max_iterations,
                min_shift=min_shift
            )
            
            # Convert back to BGR
            result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
            logging.info(f"Mean Shift completed in {time.time() - start_time:.2f} seconds")
            return result
                
        except Exception as e:
            logging.error(f"Error in Mean Shift segmentation: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return image

    def region_growing_segmentation(self, image, seed_points, threshold=100):
        try:
            start_time = time.time()
            
            # Convert image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use Cython implementation
            logging.info(f"Using Region Growing with {len(seed_points)} seed points, threshold={threshold}")
            mask = region_growing_cy.region_growing_segmentation_cy(gray_image, seed_points, threshold)
            
            # Apply mask to the original image
            segmented_image = cv2.bitwise_and(image, image, mask=mask)
            
            logging.info(f"Region Growing completed in {time.time() - start_time:.2f} seconds")
            return segmented_image
            
        except Exception as e:
            logging.error(f"Error in Region Growing segmentation: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return image
