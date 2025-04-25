from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np
import logging


class AgglomerativeMeanShift:
    def __init__(self):
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_segmentation, Topics.APPLY_SEGMENTATION)
    
    def on_apply_segmentation(self, image, method, seed_points):
        if method.lower() not in ['region growing', 'mean-shift']:
            return
            
        logging.info(f"Applying {method} segmentation")
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_in_executor(executor, self.apply, image, method,seed_points)
    
    def apply(self, image, method, seed_points):
        try:
            if method == "mean-shift":
                result_image = self.mean_shift_segmentation(image)
            elif method == "region growing":
                if not seed_points:
                    seed_points = [(image.shape[0] // 2, image.shape[1] // 2)] 
                    logging.info("No seed points provided, using center of the image as seed point")
                # Process all seed points
                result_image = self.region_growing_segmentation(image, seed_points, threshold=10)
            pub.sendMessage(
                Topics.SEGMENTATION_RESULT,
                result_image=result_image
                )
        except Exception as e:
            logging.error(f"Error in {method} segmentation: {str(e)}")
    
    def mean_shift_segmentation(self, image):
        """
        Apply Mean Shift segmentation implemented from scratch
        
        Parameters:
        - image: Input image (BGR format)
        
        Returns:
        - Segmented image
        """
        try:
            # Convert to float32 for calculations and L*a*b color space for better clustering
            img = image.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Parameters for mean shift
            spatial_radius = 10
            color_radius = 15
            spatial_radius_sq = spatial_radius ** 2
            color_radius_sq = color_radius ** 2
            max_iterations = 10  # Maximum shift iterations per pixel
            min_shift = 0.1  # Minimum shift magnitude to continue
            
            # Create output image
            height, width = img.shape[:2]
            result = np.zeros_like(img)
            
            logging.info(f"Applying Mean Shift with spatial radius={spatial_radius}, color radius={color_radius}")
            
            # Sample points (skip pixels for speed - can be adjusted)
            step = 3  # Skip pixels for faster processing
            sample_points = []
            for y in range(0, height, step):
                for x in range(0, width, step):
                    sample_points.append((y, x))
                    
            # Process each point in the image
            clusters = {}  # Store cluster centers
            
            for i, (y, x) in enumerate(sample_points):
                if i % 100 == 0:
                    logging.info(f"Processing point {i}/{len(sample_points)}")
                    
                # Get the original pixel value
                point = img[y, x].copy()
                cur_y, cur_x = y, x
                
                # Mean shift iteration
                for _ in range(max_iterations):
                    # Define search window
                    y_start = max(0, cur_y - spatial_radius)
                    y_end = min(height, cur_y + spatial_radius + 1)
                    x_start = max(0, cur_x - spatial_radius)
                    x_end = min(width, cur_x + spatial_radius + 1)
                    
                    # Initialize weighted mean
                    weight_sum = 0
                    new_point = np.zeros(3, dtype=np.float32)
                    new_pos = np.zeros(2, dtype=np.float32)
                    
                    # For each pixel in window
                    for wy in range(y_start, y_end, step):
                        for wx in range(x_start, x_end, step):
                            # Calculate spatial distance
                            spatial_dist_sq = (wy - cur_y) ** 2 + (wx - cur_x) ** 2
                            
                            # Skip if outside spatial radius
                            if spatial_dist_sq > spatial_radius_sq:
                                continue
                            
                            # Calculate color distance
                            color_dist_sq = np.sum((img[wy, wx] - point) ** 2)
                            
                            # Skip if outside color radius
                            if color_dist_sq > color_radius_sq:
                                continue
                                
                            # Calculate weight using Gaussian kernel
                            weight = np.exp(-0.5 * (spatial_dist_sq / spatial_radius_sq + 
                                                  color_dist_sq / color_radius_sq))
                            
                            # Update weighted mean
                            weight_sum += weight
                            new_point += weight * img[wy, wx]
                            new_pos += weight * np.array([wy, wx])
                    
                    # If no points in window, break
                    if weight_sum == 0:
                        break
                        
                    # Calculate new center
                    new_point /= weight_sum
                    new_pos /= weight_sum
                    
                    # Calculate shift magnitude
                    shift = np.sqrt(np.sum((new_point - point) ** 2))
                    
                    # Update point
                    point = new_point
                    cur_y, cur_x = int(round(new_pos[0])), int(round(new_pos[1]))
                    
                    # Check convergence
                    if shift < min_shift:
                        break
                
                # Store result
                key = tuple(np.round(point, 1))  # Round for better clustering
                if key not in clusters:
                    clusters[key] = point
                
                # Assign to closest cluster
                result[y, x] = point
            
            # Fill in skipped pixels by finding nearest processed pixel
            for y in range(height):
                for x in range(width):
                    if (y % step != 0) or (x % step != 0):
                        # Find closest processed pixel
                        y_base, x_base = (y // step) * step, (x // step) * step
                        y_base = min(y_base, height - step)
                        x_base = min(x_base, width - step)
                        result[y, x] = result[y_base, x_base]
            
            # Convert back to BGR
            result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2BGR)
            return result
                
        except Exception as e:
            logging.error(f"Error in Mean Shift segmentation: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            # Return the original image if there's an error
            return image

    def region_growing_segmentation(self, image, seed_points, threshold=10):
        """
        Apply Region Growing segmentation implemented from scratch
        
        Parameters:
        - image: Input image (BGR format)
        - seed_points: List of starting points (y, x)
        - threshold: Intensity difference threshold
        
        Returns:
        - Segmented image
        """
        try:
            # Convert image to grayscale for simpler processing
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Initialize mask
            mask = np.zeros_like(gray_image, dtype=np.uint8)
            
            # Initialize queue for the region growing algorithm
            queue = []
            
            # Process each seed point
            for seed_point in seed_points:
                y, x = seed_point  # Seed points are stored as (y, x)
                
                # Check if the seed point is within image bounds
                if 0 <= y < gray_image.shape[0] and 0 <= x < gray_image.shape[1]:
                    mask[y, x] = 255
                    queue.append((y, x))
            
            # Process the queue using breadth-first approach
            while queue:
                y, x = queue.pop(0)
                pixel_value = int(gray_image[y, x])
                
                # Check 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue  # Skip the center pixel
                            
                        ny, nx = y + dy, x + dx
                        
                        # Check bounds
                        if 0 <= ny < gray_image.shape[0] and 0 <= nx < gray_image.shape[1]:
                            # Check if not already in region and within threshold
                            if (mask[ny, nx] == 0 and 
                                abs(pixel_value - int(gray_image[ny, nx])) < threshold):
                                mask[ny, nx] = 255
                                queue.append((ny, nx))
            
            # Apply mask to the original image
            segmented_image = cv2.bitwise_and(image, image, mask=mask)
            return segmented_image
            
        except Exception as e:
            logging.error(f"Error in Region Growing segmentation: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            # Return the original image if there's an error
            return image
