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
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            print(f"Applying thresholding with method: {method}")
            result = image.copy()

            if method == "otsu":
                result = self.otsu_threshold(image, threshold_type)
            elif method == "optimal":
                result = self.optimal_threshold(image, threshold_type)
            elif method == "spectral":
                result = self.spectral_threshold(image, threshold_type)
            elif method == "local":
                result = self.local_threshold(image, threshold_type)
            else:
                raise ValueError(f"Unknown thresholding method: {method}")
                

            result_image = result

            ###
            pub.sendMessage(
                Topics.THRESHOLD_RESULT,
                result_image=result_image
                )
        except Exception as e:
            logging.error(f"Error with method {method}: {str(e)}")


    def otsu_threshold(self, image, threshold_type):
        if len(image.shape) > 2:
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = image.copy()
        
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        def compute_otsu_criteria(img, th):
            thresholded_im = np.zeros(img.shape)
            thresholded_im[img >= th] = 1

            nb_pixels = img.size
            nb_pixels1 = np.count_nonzero(thresholded_im)
            weight1 = nb_pixels1 / nb_pixels
            weight0 = 1 - weight1

            if weight1 == 0 or weight0 == 0:
                return np.inf

            val_pixels1 = img[thresholded_im == 1]
            val_pixels0 = img[thresholded_im == 0]

            var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
            var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

            return weight0 * var0 + weight1 * var1

        if threshold_type == "global":
            threshold_range = range(np.max(gray) + 1)
            criteria = np.array([compute_otsu_criteria(gray, th) for th in threshold_range])
            best_threshold = threshold_range[np.argmin(criteria)]

            result = gray.copy()
            result[result < best_threshold] = 0
            result[result >= best_threshold] = 255

        elif threshold_type == "local":
            block_size = 40
            sigma = 1
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
            height, width = gray.shape
            result = np.zeros_like(gray)

            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    block = blurred[y:min(y + block_size, height), 
                                  x:min(x + block_size, width)]
                    threshold_range = range(np.min(block), np.max(block) + 1)
                    criteria = np.array([compute_otsu_criteria(block, th) for th in threshold_range])
                    best_threshold = threshold_range[np.argmin(criteria)]

                    binary_block = np.zeros_like(block)
                    binary_block[block >= best_threshold] = 255

                    end_y = min(y + block_size, height)
                    end_x = min(x + block_size, width)
                    result[y:end_y, x:end_x] = binary_block

        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

        print(result.shape)
        return result




    def optimal_threshold(self, image, threshold_type):
        if len(image.shape) > 2:
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = image.copy()

        if threshold_type == "global":
            # Initialize threshold with midpoint of intensity range
            min_intensity = np.min(gray)
            max_intensity = np.max(gray)
            threshold = (min_intensity + max_intensity) / 2

            # Iterate until convergence
            while True:
                # Classify pixels into foreground and background
                foreground_pixels = gray[gray > threshold]
                background_pixels = gray[gray <= threshold]

                # Calculate mean intensity values
                mean_foreground = np.mean(foreground_pixels)
                mean_background = np.mean(background_pixels)

                # Calculate new threshold
                new_threshold = (mean_foreground + mean_background) / 2

                # Check convergence
                if np.abs(new_threshold - threshold) < 1e-3:
                    break

                threshold = new_threshold

            result = (gray > threshold).astype(np.uint8) * 255

        elif threshold_type == "local":
            block_size = 40
            height, width = gray.shape
            result = np.zeros_like(gray, dtype=np.uint8)

            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    # Get current block
                    block = gray[y:y + block_size, x:x + block_size]
                    
                    # Skip empty blocks
                    if np.all(block == block[0, 0]):
                        continue

                    # Initialize threshold for block
                    threshold = np.mean(block)

                    # Iterate until convergence
                    while True:
                        foreground_pixels = block[block > threshold]
                        background_pixels = block[block <= threshold]

                        if len(foreground_pixels) == 0 or len(background_pixels) == 0:
                            break

                        mean_foreground = np.mean(foreground_pixels)
                        mean_background = np.mean(background_pixels)
                        new_threshold = (mean_foreground + mean_background) / 2

                        if np.abs(new_threshold - threshold) < 1e-3:
                            break

                        threshold = new_threshold

                    # Apply threshold to block
                    block_height, block_width = block.shape
                    result[y:y + block_height, x:x + block_width] = (block > threshold).astype(np.uint8) * 255

        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

        print(result.shape)
        return result
    

    
    def spectral_threshold(self, image, threshold_type):
        if len(image.shape) > 2:
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = image.copy()
        
        gray = (gray * 255).astype(np.uint8)

        if threshold_type == "global":
            return self.spectral_global(gray)[0]
        elif threshold_type == "local":
            return self.spectral_local(gray)[0]
        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

    
    def get_thresholds(self, image):
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        cumulative_sum = np.cumsum(histogram)
        total = image.shape[0] * image.shape[1]
        tau1 = 0.0
        tau2 = 0.0
        alpha = 0.5

        background_pixels = 0
        foreground_pixels = 0
        background_pixels_sum = 0.0
        foreground_pixels_sum = 0.0
        max_variance = 0.0
        threshold_1 = 0
        threshold_2 = 0

        for i in range(256):
            background_pixels += histogram[i]
            if background_pixels == 0:
                continue
            foreground_pixels = total - background_pixels
            if foreground_pixels == 0:
                break

            background_pixels_sum += i * histogram[i]
            foreground_pixels_sum = cumulative_sum[-1] - background_pixels_sum

            background_mean = background_pixels_sum / background_pixels
            foreground_mean = foreground_pixels_sum / foreground_pixels

            variance_difference = background_pixels * foreground_pixels * (background_mean - foreground_mean) ** 2

            if variance_difference > max_variance:
                max_variance = variance_difference
                threshold_1 = i

            tau1 += histogram[i] * i
            if tau1 > alpha * total and tau2 == 0:
                tau2 = i

        threshold_2 = round((threshold_1 + tau2) / 2.0)
        threshold_1 -= 50
        threshold_2 -= 50

        return threshold_1, threshold_2


    def spectral_global(self, img):
        thresholds = self.get_thresholds(img)
        thresholded_img = self.double_spectral_thresholding(img, thresholds)
        return [thresholded_img]

    def spectral_local(self, image, block_size=70):
        local_threshold = np.zeros_like(image)
        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                block_height = min(block_size, image.shape[0] - i)
                block_width = min(block_size, image.shape[1] - j)
                sub_image = image[i:i+block_height, j:j+block_width]

                thresholds = self.get_thresholds(sub_image)

                local_threshold[i:i+block_height, j:j+block_width] = self.double_spectral_thresholding(sub_image, thresholds)

        return [local_threshold]

    def double_spectral_thresholding(self, img, thresholds):
        thresholded_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] <= thresholds[1]:
                    thresholded_img[i, j] = 0
                elif thresholds[1] < img[i, j] <= thresholds[0]:
                    thresholded_img[i, j] = 128
                else:
                    thresholded_img[i, j] = 255
        return thresholded_img
