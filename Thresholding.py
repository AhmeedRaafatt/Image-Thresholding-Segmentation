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
            grayImage = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        else:
            grayImage = image.copy()
        
        grayImage = cv2.normalize(grayImage, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        def compute_otsu_variance(image, threshold):
            thresholded_image = np.zeros(image.shape)
            thresholded_image[image >= threshold] = 1

            weight1 = len(thresholded_image[thresholded_image == 1]) / len(thresholded_image)
            weight0 = 1 - weight1

            if weight1 == 0 or weight0 == 0:
                return np.inf

            val_pixels_of_1 = image[thresholded_image == 1]
            val_pixels_of_0 = image[thresholded_image == 0]

            if len(val_pixels_of_1) > 0:
                var1 = np.var(val_pixels_of_1) 
            else: 
                var1 = 0

            if len(val_pixels_of_0) > 0:
                var0 = np.var(val_pixels_of_0) 
            else:
                var0 = 0

            badness = weight0 * var0 + weight1 * var1
            return badness



        if threshold_type == "global":
            threshold_range = range(np.max(grayImage) + 1)                
            criteria = np.array([compute_otsu_variance(grayImage, threshold) for threshold in threshold_range])
            best_threshold = threshold_range[np.argmin(criteria)]

            result = grayImage.copy()
            result[result < best_threshold] = 0
            result[result >= best_threshold] = 255


        elif threshold_type == "local":
            block_size = 40
            sigma = 1
            blurredImage = cv2.GaussianBlur(grayImage, (0, 0), sigmaX=sigma, sigmaY=sigma)
            height, width = grayImage.shape
            result = np.zeros_like(grayImage)

            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    block = blurredImage[y:min(y + block_size, height), 
                                  x:min(x + block_size, width)]
                    
                    threshold_range = range(np.min(block), np.max(block) + 1)
                    criteria = np.array([compute_otsu_variance(block, th) for th in threshold_range])
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
            grayImage = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        else:
            grayImage = image.copy()

        if threshold_type == "global":
            min_intensity = np.min(grayImage)
            max_intensity = np.max(grayImage)
            # Initial Threshold
            threshold = (min_intensity + max_intensity) / 2

            while True:
                foreground_pixels = grayImage[grayImage > threshold]
                background_pixels = grayImage[grayImage <= threshold]

                mean_foreground = np.mean(foreground_pixels)
                mean_background = np.mean(background_pixels)

                new_threshold = (mean_foreground + mean_background) / 2

                if np.abs(new_threshold - threshold) < 1e-3:
                    break

                threshold = new_threshold

            result = (grayImage > threshold).astype(np.uint8) * 255

        elif threshold_type == "local":
            window_size = 40
            height, width = grayImage.shape
            result = np.zeros_like(grayImage, dtype=np.uint8)

            for y in range(0, height, window_size):
                for x in range(0, width, window_size):
                    window = grayImage[y:y + window_size, x:x + window_size]
                    
                    if np.all(window == window[0, 0]):
                        continue

                    threshold = np.mean(window)

                    while True:
                        foreground_pixels = window[window > threshold]
                        background_pixels = window[window <= threshold]

                        if len(foreground_pixels) == 0 or len(background_pixels) == 0:
                            break

                        mean_foreground = np.mean(foreground_pixels)
                        mean_background = np.mean(background_pixels)

                        new_threshold = (mean_foreground + mean_background) / 2

                        # Break the loop if the new threshold is too close to the old one (Reach Convergence)
                        if np.abs(new_threshold - threshold) < 1e-3:
                            break

                        threshold = new_threshold

                    window_height, window_width = window.shape
                    # update the result image with the binary thresholded window
                    result[y:y + window_height, x: x + window_width] = (window > threshold).astype(np.uint8) * 255

        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

        print(result.shape)
        return result
    

    
    def spectral_threshold(self, image, threshold_type):
        if len(image.shape) > 2:
            grayImage = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        else:
            grayImage = image.copy()
        
        grayImage = (grayImage * 255).astype(np.uint8)

        if threshold_type == "global":
            thresholds = self.get_thresholds(grayImage)
            thresholded_img = self.spectral_double_thresholding(grayImage, thresholds)
            return thresholded_img
        
        elif threshold_type == "local":
            local_threshold = np.zeros_like(grayImage)
            window_size = 40
            for i in range(0, grayImage.shape[0], window_size):
                for j in range(0, grayImage.shape[1], window_size):
                    window_height = min(window_size, grayImage.shape[0] - i)
                    window_width = min(window_size, grayImage.shape[1] - j)
                    sub_image = grayImage[i:i+window_height, j:j+window_width]

                    thresholds = self.get_thresholds(sub_image)

                    thresholded_window = self.spectral_double_thresholding(sub_image, thresholds)
                    local_threshold[i:i+window_height, j:j+window_width] = thresholded_window

            return local_threshold

        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

    
    def get_thresholds(self, image):
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        cumulative_sum = np.cumsum(histogram)
        total = image.shape[0] * image.shape[1]
        tau1 = 0
        tau2 = 0
        alpha = 0.5

        background_pixels = 0
        foreground_pixels = 0
        background_pixels_sum = 0
        foreground_pixels_sum = 0
        max_variance = 0
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


    def spectral_double_thresholding(self, image, thresholds):
        thresholded_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] <= thresholds[1]:
                    thresholded_image[i, j] = 0
                elif thresholds[1] < image[i, j] <= thresholds[0]:
                    thresholded_image[i, j] = 128
                else:
                    thresholded_image[i, j] = 255
        
        return thresholded_image
