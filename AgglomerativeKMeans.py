from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np
import logging
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering


class AgglomerativeKMeans:
    def __init__(self):
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_segmentation, Topics.APPLY_SEGMENTATION)
    
    def on_apply_segmentation(self, image, method,seed_points):
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
                result_image = self.agglomerative_segmentation(image, n_clusters=3)


            print("image is returned")
            
            
            ###
            pub.sendMessage(
                Topics.SEGMENTATION_RESULT,
                result_image=result_image
                )
            
            return result_image
        
        except Exception as e:
            logging.error(f"Error in {method} segmentation: {str(e)}")

    def euclidean_distance(a, b):
        diff = a - b
        squared = diff ** 2
        sum_squared = np.sum(squared)
        distance = np.sqrt(sum_squared)
        return distance        

    def kmeans_segmentation(self, image, k=3, max_iters=100, tol=1e-4):
        print('We entered KMeans Segmentation')

        # flattening the image to a 2D array of pixels
        pixel_values = image.reshape((-1, 3)).astype(np.float32) #h*w , 3 (each row is a pixel with 3 channels rgb)
        n_pixels = pixel_values.shape[0]

        #initializing cluster centers
        rng = np.random.default_rng()
        indices = rng.choice(n_pixels, size=k, replace=False) # picks k unique pixel indices from the total number of pixels
        centers = pixel_values[indices]

        for iteration in range(max_iters):
            # compute distances to each center
            distances = np.linalg.norm(pixel_values[:, np.newaxis] - centers, axis=2)  # shape (n_pixels, k) 

            ##########################################################        # This is the same as the above line but using a for loop instead of broadcasting
            # geet number of pixels and centers

            # n_pixels = pixel_values.shape[0]
            # k = centers.shape[0]

            # #initialize distance matrix
            # distances = np.zeros((n_pixels, k))

            # # compute distance from each pixel to each center
            # for i in range(n_pixels):
            #     for j in range(k):
            #         distances[i, j] = self.euclidean_distance(pixel_values[i], centers[j])# shape of (n_pixels, k)

            ############################################################     

            # assign each pixel to the closest center
            labels = np.argmin(distances, axis=1) # shape of (n_pixels,) ex.: [0, 1, 2, 0, 1, 2, ...] where each number is the index of the closest center


            # Compute new centers as mean of assigned pixels
            new_centers = []

            for i in range(k):
                cluster_pixels = pixel_values[labels == i]

                if len(cluster_pixels) > 0:
                    mean_color = cluster_pixels.mean(axis=0) # accross each row
                else:
                    mean_color = centers[i]

                new_centers.append(mean_color)

            new_centers = np.array(new_centers)


            # Check for convergence
            center_shift = np.linalg.norm(new_centers - centers) # eculidean distance between new centers and old centers
            if center_shift < tol: #  if the distance is less than the tolerance
                break
            centers = new_centers

        # Convert centers to uint8
        centers = np.uint8(centers)
        segmented_image = centers[labels] # map each pixel in the original image to the corresponding color (or intensity) from the centers array,
        segmented_image = segmented_image.reshape(image.shape)

        print('Segmenting KMeans Done')
        return segmented_image

        
    def agglomerative_segmentation(self, image, n_clusters=30):
        print('We entered Agglomerative Segmentation')
        start_time = time.time()
        
        # downsampling the image aggressively
        max_size = 20
        h, w = image.shape[:2]
        scale = min(max_size/h, max_size/w)
        new_h, new_w = int(h * scale), int(w * scale) 
        
        resized = cv2.resize(image, (new_w, new_h))
        print(f"Downsampled from {image.shape[:2]} to {resized.shape[:2]}")
        
        # Prepare pixel values
        pixel_values = resized.reshape((-1, 3)).astype(np.float32) # 2d array of pixels, where each row is a pixel with 3 channels (RGB)
        n_samples = pixel_values.shape[0]
        
        print(f"Working with {n_samples} pixels")
        
        # Compute initial distance matrix using scipy's efficient pdist
        print("Computing distance matrix...")
        distances = squareform(pdist(pixel_values, 'euclidean')) # pdist computes pairwise distances between all points in the array, and squareform converts the condensed distance matrix to a square form
        # The entry at position (i, j) represents the Euclidean distance between the i-th and j-th pixels.
        
        # initialize clusters - each pixel starts as its own cluster
        clusters = list(range(n_samples))
        # Mapping from pixel index to cluster label
        cluster_map = {i: [i] for i in range(n_samples)} # {0: [0], 1: [1], 2: [2], ...} initially each pixel is its own cluster
        active_clusters = set(range(n_samples))
        
        print(f"Starting with {len(active_clusters)} clusters")
        
        # Keep merging until we reach desired number of clusters
        merge_count = 0 #  Each time two clusters are merged, this count is incremented.
        while len(active_clusters) > n_clusters: # loop will stop once the number of active clusters is equal to n_clusters.
            merge_count += 1
            if merge_count % 100 == 0:
                print(f"Merging progress: {n_samples - len(active_clusters)}/{n_samples - n_clusters} merges")
            
            # Find the closest pair of clusters
            min_dist = float('inf')
            merge_pair = None
            
            # Convert to list for faster iteration
            active_list = list(active_clusters)
            for i in range(len(active_list)):
                c1 = active_list[i]
                for j in range(i+1, len(active_list)):
                    c2 = active_list[j]
                    if distances[c1, c2] < min_dist:
                        min_dist = distances[c1, c2]
                        merge_pair = (c1, c2)
            
            if merge_pair is None:
                break
                
            c1, c2 = merge_pair
            
            # Update cluster map
            cluster_map[c1].extend(cluster_map[c2])#The pixels in c2 are added to c1 by extending
            del cluster_map[c2]
            
            # Remove c2 from active clusters
            active_clusters.remove(c2)
            
            # Update distances using average linkage
            #For each remaining cluster c, if it is not c1, we calculate the average distance between c1 and c
            for c in active_clusters:
                if c != c1:
                    # Calculate average distance between clusters
                    c1_pixels = cluster_map[c1]
                    c_pixels = cluster_map[c]
                    
                    total_dist = 0
                    for i in c1_pixels:
                        for j in c_pixels:
                            total_dist += distances[i, j]
                    
                    avg_dist = total_dist / (len(c1_pixels) * len(c_pixels))#  total sum of distances divided by the number of pixel pairs.
                    
                    # Update distance in matrix
                    distances[c1, c] = distances[c, c1] = avg_dist
        
        print(f"Finished with {len(active_clusters)} clusters after {merge_count} merges")
        
        # final label array
        labels = np.zeros(n_samples, dtype=int) #store the cluster label for each pixel.
        for i, cluster_idx in enumerate(active_clusters): # For each cluster (cluster_idx) in the active_clusters set, assign a unique label (from i, starting from 0).
            for pixel_idx in cluster_map[cluster_idx]:
                labels[pixel_idx] = i
        
        # Assign random colors to clusters
        unique_colors = np.random.randint(0, 255, size=(n_clusters, 3), dtype=np.uint8)
        
        # Create the segmented image
        segmented_small = unique_colors[labels].reshape(resized.shape)
        
        # Resize back to original size
        segmented_image = cv2.resize(segmented_small, (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        print(f'Agglomerative Segmentation Done in {time.time() - start_time:.2f} seconds')
        return segmented_image
    

    def builtin_agglomerative_segmentation(self, image, n_clusters=3):
        print('We entered builtin Agglomerative Segmentation')
        start_time = time.time()

        # Downsample the image aggressively
        max_size = 30  # Even smaller size - each pixel becomes a feature
        h, w = image.shape[:2]
        scale = min(max_size / h, max_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        print(f"Downsampled from {image.shape[:2]} to {resized.shape[:2]}")
        
        # Prepare pixel values
        pixel_values = resized.reshape((-1, 3)).astype(np.float32)
        print(f"Working with {pixel_values.shape[0]} pixels")

        # Using scikit-learn's AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        
        # Fit the model to pixel values and predict labels
        print("Running clustering algorithm...")
        labels = clustering.fit_predict(pixel_values)
        
        print(f"Clustering completed in {time.time() - start_time:.2f} seconds")
        
        # Generate random colors for each cluster
        colors = np.random.randint(0, 255, size=(n_clusters, 3), dtype=np.uint8)
        
        # Map labels to colors
        segmented_flat = colors[labels]
        segmented_small = segmented_flat.reshape(resized.shape)
        
        # Resize back to original size
        segmented_image = cv2.resize(segmented_small, (image.shape[1], image.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        print(f'Agglomerative Segmentation Completed in {time.time() - start_time:.2f} seconds')
        return segmented_image