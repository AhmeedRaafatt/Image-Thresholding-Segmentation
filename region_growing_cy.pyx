import numpy as np
cimport numpy as np
cimport cython
from collections import deque

@cython.boundscheck(False)
@cython.wraparound(False)
def region_growing_segmentation_cy(np.ndarray[np.uint8_t, ndim=2] gray_image, list seed_points, int threshold=10):
    """
    Region Growing segmentation implemented in Cython
    """
    cdef int height = gray_image.shape[0]
    cdef int width = gray_image.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=2] mask = np.zeros_like(gray_image, dtype=np.uint8)
    
    # Define variables
    cdef int y, x, ny, nx, dy, dx, pixel_value
    cdef queue = deque()
    
    # Process each seed point
    for seed_point in seed_points:
        y, x = seed_point
        
        # Check if within bounds
        if 0 <= y < height and 0 <= x < width:
            mask[y, x] = 255
            queue.append((y, x))
    
    # Process queue (BFS approach)
    cdef int directions[8][2]
    directions = [[-1, -1], [-1, 0], [-1, 1], 
                  [0, -1],           [0, 1],
                  [1, -1],  [1, 0],  [1, 1]]
                  
    while queue:
        y, x = queue.popleft()
        pixel_value = gray_image[y, x]
        
        # Check all 8 neighbors
        for i in range(8):
            dy = directions[i][0]
            dx = directions[i][1]
            ny = y + dy
            nx = x + dx
            
            # Check bounds
            if 0 <= ny < height and 0 <= nx < width:
                # Check if not in region and within threshold
                if mask[ny, nx] == 0 and abs(int(pixel_value) - int(gray_image[ny, nx])) < threshold:
                    mask[ny, nx] = 255
                    queue.append((ny, nx))
    
    return mask