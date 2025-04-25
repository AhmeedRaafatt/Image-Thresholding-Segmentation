import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, exp

@cython.boundscheck(False)
@cython.wraparound(False)
def mean_shift_segmentation_cy(np.ndarray[np.float32_t, ndim=3] img_lab, 
                              int spatial_radius=10, 
                              int color_radius=40, 
                              int step=3,
                              int max_iterations=10,
                              float min_shift=0.1):
    """
    Mean Shift segmentation implemented in Cython
    
    Parameters
    ----------
    img_lab : ndarray
        Input image in LAB color space
    spatial_radius : int
        Spatial radius for the mean shift window
    color_radius : int
        Color radius for the mean shift window
    step : int
        Sampling step size
    max_iterations : int
        Maximum number of iterations for convergence
    min_shift : float
        Minimum shift threshold for convergence
    """
    cdef int height = img_lab.shape[0]
    cdef int width = img_lab.shape[1]
    cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros_like(img_lab)
    
    # Parameters for mean shift
    cdef float spatial_radius_sq = spatial_radius * spatial_radius
    cdef float color_radius_sq = color_radius * color_radius
    
    # Sample grid
    cdef list sample_points = []
    cdef int y, x, i, j, wy, wx
    cdef int y_start, y_end, x_start, x_end
    cdef float weight_sum, weight, shift
    cdef float spatial_dist_sq, color_dist_sq
    
    # Generate sample points
    for y in range(0, height, step):
        for x in range(0, width, step):
            sample_points.append((y, x))
    
    # Process each point
    cdef np.ndarray[np.float32_t, ndim=1] point = np.zeros(3, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] new_point = np.zeros(3, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] new_pos = np.zeros(2, dtype=np.float32)
    cdef int cur_y, cur_x
    cdef dict clusters = {}
    
    for i, (y, x) in enumerate(sample_points):
        # Get original pixel
        point[0] = img_lab[y, x, 0]
        point[1] = img_lab[y, x, 1]
        point[2] = img_lab[y, x, 2]
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
            new_point[0] = 0
            new_point[1] = 0
            new_point[2] = 0
            new_pos[0] = 0
            new_pos[1] = 0
            
            # For each pixel in window
            for wy in range(y_start, y_end, step):
                for wx in range(x_start, x_end, step):
                    # Calculate spatial distance
                    spatial_dist_sq = (wy - cur_y) * (wy - cur_y) + (wx - cur_x) * (wx - cur_x)
                    
                    # Skip if outside spatial radius
                    if spatial_dist_sq > spatial_radius_sq:
                        continue
                    
                    # Calculate color distance
                    color_dist_sq = (img_lab[wy, wx, 0] - point[0]) * (img_lab[wy, wx, 0] - point[0]) + \
                                   (img_lab[wy, wx, 1] - point[1]) * (img_lab[wy, wx, 1] - point[1]) + \
                                   (img_lab[wy, wx, 2] - point[2]) * (img_lab[wy, wx, 2] - point[2])
                    
                    # Skip if outside color radius
                    if color_dist_sq > color_radius_sq:
                        continue
                    
                    # Calculate weight using Gaussian kernel
                    weight = exp(-0.5 * (spatial_dist_sq / spatial_radius_sq + 
                                        color_dist_sq / color_radius_sq))
                    
                    # Update weighted mean
                    weight_sum += weight
                    new_point[0] += weight * img_lab[wy, wx, 0]
                    new_point[1] += weight * img_lab[wy, wx, 1]
                    new_point[2] += weight * img_lab[wy, wx, 2]
                    new_pos[0] += weight * wy
                    new_pos[1] += weight * wx
            
            # If no points in window, break
            if weight_sum == 0:
                break
            
            # Calculate new center
            new_point[0] /= weight_sum
            new_point[1] /= weight_sum
            new_point[2] /= weight_sum
            new_pos[0] /= weight_sum
            new_pos[1] /= weight_sum
            
            # Calculate shift magnitude
            shift = sqrt((new_point[0] - point[0]) * (new_point[0] - point[0]) + 
                        (new_point[1] - point[1]) * (new_point[1] - point[1]) +
                        (new_point[2] - point[2]) * (new_point[2] - point[2]))
            
            # Update point
            point[0] = new_point[0]
            point[1] = new_point[1]
            point[2] = new_point[2]
            cur_y = int(round(new_pos[0]))
            cur_x = int(round(new_pos[1]))
            
            # Check convergence
            if shift < min_shift:
                break
        
        # Store result
        result[y, x, 0] = point[0]
        result[y, x, 1] = point[1]
        result[y, x, 2] = point[2]
    
    # Fill in skipped pixels by finding nearest processed pixel
    for y in range(height):
        for x in range(width):
            if (y % step != 0) or (x % step != 0):
                y_base = (y // step) * step
                x_base = (x // step) * step
                y_base = min(y_base, height - step)
                x_base = min(x_base, width - step)
                result[y, x, 0] = result[y_base, x_base, 0]
                result[y, x, 1] = result[y_base, x_base, 1]
                result[y, x, 2] = result[y_base, x_base, 2]
    
    return result