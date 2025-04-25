class Topics:
    """
    Defines all PubSub topics used for communication within the application.
    
    This class serves as a central registry of all the event topics that components
    can publish to or subscribe to, providing a structured communication framework.
    """
    
    # UI Events
    TAB_CHANGED = "tab_changed"
    """
    Published when the user switches between application tabs.
    
    Args:
        tab (str): The name of the selected tab in lowercase ("threshold" or "segmentation").
    """
    
    IMAGE_LOADED = "image_loaded"
    """
    Published when an image is successfully loaded into the application.
    
    Args:
        image (numpy.ndarray): The loaded image as an OpenCV matrix.
        tab (str): The tab from which the image was loaded ("threshold" or "segmentation").
    """
    
    # Thresholding related events
    APPLY_THRESHOLD = "apply_threshold"
    """
    Published when the user requests to apply a thresholding algorithm to an image.
    
    Args:
        image (numpy.ndarray): The image to be thresholded.
        method (str): The selected thresholding method in lowercase ("optimal", "otsu", or "spectral").
        threshold_type (str): The type of thresholding to apply ("global" or "local").
    """
    
    THRESHOLD_METHOD_CHANGED = "threshold_method_changed"
    """
    Published when the user selects a different thresholding method from the dropdown menu.
    
    Args:
        method (str): The newly selected thresholding method in lowercase ("optimal", "otsu", or "spectral").
    """
    
    THRESHOLD_TYPE_CHANGED = "threshold_type_changed"
    """
    Published when the user switches between global and local thresholding options.
    
    Args:
        threshold_type (str): The selected thresholding type ("global" or "local").
    """
    
    THRESHOLD_RESULT = "threshold_result"
    """
    Published when thresholding is complete, containing the processed image.
    
    Args:
        result_image (numpy.ndarray): The thresholded image.
    """
    
    # Segmentation related events
    APPLY_SEGMENTATION = "apply_segmentation"
    """
    Published when the user requests to apply a segmentation algorithm to an image.
    
    Args:
        image (numpy.ndarray): The image to be segmented.
        method (str): The selected segmentation method in lowercase 
                     (e.g., "k-means", "region growing", "agglomerative", "mean-shift").
    """
    
    SEGMENTATION_METHOD_CHANGED = "segmentation_method_changed"
    """
    Published when the user selects a different segmentation method from the dropdown menu.
    
    Args:
        method (str): The newly selected segmentation method in lowercase
                     (e.g., "k-means", "region growing", "agglomerative", "mean-shift").
    """
    
    SEGMENTATION_RESULT = "segmentation_result"
    """
    Published when segmentation is complete, containing the processed image.
    
    Args:
        result_image (numpy.ndarray): The segmented image.
    """