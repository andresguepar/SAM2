"""
Utility functions for calculating mask metrics in tests.
Based on functions from test1.py
"""

import numpy as np
from scipy.ndimage import label, find_objects, center_of_mass
from skimage.measure import perimeter


def calculate_mask_metrics(mask: np.ndarray, image: np.ndarray = None) -> dict:
    """Calculate comprehensive mask metrics including shape, color, and texture features."""
    try:
        # Ensure mask is 2D
        mask_2d = mask.squeeze()
        if mask_2d.ndim != 2:
            raise ValueError("Mask must be 2D")
            
        # Area calculation
        area = np.sum(mask_2d)
        
        # Bounding box calculation
        labeled, num = label(mask_2d)
        slices = find_objects(labeled)
        
        if slices and slices[0]:
            bbox = slices[0]
            min_row, min_col = bbox[0].start, bbox[1].start
            max_row, max_col = bbox[0].stop, bbox[1].stop
            width = max_col - min_col
            height = max_row - min_row
            bbox_tuple = (min_row, min_col, max_row, max_col)
        else:
            bbox_tuple = (0, 0, 0, 0)
            width = height = 0
        
        # Perimeter calculation
        peri = perimeter(mask_2d.astype(np.uint8), neighborhood=8) if area > 0 else 0
        
        # Centroid calculation
        centroid = center_of_mass(mask_2d) if area > 0 else (0.0, 0.0)
        
        # Shape features
        aspect_ratio = width / height if height > 0 else 0.0
        compactness = (4 * np.pi * area) / (peri ** 2) if peri > 0 else 0.0
        solidity = area / (width * height) if width * height > 0 else 0.0
        
        # Color statistics (if image provided)
        if image is not None and mask_2d.shape == image.shape[:2]:
            masked_pixels = image[mask_2d.astype(bool)]
            if masked_pixels.size > 0:
                mean_color = np.mean(masked_pixels, axis=0)
                var_color = np.var(masked_pixels, axis=0)
            else:
                mean_color = var_color = np.zeros(3)
        else:
            mean_color = var_color = np.zeros(3)
        
        return {
            'area': int(area),
            'bbox': bbox_tuple,
            'perimeter': float(peri),
            'centroid': tuple(float(x) for x in centroid),
            'aspect_ratio': float(aspect_ratio),
            'compactness': float(compactness),
            'solidity': float(solidity),
            'mean_color': mean_color.tolist(),
            'var_color': var_color.tolist(),
            'valid': True
        }
        
    except Exception as e:
        print(f"Error calculating mask metrics: {e}")
        return {
            'area': 0, 'bbox': (0, 0, 0, 0), 'perimeter': 0.0,
            'centroid': (0.0, 0.0), 'aspect_ratio': 0.0,
            'compactness': 0.0, 'solidity': 0.0,
            'mean_color': [0.0, 0.0, 0.0], 'var_color': [0.0, 0.0, 0.0],
            'valid': False
        }


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0 