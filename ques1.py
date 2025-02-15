import numpy as np

def mouth_aspect_ratio(p1, p2, p3, p4):
    """
    Calculates the Mouth Aspect Ratio (MAR) to measure how open the mouth is.
    MAR = (Vertical Distance P2-P3) / (Horizontal Distance P1-P4)
    
    Parameters:
    - p1: (x, y) Left mouth corner
    - p2: (x, y) Upper lip center
    - p3: (x, y) Lower lip center
    - p4: (x, y) Right mouth corner
    
    Returns:
    - MAR value (higher means a more open mouth)
    """
    vertical_dist = np.linalg.norm(np.array(p2) - np.array(p3))
    horizontal_dist = np.linalg.norm(np.array(p1) - np.array(p4))
    
    return vertical_dist / horizontal_dist if horizontal_dist != 0 else 0

def mouth_curvature(p1, p4, p2):
    """
    Determines if the mouth corners curve upwards (a sign of smiling).
    
    Parameters:
    - p1: (x, y) Left mouth corner
    - p4: (x, y) Right mouth corner
    - p2: (x, y) Upper lip center
    
    Returns:
    - True if the upper lip is higher than the midpoint of the mouth corners
    """
    mid_y = (p1[1] + p4[1]) / 2  # Average y-coordinate of mouth corners
    return p2[1] < mid_y  # True if top lip is above the midpoint (smile detected)

def mouth_symmetry(p1, p2, p3, p4):
    """
    Checks the symmetry of the mouth by comparing distances from the center of the lips to the corners.
    
    Parameters:
    - p1: (x, y) Left mouth corner
    - p2: (x, y) Upper lip center
    - p3: (x, y) Lower lip center
    - p4: (x, y) Right mouth corner
    
    Returns:
    - True if the mouth is symmetric, False otherwise
    """
    left_dist = np.linalg.norm(np.array(p1) - np.array(p2))
    right_dist = np.linalg.norm(np.array(p4) - np.array(p3))
    
    return abs(left_dist - right_dist) < 5  # Threshold for symmetry

def is_smiling(p1, p2, p3, p4):
    """
    Determines if a person is smiling using MAR, mouth curvature, and symmetry.
    Returns True if the person is smiling, otherwise False.
    """
    mar = mouth_aspect_ratio(p1, p2, p3, p4)
    curvature = mouth_curvature(p1, p4, p2)
    symmetry = mouth_symmetry(p1, p2, p3, p4)

    # Define thresholds for smiling
    MAR_THRESHOLD = 0.3  # If MAR is above this, the mouth is open (possible smile)
    
    # A smile occurs when:
    # - Mouth is open enough (MAR > threshold)
    # - Mouth corners are curved upwards
    # - Lips are reasonably symmetric
    if mar > MAR_THRESHOLD and curvature and symmetry:
        return True
    else:
        return False

# Example key points (x, y)
p1 = (30, 50)  # Left mouth corner
p2 = (50, 40)  # Upper lip center
p3 = (50, 60)  # Lower lip center
p4 = (70, 50)  # Right mouth corner

# Manually calculate feature values
mar_value = mouth_aspect_ratio(p1, p2, p3, p4)
curvature = mouth_curvature(p1, p4, p2)
symmetry = mouth_symmetry(p1, p2, p3, p4)

# Final decision: Is the person smiling?
smiling = is_smiling(p1, p2, p3, p4)

# Display the results
print("Mouth Aspect Ratio (MAR):", mar_value)
print("Mouth Curvature (Smile Detected?):", curvature)
print("Mouth Symmetry (Balanced Lips?):", symmetry)
print("\nFinal Decision: The person is", "SMILING" if smiling else "NOT Smiling")
