import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def roberts_cross_edge_detection(image):
    """Apply Roberts Cross edge detection with improved normalization."""
    image = image.astype(np.float32) / 255.0  # Normalize image

    # Roberts Cross kernels
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Apply convolution
    grad_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)

    # Compute magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi)  # Convert to degrees
    orientation[orientation < 0] += 180  # Ensure angles are in [0, 180]

    return magnitude, orientation

def compute_hog_features(image, cell_size=(8, 8), block_size=(3, 3), bins=12):
    """Compute improved HoG features using Roberts Cross Edge Detector."""
    h, w = image.shape
    magnitude, orientation = roberts_cross_edge_detection(image)

    # Define histogram bins
    bin_edges = np.linspace(0, 180, bins + 1)  # More bins for finer details

    # Compute cell histograms
    cell_h, cell_w = h // cell_size[0], w // cell_size[1]
    histograms = np.zeros((cell_h, cell_w, bins))

    for i in range(cell_h):
        for j in range(cell_w):
            # Get cell region
            mag_cell = magnitude[i * cell_size[0]:(i + 1) * cell_size[0], 
                                 j * cell_size[1]:(j + 1) * cell_size[1]]
            ori_cell = orientation[i * cell_size[0]:(i + 1) * cell_size[0], 
                                   j * cell_size[1]:(j + 1) * cell_size[1]]

            # Compute histogram for the cell
            hist, _ = np.histogram(ori_cell, bins=bin_edges, weights=mag_cell)
            histograms[i, j, :] = hist

    # Normalize histograms in overlapping blocks
    block_h, block_w = cell_h - block_size[0] + 1, cell_w - block_size[1] + 1
    features = []

    for i in range(block_h):
        for j in range(block_w):
            block = histograms[i:i + block_size[0], j:j + block_size[1], :].flatten()
            norm = np.linalg.norm(block) + 1e-6  # Avoid division by zero
            features.append(block / norm)

    return np.concatenate(features)

def preprocess_image(image_path):
    """Load image, apply preprocessing, and extract HoG features."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Resize for consistency
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # Apply Gaussian Blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return compute_hog_features(image)

def load_dataset(dataset_path):
    """Load images and extract features."""
    print(f"Loading dataset from {dataset_path}...")
    X, y = [], []
    labels = {'cats': 0, 'dogs': 1}

    for label in labels.keys():
        folder_path = os.path.join(dataset_path, label)
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found.")
            continue
        
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Warning: Unable to read image {img_path}")
                continue
            
            features = preprocess_image(img_path)
            X.append(features)
            y.append(labels[label])

    print(f"Finished loading dataset. Total samples: {len(X)}")
    return np.array(X), np.array(y)

# Load training and test datasets
train_path = "training_set"
test_path = "test_set"

print("Loading training dataset...")
X_train, y_train = load_dataset(train_path)
print("Loading test dataset...")
X_test, y_test = load_dataset(test_path)

# Feature Scaling
print("Applying feature scaling...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
print("Applying PCA...")
pca = PCA(n_components=250)  # Increase components to retain more features
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Train a Random Forest classifier with optimized hyperparameters
print("Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=800, max_depth=30, min_samples_split=3, random_state=42)
clf.fit(X_train, y_train)

# Predictions and evaluation
print("Making predictions...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

