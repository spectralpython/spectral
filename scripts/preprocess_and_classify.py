"""
Preprocessing and PCA-based classification example
using Spectral Python library.
"""

import spectral
from spectral import open_image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---- Load sample hyperspectral image ----
# NOTE: Replace path with actual .hdr file when running
# Example datasets are mentioned in Spectral Python docs
# OPTION 1: Synthetic hyperspectral-like data (used for pipeline validation)
img = np.random.rand(50, 50, 20)

# OPTION 2: Real hyperspectral data (uncomment when dataset is available)
# img = open_image('data/example/your_dataset.hdr').load()

# NOTE:
# Synthetic data is used by default to validate the preprocessing and
# classification pipeline. Real hyperspectral datasets can be loaded
# by uncommenting the line below and providing a valid .hdr file path.

# ---- Basic preprocessing ----
# Normalize data
img = (img - np.mean(img)) / np.std(img)

# ---- Reshape for ML ----
h, w, bands = img.shape
pixels = img.reshape(-1, bands)

# ---- Dimensionality reduction ----
pca = PCA(n_components=10)
pixels_pca = pca.fit_transform(pixels)

# ---- Dummy labels (placeholder) ----
# In real datasets, labels come from ground truth files
labels = np.random.randint(0, 2, size=pixels_pca.shape[0])

# ---- Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(
    pixels_pca, labels, test_size=0.2, random_state=42
)

# ---- Train classifier ----
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc:.4f}")

# ---- Visualization ----
plt.imshow(img[:, :, :3])
plt.title("RGB Composite (First 3 Bands)")
plt.axis('off')
plt.show()