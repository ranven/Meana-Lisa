from skimage.io import imread
from skimage.transform import resize
from sklearn.cluster import KMeans
import numpy as np
import time

# Download and resize image to reduce computation
print("Downloading image...")
start_time = time.time()
img = imread("https://images.metmuseum.org/CRDImages/ep/original/DP124058.jpg")
print(f"Original image shape: {img.shape}")

# Resize image to max 300x300 pixels for much faster processing
max_size = 300
h, w = img.shape[:2]
if h > w:
    new_h, new_w = max_size, int(w * max_size / h)
else:
    new_h, new_w = int(h * max_size / w), max_size

img_resized = resize(img, (new_h, new_w), anti_aliasing=True)
img_resized = (img_resized * 255).astype(np.uint8)
print(f"Resized image shape: {img_resized.shape}")

# Sample pixels instead of using all pixels (every 4th pixel)
img_sampled = img_resized[::4, ::4]
print(f"Sampled image shape: {img_sampled.shape}")

img_flat = img_sampled.reshape((-1, 3))
print(f"Total pixels to process: {len(img_flat)}")

# Reduce clusters for faster computation
print("Running K-means clustering...")
kmeans = KMeans(n_clusters=10, random_state=0, n_init=10).fit(img_flat)
palette = kmeans.cluster_centers_.astype(int)
labels = kmeans.labels_

# Convert RGB to hex
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

hex_palette = [rgb_to_hex(color) for color in palette]

# Calculate weights (percentages) for each color
total_pixels = len(labels)
color_counts = np.bincount(labels)
color_percentages = (color_counts / total_pixels) * 100

# Create array of tuples (hex_color, percentage) and sort by percentage (highest first)
color_distribution = [(hex_color, float(percentage)) for hex_color, percentage in zip(hex_palette, color_percentages)]
color_distribution.sort(key=lambda x: x[1], reverse=True)



end_time = time.time()
total_time = end_time - start_time

print(f"\nProcessing completed in {total_time:.2f} seconds")
print("\nColor Distribution Array (sorted by percentage, highest first):")
print(color_distribution)
