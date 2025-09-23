from skimage.io import imread
from sklearn.cluster import KMeans
import numpy as np

img = imread("https://images.metmuseum.org/CRDImages/ep/original/DP124058.jpg")
img = img.reshape((-1, 3))

kmeans = KMeans(n_clusters=20, random_state=0).fit(img)
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

print("RGB Palette:", palette)
print("Hex Palette:", hex_palette)
print("\nColor Distribution Array (sorted by percentage, highest first):")
print(color_distribution)
