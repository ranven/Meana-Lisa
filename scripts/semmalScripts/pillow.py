from PIL import Image
from collections import Counter
import numpy as np

img = Image.open("images/DP167067.jpg")

# Convert to RGB if not already
if img.mode != 'RGB':
    img = img.convert('RGB')

# Convert to numpy array for easier processing
img_array = np.array(img)

# Filter out very light colors (background) - pixels with brightness > 240
brightness = np.mean(img_array, axis=2)
mask = brightness < 240  # Keep darker pixels (the bat)

# Get only the non-background pixels
foreground_pixels = img_array[mask]

# Convert back to list of RGB tuples
pixels = [tuple(pixel) for pixel in foreground_pixels]

# Count color frequencies
color_counts = Counter(pixels)

# Sort by most frequent
colors = sorted(color_counts.items(), key=lambda x: -x[1])

# Convert RGB to hex
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# Get top 5 colors and convert to hex
top_colors = colors[:5]
hex_colors = []

# Calculate total pixels for percentage
total_pixels = sum(count for _, count in colors)

print("Top 5 colors (count, percentage, RGB, Hex):")
for rgb, count in top_colors:
    percentage = (count / total_pixels) * 100
    hex_color = rgb_to_hex(rgb)
    hex_colors.append(hex_color)
    print(f"Count: {count:,} ({percentage:.2f}%), RGB: {rgb}, Hex: {hex_color}")

print(f"\nHex palette: {hex_colors}")

# Show distribution summary
print(f"\nDistribution Summary:")
print(f"Total pixels analyzed: {total_pixels:,}")
print(f"Unique colors: {len(colors):,}")
print(f"Top 5 colors represent: {(sum(count for _, count in top_colors) / total_pixels) * 100:.2f}% of the foreground")

# Show some additional debug info
print(f"\nDebug Info:")
print(f"Original image size: {img.size}")
print(f"Total pixels in image: {img.size[0] * img.size[1]:,}")
print(f"Foreground pixels analyzed: {len(pixels):,}")
print(f"Background pixels filtered out: {(img.size[0] * img.size[1]) - len(pixels):,}")
