from PIL import Image

img = Image.open("images/DP167067.jpg")
colors = img.getcolors(maxcolors=1000000)  # (count, color)

# Sort by most frequent
colors = sorted(colors, key=lambda x: -x[0])

# Convert RGB to hex
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# Get top 5 colors and convert to hex
top_colors = colors[:5]
hex_colors = []

# Calculate total pixels for percentage
total_pixels = sum(count for count, _ in colors)

print("Top 5 colors (count, percentage, RGB, Hex):")
for count, rgb in top_colors:
    percentage = (count / total_pixels) * 100
    hex_color = rgb_to_hex(rgb)
    hex_colors.append(hex_color)
    print(f"Count: {count:,} ({percentage:.2f}%), RGB: {rgb}, Hex: {hex_color}")

print(f"\nHex palette: {hex_colors}")

# Show distribution summary
print(f"\nDistribution Summary:")
print(f"Total pixels: {total_pixels:,}")
print(f"Unique colors: {len(colors):,}")
print(f"Top 5 colors represent: {(sum(count for count, _ in top_colors) / total_pixels) * 100:.2f}% of the image")
