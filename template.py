import numpy as np
import os

FOLDER = "retina_database"

def save_template(template, individual, index_sample):
    sample_name = individual
    if (not index_sample is None):
        sample_name+=f"_{index_sample}"
    path_to_individual = os.path.join(FOLDER, sample_name)
    np.save(path_to_individual, template)

def bresenham_circle(radius):
    """Generate coordinates of a circle using Bresenham's algorithm."""
    x = 0
    y = radius
    d = 3 - 2 * radius
    coords = set()
    while x <= y:
        coords.add((x, y))
        coords.add((-x, y))
        coords.add((x, -y))
        coords.add((-x, -y))
        coords.add((y, x))
        coords.add((-y, x))
        coords.add((y, -x))
        coords.add((-y, -x))
        if d < 0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y -= 1
        x += 1
    return coords

def generate_code(image, center, num_circles=15, navg=3):
    """Get the values of the concentric circles around a center point."""
    # Generate the coordinates of the circles
    max_radius = min(center[0], center[1], image.shape[0] - center[0], image.shape[1] - center[1])
    radii = np.linspace(0, max_radius, num_circles)

    # Extract the values of the circles
    circle_values = np.zeros((num_circles, 360))
    for i, r in enumerate(radii):
        for j in range(360):
            angle = j * np.pi / 180
            x = int(center[0] + round(r * np.cos(angle)))
            y = int(center[1] + round(r * np.sin(angle)))
            values = []
            for dx in range(-navg, navg+1):
                for dy in range(-navg, navg+1):
                    if dx == 0 and dy == 0:
                        continue
                    if x+dx >= 0 and x+dx < image.shape[0] and y+dy >= 0 and y+dy < image.shape[1]:
                        values.append(image[x+dx, y+dy])
            if len(values) > 0:
                val = sum(values) / len(values)
                circle_values[i, j] = 1 if val>0 else 0
    return circle_values

def compute_template(image, optic_disk_center):
    code = generate_code(image, optic_disk_center)
    if (np.all(code <= 0)):
        code = None
    return code