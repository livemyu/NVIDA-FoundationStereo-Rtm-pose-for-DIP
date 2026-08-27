import csv
import math
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

file_path = '/home/elp/picture_resize_recording_NVIDA/output/vio.csv'

x_vals = []
y_vals = []

with open(file_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        try:
            x_vals.append(float(row[1]))
            y_vals.append(float(row[2]))
        except ValueError:
            pass

# Calculate distance from origin over time
dists = []
start_x, start_y = x_vals[0], y_vals[0]
for x, y in zip(x_vals, y_vals):
    dist = math.sqrt((x - start_x)**2 + (y - start_y)**2)
    # Give it a sign based on projection onto the primary axis of movement
    dists.append(dist)

# To handle back and forth, it's better to project onto the PCA axis or just use the dominant axis
x_diff = x_vals[-1] - x_vals[0]
y_diff = y_vals[-1] - y_vals[0]

# Actually, if they walk back and forth, the Euclidean distance to the start point will oscillate.
# It will start at 0, go up to max distance, come back to ~0, go up to max distance, etc.
# Finding peaks in the distance-to-start array gives the number of times they reached the far end.
# Finding valleys gives the number of times they returned to the start.

dists = np.array(dists)

# Find peaks (when they reach the far end)
peaks, _ = find_peaks(dists, prominence=0.5, distance=50)

# Find valleys (when they return to the start)
# Invert distance to find valleys
valleys, _ = find_peaks(-dists, prominence=0.5, distance=50)

print(f"Number of peaks (far ends reached): {len(peaks)}")
print(f"Number of valleys (returns to start): {len(valleys)}")
print(f"Total round trips: {len(peaks)}")

# Plot it to visually confirm
plt.figure(figsize=(10, 4))
plt.plot(dists, label='Distance from Start (m)')
plt.plot(peaks, dists[peaks], 'ro', label='Far End (Peaks)')
plt.plot(valleys, dists[valleys], 'go', label='Start (Valleys)')
plt.xlabel('Frame')
plt.ylabel('Distance (m)')
plt.title('Forward/Backward Movement Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('/home/elp/.gemini/antigravity-ide/brain/2c8194dd-e972-443c-8e4f-b30e9b5af3a6/trips.png')
print("Trips plot saved.")
