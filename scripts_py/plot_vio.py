import csv
import matplotlib.pyplot as plt
import os

file_path = '/home/elp/picture_resize_recording_NVIDA/output/vio.csv'
out_path = '/home/elp/.gemini/antigravity-ide/brain/2c8194dd-e972-443c-8e4f-b30e9b5af3a6/trajectory.png'

x_vals = []
y_vals = []
z_vals = []

with open(file_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        try:
            x_vals.append(float(row[1]))
            y_vals.append(float(row[2]))
            z_vals.append(float(row[3]))
        except ValueError:
            pass

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Top-down view
ax1.plot(x_vals, y_vals, label='Trajectory', color='blue')
ax1.scatter(x_vals[0], y_vals[0], color='green', s=100, label='Start', marker='o')
ax1.scatter(x_vals[-1], y_vals[-1], color='red', s=100, label='End', marker='x')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Top-Down View (X-Y Plane)')
ax1.axis('equal')
ax1.grid(True)
ax1.legend()

# Altitude profile
ax2.plot(range(len(z_vals)), z_vals, label='Z (Altitude)', color='purple')
ax2.set_xlabel('Frame Step')
ax2.set_ylabel('Z (m)')
ax2.set_title('Altitude Profile (Z Axis)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig(out_path, dpi=150)
print("Trajectory plotted successfully.")
