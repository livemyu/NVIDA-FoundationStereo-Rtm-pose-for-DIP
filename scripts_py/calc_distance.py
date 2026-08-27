import csv
import math

file_path = '/home/elp/picture_resize_recording_NVIDA/output/vio.csv'
positions = []

with open(file_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        try:
            # timestamp, x, y, z, qw, qx, qy, qz, ...
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
            positions.append((x, y, z))
        except ValueError:
            pass

if not positions:
    print("No positions found.")
else:
    start_pt = positions[0]
    end_pt = positions[-1]
    
    total_dist = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        dz = positions[i][2] - positions[i-1][2]
        total_dist += math.sqrt(dx*dx + dy*dy + dz*dz)
        
    straight_dist = math.sqrt((end_pt[0]-start_pt[0])**2 + (end_pt[1]-start_pt[1])**2 + (end_pt[2]-start_pt[2])**2)
    
    print(f"Start Point: X={start_pt[0]:.4f}, Y={start_pt[1]:.4f}, Z={start_pt[2]:.4f}")
    print(f"End Point: X={end_pt[0]:.4f}, Y={end_pt[1]:.4f}, Z={end_pt[2]:.4f}")
    print(f"Straight-line Distance: {straight_dist:.4f} meters")
    print(f"Total Trajectory Distance: {total_dist:.4f} meters")
