import os

def convert_csv_to_tum(csv_path, tum_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    count = 0
    with open(csv_path, 'r') as fin, open(tum_path, 'w') as fout:
        for line in fin:
            line = line.strip().rstrip(',')
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) >= 8:
                try:
                    t_val = float(parts[0])
                    # If timestamp is in nanoseconds (> 1e12), convert to seconds
                    if t_val > 1e12:
                        t_val = t_val / 1e9
                    
                    x = parts[1]
                    y = parts[2]
                    z = parts[3]
                    qw = parts[4]
                    qx = parts[5]
                    qy = parts[6]
                    qz = parts[7]
                    # TUM format: timestamp x y z qx qy qz qw
                    fout.write(f"{t_val:.6f} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
                    count += 1
                except ValueError:
                    continue
    print(f"Converted {count} poses from {csv_path} -> {tum_path}")

if __name__ == "__main__":
    convert_csv_to_tum("output/vio.csv", "output/vio_tum.txt")
    convert_csv_to_tum("output/vio_loop.csv", "output/vio_loop_tum.txt")
