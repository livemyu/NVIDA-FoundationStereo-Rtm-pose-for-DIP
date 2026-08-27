#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-Euro Filter for Real-time 2D & 3D Keypoint Smoothing
======================================================
Reference:
Casiez, G., Roussel, N., & Vogel, D. (2012). 
1 € filter: a simple speed-based low-pass filter for noisy input in HCI.
ACM CHI 2012.
"""

import numpy as np

def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        min_cutoff: Minimum cutoff frequency in Hz (smaller = more jitter reduction in stillness)
        beta: Speed coefficient (larger = faster cutoff scaling to reduce lag in fast motion)
        d_cutoff: Cutoff frequency for derivative filtering in Hz
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t=None):
        x = np.array(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        t_e = 1.0 / 30.0 if (t is None or self.t_prev is None or t <= self.t_prev) else (t - self.t_prev)
        self.t_prev = t

        # Filter the derivative to estimate speed
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # Dynamic cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

def determine_hand_chirality(kpts):
    """
    Computes hand chirality (Left vs Right) using palm knuckle and thumb orientation.
    kpts: (21, 3: px_u, px_v, conf)
    Returns: 'Left', 'Right', or 'Unknown'
    """
    if len(kpts) < 21:
        return 'Unknown'
    
    # 0: Wrist, 4: Thumb tip, 5: Index MCP, 9: Middle MCP, 17: Pinky MCP
    p0 = kpts[0][:2]
    p4 = kpts[4][:2]
    p5 = kpts[5][:2]
    p9 = kpts[9][:2]
    p17 = kpts[17][:2]
    
    # Check confidences
    confs = [kpts[i][2] for i in [0, 4, 5, 9, 17]]
    if min(confs) < 0.20:
        return 'Unknown'
        
    v_palm = p9 - p0
    u_trans = p5 - p17
    v_thumb = p4 - p5
    
    cp_palm = u_trans[0] * v_palm[1] - u_trans[1] * v_palm[0]
    cp_thumb = v_palm[0] * v_thumb[1] - v_palm[1] * v_thumb[0]
    
    # Positive cp_thumb indicates thumb is on the right side of the palm axis
    # In standard camera view (looking at hands), Left hand thumb points to the right (cp_thumb > 0)
    # Right hand thumb points to the left (cp_thumb < 0)
    if cp_palm > 0:
        return 'Left' if cp_thumb > 0 else 'Right'
    else:
        return 'Right' if cp_thumb > 0 else 'Left'

class HandTrack:
    def __init__(self, track_id, bbox, kpts, min_cutoff_2d, beta_2d, min_cutoff_3d, beta_3d, timestamp_sec=None):
        self.track_id = track_id
        self.bbox = bbox
        self.center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.last_timestamp = timestamp_sec
        self.missed_count = 0
        
        self.chirality = determine_hand_chirality(kpts)
        self.chirality_votes = [self.chirality] if self.chirality != 'Unknown' else []
        
        self.filters_2d = [OneEuroFilter(min_cutoff_2d, beta_2d) for _ in range(21)]
        self.filters_3d = [OneEuroFilter(min_cutoff_3d, beta_3d) for _ in range(21)]
        
        self.last_valid_kpts_2d = kpts.copy()
        self.last_valid_kpts_3d = [[0.0, 0.0, 0.0, 0.0] for _ in range(21)]
        self.avg_depth_z = 0.0

    def predict(self, dt=0.033):
        return self.center + self.velocity * dt

    def update(self, bbox, kpts, timestamp_sec=None):
        new_center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
        if timestamp_sec is not None and self.last_timestamp is not None:
            dt = max(0.001, timestamp_sec - self.last_timestamp)
            inst_v = (new_center - self.center) / dt
            self.velocity = 0.7 * self.velocity + 0.3 * inst_v
            self.last_timestamp = timestamp_sec
        
        self.center = new_center
        self.bbox = bbox
        self.missed_count = 0
        
        # Update chirality with majority vote
        new_ch = determine_hand_chirality(kpts)
        if new_ch != 'Unknown':
            self.chirality_votes.append(new_ch)
            if len(self.chirality_votes) > 15:
                self.chirality_votes.pop(0)
            # Majority vote
            left_count = self.chirality_votes.count('Left')
            right_count = self.chirality_votes.count('Right')
            self.chirality = 'Left' if left_count >= right_count else 'Right'

class HandPoseSmoother3D:
    """
    Advanced Multi-Hand Spatial Tracker & 1-Euro Filter.
    Features:
    - Hand Chirality (Left vs Right) Separation (Prevents crossing ID swaps)
    - Constant Velocity Motion Prediction
    - Bipartite Matching with Spatial & Chirality Cost Matrix
    - Occlusion Freezing (prevents ghost lines when hands overlap)
    - Rate-limited 3D depth bleeding protection
    """
    def __init__(self, min_cutoff_2d=0.8, beta_2d=0.005, min_cutoff_3d=0.6, beta_3d=0.004):
        self.min_cutoff_2d = min_cutoff_2d
        self.beta_2d = beta_2d
        self.min_cutoff_3d = min_cutoff_3d
        self.beta_3d = beta_3d
        
        self.tracks = {} # track_id -> HandTrack
        self.next_track_id = 0
        self.max_missed_frames = 8

    def smooth(self, poses, timestamp_sec=None):
        """
        poses: list of dicts with 'bbox' and 'kpts' (21, 3)
        returns smoothed_poses with stable 'hand_id' and 'chirality'
        """
        num_dets = len(poses)
        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        
        # 1. Predict track positions
        predicted_centers = {}
        for t_id in track_ids:
            predicted_centers[t_id] = self.tracks[t_id].predict()
            
        # 2. Build Bipartite Cost Matrix
        # Cost = Euclidean Distance + Chirality Mismatch Penalty
        det_assignments = [-1] * num_dets
        track_assignments = {t_id: -1 for t_id in track_ids}
        
        if num_dets > 0 and num_tracks > 0:
            cost_matrix = np.zeros((num_dets, num_tracks), dtype=np.float32)
            for i, p in enumerate(poses):
                bb = p['bbox']
                det_c = np.array([(bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5], dtype=np.float32)
                det_ch = determine_hand_chirality(p['kpts'])
                
                for j, t_id in enumerate(track_ids):
                    track = self.tracks[t_id]
                    pred_c = predicted_centers[t_id]
                    dist = float(np.linalg.norm(det_c - pred_c))
                    
                    # Chirality Hard Penalty
                    if det_ch != 'Unknown' and track.chirality != 'Unknown' and det_ch != track.chirality:
                        dist += 800.0 # Huge penalty prevents Left/Right swap when hands cross
                        
                    cost_matrix[i, j] = dist
                    
            # Greedy Minimum-Cost Matching
            used_tracks = set()
            for _ in range(min(num_dets, num_tracks)):
                min_val = np.min(cost_matrix)
                if min_val > 350.0: # Distance threshold
                    break
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                det_idx, track_idx = min_idx[0], min_idx[1]
                t_id = track_ids[track_idx]
                
                det_assignments[det_idx] = t_id
                track_assignments[t_id] = det_idx
                used_tracks.add(t_id)
                
                cost_matrix[det_idx, :] = 1e6
                cost_matrix[:, track_idx] = 1e6

        # 3. Create new tracks for unmatched detections
        for i in range(num_dets):
            if det_assignments[i] == -1:
                t_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[t_id] = HandTrack(
                    track_id=t_id,
                    bbox=poses[i]['bbox'],
                    kpts=poses[i]['kpts'],
                    min_cutoff_2d=self.min_cutoff_2d,
                    beta_2d=self.beta_2d,
                    min_cutoff_3d=self.min_cutoff_3d,
                    beta_3d=self.beta_3d,
                    timestamp_sec=timestamp_sec
                )
                det_assignments[i] = t_id

        # 4. Update matched tracks & handle missed tracks
        for t_id, track in list(self.tracks.items()):
            if track_assignments.get(t_id, -1) == -1:
                track.missed_count += 1
                if track.missed_count > self.max_missed_frames:
                    del self.tracks[t_id]

        # 5. Smooth Keypoints with Occlusion Resilience
        smoothed = []
        for i, p in enumerate(poses):
            t_id = det_assignments[i]
            track = self.tracks[t_id]
            track.update(p['bbox'], p['kpts'], timestamp_sec)
            
            bb = p['bbox']
            kpts = p['kpts']
            smoothed_kpts = np.zeros_like(kpts)
            
            for k_i in range(21):
                raw_uv = kpts[k_i][:2]
                conf = kpts[k_i][2]
                
                if conf >= 0.22:
                    # Good confidence: update 1-Euro filter
                    smooth_uv = track.filters_2d[k_i](raw_uv, timestamp_sec)
                    track.last_valid_kpts_2d[k_i] = [smooth_uv[0], smooth_uv[1], conf]
                else:
                    # Occluded / low confidence during crossing:
                    # Freeze joint position with last valid coordinate and smoothly decay confidence
                    prev_uv = track.last_valid_kpts_2d[k_i][:2]
                    # Apply small velocity drift
                    smooth_uv = prev_uv + track.velocity * 0.01
                    smooth_uv = track.filters_2d[k_i](smooth_uv, timestamp_sec)
                    conf = max(0.05, conf)
                    
                smoothed_kpts[k_i] = [smooth_uv[0], smooth_uv[1], conf]
                
            smoothed.append({
                'hand_id': t_id,
                'chirality': track.chirality,
                'bbox': bb,
                'kpts': smoothed_kpts
            })
            
        return smoothed

    def smooth_3d(self, hand_id, kpts_3d, timestamp_sec=None):
        """
        kpts_3d: list of 21 elements [x_m, y_m, z_m, conf]
        Applies 3D One-Euro filter + depth edge jump clamping.
        """
        if hand_id not in self.tracks:
            # Fallback filter
            return kpts_3d
            
        track = self.tracks[hand_id]
        smooth_3d_res = []
        
        valid_zs = [k[2] for k in kpts_3d if k[2] > 0.1 and k[3] > 0.25]
        curr_median_z = float(np.median(valid_zs)) if len(valid_zs) > 0 else track.avg_depth_z
        if curr_median_z > 0.1:
            if track.avg_depth_z == 0.0:
                track.avg_depth_z = curr_median_z
            else:
                track.avg_depth_z = 0.85 * track.avg_depth_z + 0.15 * curr_median_z
                
        for i in range(21):
            xm, ym, zm, conf = kpts_3d[i]
            
            # Check for depth bleeding / extreme spike when overlapping
            if zm > 0.1 and conf > 0.20:
                # If depth jumps wildly (> 18cm from hand average depth), clamp to hand average
                if track.avg_depth_z > 0.1 and abs(zm - track.avg_depth_z) > 0.18:
                    zm = float(track.avg_depth_z)
                    
                filtered_xyz = track.filters_3d[i]([xm, ym, zm], timestamp_sec)
                res_point = [float(filtered_xyz[0]), float(filtered_xyz[1]), float(filtered_xyz[2]), float(conf)]
                track.last_valid_kpts_3d[i] = res_point
                smooth_3d_res.append(res_point)
            else:
                # Occluded joint in 3D: retain last valid 3D position
                if track.last_valid_kpts_3d[i][2] > 0.1:
                    last_pt = track.last_valid_kpts_3d[i]
                    smooth_3d_res.append([last_pt[0], last_pt[1], last_pt[2], float(max(0.05, conf))])
                else:
                    smooth_3d_res.append([xm, ym, zm, conf])
                    track.filters_3d[i].reset()
                    
        return smooth_3d_res

class BodyPoseSmoother3D:
    """Manages 1-Euro filters for multi-person 17 2D and 3D keypoints"""
    def __init__(self, min_cutoff_2d=0.7, beta_2d=0.005, min_cutoff_3d=0.5, beta_3d=0.004):
        self.min_cutoff_2d = min_cutoff_2d
        self.beta_2d = beta_2d
        self.min_cutoff_3d = min_cutoff_3d
        self.beta_3d = beta_3d
        
        self.persons_filters_2d = {}
        self.persons_filters_3d = {}
        self.prev_bboxes = {}

    def _match_person(self, bbox):
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        
        best_id = None
        min_dist = 350.0
        
        for p_id, p_bb in self.prev_bboxes.items():
            pcx = (p_bb[0] + p_bb[2]) * 0.5
            pcy = (p_bb[1] + p_bb[3]) * 0.5
            dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
            if dist < min_dist:
                min_dist = dist
                best_id = p_id
                
        if best_id is None:
            existing = set(self.persons_filters_2d.keys())
            for i in range(8):
                if i not in existing:
                    best_id = i
                    break
            if best_id is None:
                best_id = 0
                
            self.persons_filters_2d[best_id] = [OneEuroFilter(self.min_cutoff_2d, self.beta_2d) for _ in range(17)]
            self.persons_filters_3d[best_id] = [OneEuroFilter(self.min_cutoff_3d, self.beta_3d) for _ in range(17)]
            
        self.prev_bboxes[best_id] = bbox
        return best_id

    def smooth(self, poses, timestamp_sec=None):
        smoothed = []
        active_ids = set()
        
        for p in poses:
            bb = p['bbox']
            kpts = p['kpts'] # (17, 3: px_u, px_v, conf)
            
            p_id = self._match_person(bb)
            active_ids.add(p_id)
            
            smoothed_kpts = np.zeros_like(kpts)
            for i in range(17):
                raw_uv = kpts[i][:2]
                conf = kpts[i][2]
                if conf > 0.2:
                    smooth_uv = self.persons_filters_2d[p_id][i](raw_uv, timestamp_sec)
                else:
                    smooth_uv = raw_uv
                    self.persons_filters_2d[p_id][i].reset()
                smoothed_kpts[i] = [smooth_uv[0], smooth_uv[1], conf]
                
            smoothed.append({
                'person_id': p_id,
                'bbox': bb,
                'kpts': smoothed_kpts
            })

        for old_id in list(self.prev_bboxes.keys()):
            if old_id not in active_ids:
                del self.prev_bboxes[old_id]
                if old_id in self.persons_filters_2d:
                    del self.persons_filters_2d[old_id]
                if old_id in self.persons_filters_3d:
                    del self.persons_filters_3d[old_id]

        return smoothed

    def smooth_3d(self, person_id, kpts_3d, timestamp_sec=None):
        if person_id not in self.persons_filters_3d:
            self.persons_filters_3d[person_id] = [OneEuroFilter(self.min_cutoff_3d, self.beta_3d) for _ in range(17)]
            
        smooth_3d = []
        for i in range(17):
            xm, ym, zm, conf = kpts_3d[i]
            if zm > 0.1 and conf > 0.2:
                filtered_xyz = self.persons_filters_3d[person_id][i]([xm, ym, zm], timestamp_sec)
                smooth_3d.append([float(filtered_xyz[0]), float(filtered_xyz[1]), float(filtered_xyz[2]), float(conf)])
            else:
                smooth_3d.append([xm, ym, zm, conf])
                self.persons_filters_3d[person_id][i].reset()
        return smooth_3d
