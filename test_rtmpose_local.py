import cv2
import numpy as np
import onnxruntime as ort

def letterbox_det(im, new_shape=(320, 320), color=(114, 114, 114)):
    shape = im.shape[:2] # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = 0, dh
    left, right = 0, dw
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (dw, dh)

def get_affine_transform(center, scale, output_size, rot=0):
    # scale: [w, h]
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    
    rot_rad = np.pi * rot / 180
    src_dir = np.array([0, src_w * -0.5], np.float32)
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_dir = np.array([src_dir[0] * cs - src_dir[1] * sn, src_dir[0] * sn + src_dir[1] * cs])
    
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = src[1, :] + np.array([-src_dir[1], src_dir[0]])
    
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = dst[1, :] + np.array([-dst_dir[1], dst_dir[0]])
    
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def bbox_to_center_scale(bbox, padding=1.25):
    # bbox: [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)
    size = max(w, h) * padding
    scale = np.array([size, size], dtype=np.float32)
    return center, scale

class RTMPoseHandDetector:
    def __init__(self, det_onnx, pose_onnx):
        self.det_sess = ort.InferenceSession(det_onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.pose_sess = ort.InferenceSession(pose_onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # Detector normalization (BGR)
        self.det_mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.det_std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        
        # Pose normalization (RGB)
        self.pose_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        self.pose_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
        
    def detect_hands(self, img_bgr, conf_thr=0.3):
        # img_bgr: H x W x 3
        pad_img, r, (dw, dh) = letterbox_det(img_bgr, (320, 320))
        norm_img = (pad_img.astype(np.float32) - self.det_mean) / self.det_std
        inp = np.transpose(norm_img, (2, 0, 1))[None, ...].astype(np.float32)
        
        outputs = self.det_sess.run(None, {'input': inp})
        dets = outputs[0][0] # [N, 5] (x1, y1, x2, y2, score)
        
        valid_bboxes = []
        for det in dets:
            score = det[4]
            if score > conf_thr:
                # Map back to original image
                x1 = det[0] / r
                y1 = det[1] / r
                x2 = det[2] / r
                y2 = det[3] / r
                valid_bboxes.append([x1, y1, x2, y2, score])
        return valid_bboxes

    def estimate_pose(self, img_bgr, bboxes):
        if len(bboxes) == 0:
            return []
        
        results = []
        batch_crops = []
        centers_scales = []
        inv_trans_list = []
        
        for bbox in bboxes:
            center, scale = bbox_to_center_scale(bbox[:4], padding=1.25)
            trans = get_affine_transform(center, scale, (256, 256))
            inv_trans = cv2.invertAffineTransform(trans)
            
            crop = cv2.warpAffine(img_bgr, trans, (256, 256), flags=cv2.INTER_LINEAR)
            # Convert BGR to RGB for pose
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            norm_crop = (crop_rgb.astype(np.float32) - self.pose_mean) / self.pose_std
            inp_crop = np.transpose(norm_crop, (2, 0, 1))
            
            batch_crops.append(inp_crop)
            centers_scales.append((center, scale))
            inv_trans_list.append(inv_trans)
            
        batch_crops = np.ascontiguousarray(np.stack(batch_crops, axis=0), dtype=np.float32)
        
        outputs = self.pose_sess.run(None, {'input': batch_crops})
        simcc_x, simcc_y = outputs[0], outputs[1] # [B, 21, 512]
        
        for i in range(len(bboxes)):
            sx = simcc_x[i] # [21, 512]
            sy = simcc_y[i] # [21, 512]
            
            loc_x = np.argmax(sx, axis=-1) / 2.0 # [21] in [0..256]
            loc_y = np.argmax(sy, axis=-1) / 2.0 # [21] in [0..256]
            
            score_x = np.max(sx, axis=-1)
            score_y = np.max(sy, axis=-1)
            scores = np.maximum(0.0, np.minimum(score_x, score_y))
            
            # Map back to original image coordinates using inv_trans
            kpts_256 = np.stack([loc_x, loc_y, np.ones(21)], axis=1) # [21, 3]
            kpts_orig = np.dot(kpts_256, inv_trans_list[i].T) # [21, 2]
            
            kpts = np.concatenate([kpts_orig, scores[:, None]], axis=1) # [21, 3]
            results.append({
                'bbox': bboxes[i],
                'kpts': kpts
            })
            
        return results

# 标准 MediaPipe / RTMPose 骨骼拓扑
RTMPOSE_SKELETON = [
    # 拇指
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 食指
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 中指
    (0, 9), (9, 10), (10, 11), (11, 12),
    # 无名指
    (0, 13), (13, 14), (14, 15), (15, 16),
    # 小指
    (0, 17), (17, 18), (18, 19), (19, 20),
    # 掌心横连线
    (5, 9), (9, 13), (13, 17)
]

if __name__ == '__main__':
    det_onnx = '/home/elp/picture_resize_recording_NVIDA/models/rtmpose/rtmdet_nano_hand.onnx'
    pose_onnx = '/home/elp/picture_resize_recording_NVIDA/models/rtmpose/rtmpose_m_hand.onnx'
    detector = RTMPoseHandDetector(det_onnx, pose_onnx)
    
    cap = cv2.VideoCapture('/home/elp/picture_resize_recording_NVIDA/visual_video/noj/30fps_30cm.mp4')
    for f_idx in [180, 260, 300, 340, 480]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        l_raw = frame[:, 160:2080]
        
        bboxes = detector.detect_hands(l_raw, conf_thr=0.25)
        print(f'Frame {f_idx}: detected {len(bboxes)} hands')
        poses = detector.estimate_pose(l_raw, bboxes)
        
        canvas = l_raw.copy()
        for p in poses:
            bb = p['bbox']
            cv2.rectangle(canvas, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 180, 0), 2)
            kpts = p['kpts']
            
            # 画骨骼线条
            for p1, p2 in RTMPOSE_SKELETON:
                x1, y1, c1 = kpts[p1]
                x2, y2, c2 = kpts[p2]
                if c1 > 0.2 and c2 > 0.2:
                    cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2, cv2.LINE_AA)
                    
            # 画关键点
            for i in range(21):
                x, y, c = kpts[i]
                if c > 0.2:
                    cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (int(x), int(y)), 6, (255, 255, 255), 1, cv2.LINE_AA)
                    
        out_path = f'/home/elp/.gemini/antigravity-ide/brain/bf0ba7a5-e6bd-41f2-9316-8fd3f8834376/rtmpose_test_f{f_idx}.jpg'
        cv2.imwrite(out_path, canvas)
        print(f'Saved {out_path}')
    cap.release()
