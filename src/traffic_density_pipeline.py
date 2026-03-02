#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import time
import csv
from datetime import datetime
import warnings

# Suppress torch amp future warnings
warnings.filterwarnings(
    "ignore",
    message=".*torch\\.cuda\\.amp\\.autocast.*",
    category=FutureWarning,
)


def sample_frames(video_path: str, frames_dir: str, target_size=None):
    """
    Extracts every frame from video_path into frames_dir, optionally resizing.
    """
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        fname = f"frame_{idx:04d}.jpg"
        cv2.imwrite(os.path.join(frames_dir, fname), frame)
        idx += 1
    cap.release()
    print(f"Extracted {idx} frames -> {frames_dir}")


def run_inference(frames_dir: str, out_dir: str, model, device: str = 'cpu', roi=None, motion_thresh: float = 0.05) -> str:
    """
    Runs YOLO inference with motion filtering and saves annotated frames and detections.
    """
    txt_dir = os.path.join(out_dir, 'detections')
    img_dir = os.path.join(out_dir, 'annotated')
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    if roi:
        x1, y1, x2, y2 = roi
    else:
        x1 = y1 = x2 = y2 = None

    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    prev_gray = None
    densities = []
    names = model.names
    start_inf = time.time()

    for f in files:
        frame = cv2.imread(os.path.join(frames_dir, f))
        annotated = frame.copy()

        if roi:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model(frame)
        filtered = []

        for *box, conf, cls in results.xyxy[0].tolist():
            bx1, by1, bx2, by2 = map(int, box)
            cx, cy = (bx1+bx2)//2, (by1+by2)//2
            if roi and not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            if prev_gray is not None:
                patch_curr = gray[by1:by2, bx1:bx2]
                patch_prev = prev_gray[by1:by2, bx1:bx2]
                if patch_curr.size and patch_prev.size:
                    diff = cv2.absdiff(patch_curr, patch_prev)
                    _, bin_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    ratio = cv2.countNonZero(bin_mask) / bin_mask.size
                    if ratio < motion_thresh:
                        continue
            filtered.append((bx1, by1, bx2, by2, conf, int(cls)))

        prev_gray = gray
        count = len(filtered)
        densities.append(count)

        if count <= 5:
            zone, bgcolor = 'Light Traffic', (0, 255, 0)
        elif count <= 14:
            zone, bgcolor = 'Moderate Traffic', (0, 165, 255)
        else:
            zone, bgcolor = 'Heavy Traffic', (0, 0, 255)

        txt_file = os.path.join(txt_dir, f.replace('.jpg', '.txt'))
        with open(txt_file, 'w') as tf:
            for bx1, by1, bx2, by2, conf, cls in filtered:
                tf.write(f"{cls} {conf:.3f} {bx1} {by1} {bx2} {by2}\n")

        for bx1, by1, bx2, by2, conf, cls in filtered:
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), bgcolor, 2)
            lbl = f"{names[cls]} {conf:.2f}"
            tw, th = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(annotated, (bx1, by1-th-4), (bx1+tw, by1), bgcolor, -1)
            cv2.putText(annotated, lbl, (bx1, by1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        txt1 = f"Count: {count}"
        w1, h1 = cv2.getTextSize(txt1, cv2.FONT_HERSHEY_SIMPLEX,1,2)[0]
        cv2.rectangle(annotated, (10,10),(10+w1+10,10+h1+10), bgcolor,-1)
        cv2.putText(annotated, txt1,(15,10+h1+2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        w2,h2 = cv2.getTextSize(zone, cv2.FONT_HERSHEY_SIMPLEX,1,2)[0]
        y0 = 20+h1+10
        cv2.rectangle(annotated,(10,y0),(10+w2+10,y0+h2+4), bgcolor,-1)
        cv2.putText(annotated, zone,(15,y0+h2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

        cv2.imwrite(os.path.join(img_dir,f), annotated)

    elapsed_inf = time.time() - start_inf
    print(f"Inference on {len(files)} frames in {elapsed_inf:.1f}s ({len(files)/elapsed_inf:.1f} FPS)")

    csv_file = os.path.join(out_dir,'density_zones.csv')
    with open(csv_file,'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['frame','density','zone'])
        for i,d in enumerate(densities):
            z = 'Light Traffic' if d<=5 else 'Moderate Traffic' if d<=14 else 'Heavy Traffic'
            writer.writerow([i,d,z])

    print(f"Average density: {sum(densities)/len(densities):.2f}")
    return img_dir


def frames_to_video(frames_dir: str, output_path: str, fps: int=1):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    if not files:
        raise IOError(f"No frames in {frames_dir}")
    h,w = cv2.imread(os.path.join(frames_dir,files[0])).shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,(w,h))
    for f in files:
        out.write(cv2.imread(os.path.join(frames_dir,f)))
    out.release()
    print(f"Saved video: {output_path}")


def main():
    # overall runtime start
    start_all = time.time()

    parser = argparse.ArgumentParser(description='YOLOv5-nano Traffic Density w/o Subsampling')
    parser.add_argument('--video', type=str, default='traffic1_720p.mp4')
    parser.add_argument('--resize', type=str, help='WxH, e.g. 1280x720')
    parser.add_argument('--device', choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--roi', type=str, help='x1,y1,x2,y2')
    parser.add_argument('--motion_thresh', type=float, default=0.05)
    args = parser.parse_args()

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.abspath(os.path.join('runs','motion',run_id))
    frames_dir = os.path.join(base_dir,'frames')
    os.makedirs(base_dir, exist_ok=True)

    roi = tuple(map(int,args.roi.split(','))) if args.roi else None
    target_size = tuple(map(int,args.resize.split('x'))) if args.resize else None

    torch.set_num_threads(args.threads)
    print(f"Run:{run_id}|Video:{args.video}|Resize:{target_size}|ROI:{roi}|MotionTh:{args.motion_thresh}")

    sample_frames(args.video, frames_dir, target_size)
    model = torch.hub.load('ultralytics/yolov5','yolov5n',pretrained=True)
    model.conf = args.conf
    model.to(args.device).eval()

    annotated_dir = run_inference(frames_dir, base_dir, model, args.device, roi, args.motion_thresh)
    out_video = os.path.join(base_dir,'output.mp4')
    frames_to_video(annotated_dir, out_video)

    total_time = time.time() - start_all
    print(f"Total pipeline time: {total_time:.1f}s")

if __name__=='__main__':
    main()