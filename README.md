# Lightweight Traffic Density Estimation (YOLO + Subsampling)

CPU-friendly traffic density estimation from existing roadside cameras using:
- YOLO object detection
- aggressive frame subsampling (process 1 in every 10 frames)
- ROI cropping (focus on relevant lanes)
- motion-based filtering to exclude parked vehicles

## Why this matters
Many traffic analytics pipelines are too expensive to run in real-time without GPUs.  
This project demonstrates a deployable approach that remains responsive on low-power machines.

## Core pipeline
1) **Preprocess**: resize to 720p (optional), ROI crop  
2) **Subsample**: 1 fps (1/10 frames)  
3) **Detect vehicles** with YOLO  
4) **Motion filter**: remove stationary detections (parked cars)  
5) **Density scoring**:
   - Light: ≤ 5 vehicles/frame
   - Moderate: 6–14
   - Heavy: ≥ 15

## Results (high level)
- >90% compute reduction via subsampling (compared to full-frame processing)
- Robust behavior across video resolutions (360p to 4K)
- Reduced false positives via ROI + motion filtering

## Repository structure
- `src/` — pipeline code
- `configs/` — ROI & thresholds
- `data_samples/` — tiny demo clips only
- `assets/` — README visuals (GIFs/screens)
- `docs/` — report

## Notes on data
Only small demo clips are included. Full videos are referenced externally if needed (license permitting).

## Demo

### Raw input
![Raw Traffic](assets/parked_cars_raw.png)

### Parked Cars classifiyng as traffic vehicles
![Output Demo](assets/parked_cars_issues.gif)

### Parked cars excluded
![Output_Demo](assets/parked_cars_issue_fixed.png)
