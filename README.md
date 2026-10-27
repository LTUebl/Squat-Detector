# Squat Biomechanics & Form Analyzer v2.0

**Lawrence Technological University · Department of Biomedical Engineering · Experimental Biomechanics Laboratory (EBL)**

[![Live Demo](https://img.shields.io/badge/Live_Application-LTU_EBL_GitHub_Pages-2563EB?style=for-the-badge&logo=github)](https://ltuebl.github.io/Squat-Detector/)
[![OpenSim Validated](https://img.shields.io/badge/Validated-OpenSim_LaiUhlrich2022_60Hz-059669?style=for-the-badge)](https://ltuebl.github.io/Squat-Detector/#validation-section)
[![LTU EBL Portal](https://img.shields.io/badge/Portal-LTU_EBL_Website-002D62?style=for-the-badge)](https://ltuebl.github.io/EBLWeb/)

An interactive, browser-based markerless squat biomechanics and form diagnostic platform powered by **MediaPipe Pose (2D Pixel + 3D World Landmark Fusion)** and **Plotly.js**, calibrated and validated against **OpenCap & OpenSim musculoskeletal inverse kinematics (`LaiUhlrich2022`, 60 Hz)**.

- **Live Web Application**: [https://ltuebl.github.io/Squat-Detector/](https://ltuebl.github.io/Squat-Detector/)
- **LTU EBL Main Portal**: [https://ltuebl.github.io/EBLWeb/](https://ltuebl.github.io/EBLWeb/)

---

## Key Features

### 1. Bilateral Visibility-Weighted 2D Pixel + 3D World Landmark Fusion Engine
- **Aspect-Ratio Invariant Pixel Geometry**: Converts normalized `[0, 1]` MediaPipe landmarks into true pixel coordinates $(X_i = x_i \cdot W, Y_i = y_i \cdot H)$ prior to trigonometric evaluation, eliminating the $1.78\times$ horizontal distortion observed on vertical $9:16$ smartphone cameras.
- **2D + 3D Foreshortening Compensation**: Dynamically blends bilateral 2D pixel joint angles with MediaPipe's metric 3D `poseWorldLandmarks` when perspective foreshortening is detected (such as in frontal or oblique camera views).
- **5 Real-Time Joint Kinematic Streams** (smoothed with an exponential moving average filter, $\alpha_{\text{EMA}} = 0.35$):
  1. **Knee Flexion ($\theta_{\text{knee}}$)**: Bilateral R/L + fused flexion angle ($\ge 80^\circ$ Full Squat target).
  2. **Hip Flexion ($\theta_{\text{hip}}$)**: Bilateral R/L + fused hip flexion with soft lumbar/deltoid saturation ($\ge 65^\circ$ Full Squat target).
  3. **Sagittal Trunk Lean ($\theta_{\text{trunk}}$)**: Torso inclination from vertical ($\le 40^\circ$ safe athletic range).
  4. **Ankle Dorsiflexion ($\theta_{\text{ankle}}$)**: Tibial inclination from vertical scaled by anatomical ankle-to-tibia ratio ($0.74\times$).
  5. **Frontal-Plane Knee Valgus ($\theta_{\text{valgus}}$)**: Medial knee collapse angle evaluated in frontal/anterior views ($\le 10^\circ$ clinical cutoff).

### 2. Coupled Multi-Joint Squat State Machine & Form Error Detection
- **Coupled Descent Trigger (`STANDING -> DESCENDING`)**: Requires simultaneous knee flexion ($\theta_{\text{knee}} > 35^\circ$) **and** hip flexion ($\theta_{\text{hip}} > 30^\circ$). Pure waist bends / hip hinges ($\theta_{\text{hip}} > 35^\circ$ while $\theta_{\text{knee}} < 25^\circ$) and standing $90^\circ$ pivot turns are automatically rejected with real-time coaching feedback.
- **Repetition Classification**:
  - **Full Squat Rep**: Reaches $\max(\theta_{\text{knee}}) \ge 80^\circ$ and $\max(\theta_{\text{hip}}) \ge 65^\circ$ before returning to standing ($\theta_{\text{knee}} < 25^\circ, \theta_{\text{hip}} < 28^\circ$).
  - **Partial Squat Rep**: Reaches $45^\circ \le \max(\theta_{\text{knee}}) < 80^\circ$ (or $\max(\theta_{\text{hip}}) < 65^\circ$).
- **Biomechanical Form Error Diagnostics**:
  - **Shallow Depth**: Partial squat ($45^\circ \le \max \theta_{\text{knee}} < 80^\circ$).
  - **Excessive Trunk Lean**: Forward torso inclination $> 40^\circ$ while squatting ($\theta_{\text{knee}} > 45^\circ$).
  - **Dynamic Knee Valgus**: Medial frontal-plane knee collapse $> 10^\circ$ while squatting.

### 3. Smartphone & Cross-Browser Compatibility (iOS Safari, Android Chrome, Desktop)
- Direct `navigator.mediaDevices.getUserMedia` camera controller supporting **Front (Selfie) / Rear (Environment) Flip**, camera device selection, and horizontal selfie mirroring with legible un-mirrored canvas text callouts.
- Cross-browser `MediaRecorder` MIME negotiation (`video/mp4`, `video/webm;codecs=vp9`, `video/webm;codecs=vp8`, `video/webm`).
- Built-in **Validated Sample Squat Videos** and **Custom Video Upload (`.mp4`, `.mov`, `.webm`)**.

### 4. Post-Recording Video Replay & Kinematic Angle Analysis Studio
- **Synchronized Video & Skeleton Replay**: Frame-by-frame video player (`-1f` / `+1f` step buttons, `0.25x` / `0.5x` / `1.0x` playback speeds, interactive timeline scrubber) with toggleable 2D pose skeleton overlay.
- **Interactive Plotly.js Joint Angle Trajectories**: Plots continuous Knee, Hip, Trunk, Ankle, and Valgus angles vs. Time (s) with shaded repetition windows, threshold lines, and a synchronized vertical playhead cursor (click anywhere on the graph to seek the video).
- **OpenSim Ground-Truth Overlay**: Toggleable overlay of the 60 Hz OpenSim `Squat.mot` inverse kinematics reference curves (`knee_angle_r`, `hip_flexion_r`, `ankle_angle_r`, `pelvis_tilt + lumbar_extension`).
- **One-Click CSV & Video Export**: Download the recorded video and export frame-by-frame joint kinematics to CSV.

---

## OpenSim Ground-Truth Validation Summary

| Trial & Camera View | Metric | OpenSim Ground Truth (`Squat.mot`) | Original 2D Prototype | Updated v2.0 Engine (2D+3D Fusion) |
| :--- | :--- | :--- | :--- | :--- |
| **`Cam0` (`sample_squat_sagittal.mp4`)**<br>360×640 Portrait · 60 fps · Oblique (~41°) | **Full / Partial / Errors**<br>**Peak Knee Flexion**<br>**Peak Hip Flexion**<br>**Peak Ankle Dorsiflexion**<br>**Bottom Timing ($t_{\min}$)** | **1 / 0 / 0**<br>**123.6° (R) / 122.7° (L)**<br>**93.3° (R) / 90.1° (L)**<br>**41.1° (R) / 34.2° (L)**<br>**1.32 s** | 1 / 0 / 0 *(stuck in `down`)*<br>134.5° *(+10.9° error)*<br>99.1° *(+7.4° vs mean)*<br>*Not computed*<br>1.35 s | **1 / 0 / 0** (`EndState: up`)<br>**121.4°** *(-1.8° vs mean)*<br>**87.7°** *(-2.4° vs L)*<br>**40.8°** *(-0.3° vs R)*<br>**1.32 s** *(0.00 s error)* |
| **`Cam1` (`sample_squat_frontal.mp4`)**<br>360×640 Portrait · 60 fps · Frontal (~15°) | **Full / Partial / Errors**<br>**Peak Knee Flexion**<br>**Peak Hip Flexion**<br>**Trunk Lean (Bottom)** | **1 / 0 / 0**<br>**123.6° (R) / 122.7° (L)**<br>**93.3° (R) / 90.1° (L)**<br>**24.4°** | **0 / 0 / 0** *(0% Recall!)*<br>97.3° *(-26.3° error)*<br>80.1° *(-13.2° error)*<br>34.6° *(False Error!)* | **1 / 0 / 0** *(100% Recall)*<br>**121.6°** *(-2.0° vs R)*<br>**88.1°** *(-2.0° vs L)*<br>**21.9°** *(0 False Errors)* |
| **`Sample 2` (`sample_squat_multiview.mp4`)**<br>342×562 Portrait · 30 fps · 4 Orientations | **Full / Partial / Errors**<br>**4-Rep Peak Knee Angles** | **4 / 0 / 0** (4 Healthy Squats)<br>**~128°–136°** | 4 / 0 / **4** *(100% False Errors)*<br>Hip inflated to 148.6° | **4 / 0 / 0** *(0% False Errors)*<br>**130.1°, 136.4°, 128.4°, 130.3°** |

---

## Repository Structure

```text
Squat-Detector/
├── index.html                      # Main application UI & Post-Recording Analysis Studio
├── styles.css                      # Responsive desktop/smartphone layout & viewport styles
├── app.js                          # 2D+3D Biomechanical Fusion Engine, State Machine & Plotly Studio
├── EBLlogo.png                     # Official LTU Experimental Biomechanics Lab logo
├── data/
│   └── opensim_ground_truth.json   # 60 Hz OpenSim (LaiUhlrich2022) Squat.mot reference kinematics
└── videos/
    ├── sample_squat_sagittal.mp4   # Sample 1: Deep Squat (Cam0 Oblique, synchronized with OpenSim GT)
    ├── sample_squat_frontal.mp4    # Sample 1B: Deep Squat (Cam1 Frontal, synchronized with OpenSim GT)
    └── sample_squat_multiview.mp4  # Sample 2: 4-Rep Multi-View Squat across 4 camera perspectives
```

---

## License & Attribution

Developed by the **Experimental Biomechanics Laboratory (EBL)** in the **Department of Biomedical Engineering** at **Lawrence Technological University** for biomechanics research and undergraduate/graduate kinesiology education.
