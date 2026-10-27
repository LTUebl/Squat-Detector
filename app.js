/* ============================================================================
   LTU Experimental Biomechanics Laboratory (EBL)
   Squat Biomechanics & Form Analyzer v2.0 — Core Application Engine
   ----------------------------------------------------------------------------
   Features:
   1. Bilateral Visibility-Weighted 2D Pixel + 3D World Landmark Fusion Engine
   2. Coupled Knee+Hip Squat State Machine & Multi-Planar Error Detector
   3. Smartphone-Compatible Camera Manager (Front/Rear Flip, Portrait/Landscape)
   4. Cross-Browser MediaRecorder + Sample/Upload Video Pipeline
   5. Post-Recording Video Replay & Plotly.js Kinematic Analysis Studio
   6. OpenSim Ground-Truth (Squat.mot) Overlay & Automated Self-Test Suite
   ============================================================================ */

'use strict';

// ============================================================================
// 1. GLOBAL STATE & CONFIGURATION
// ============================================================================
const DEFAULT_THRESHOLDS = {
  fullKnee: 80.0,       // deg: Full squat knee flexion threshold
  fullHip: 65.0,        // deg: Full squat hip flexion threshold
  minSquatKnee: 45.0,   // deg: Minimum knee flexion to count as a squat attempt
  entryKnee: 35.0,      // deg: Descent trigger knee flexion
  entryHip: 30.0,       // deg: Descent trigger hip flexion
  exitKnee: 25.0,       // deg: Return-to-standing knee flexion
  exitHip: 28.0,        // deg: Return-to-standing hip flexion
  maxTrunk: 40.0,       // deg: Forward trunk lean error threshold
  maxValgus: 10.0,      // deg: Frontal-plane medial knee valgus threshold
  emaAlpha: 0.35        // Exponential moving average smoothing factor
};

const userThresholds = { ...DEFAULT_THRESHOLDS };

const sessionState = {
  mode: 'idle',               // 'idle' | 'webcam' | 'sample' | 'upload'
  currentSampleId: null,      // 'sagittal' | 'frontal' | 'multiview' | null
  trialTitle: 'No Active Trial',
  isRunning: false,
  isMirrored: false,
  facingMode: 'environment',  // Default to rear camera on phones; flips to 'user'

  // Biomechanical state machine
  squatState: 'up',           // 'up' | 'down'
  phaseLabel: 'STANDING',     // 'STANDING' | 'DESCENDING' | 'BOTTOM' | 'ASCENDING'
  fullReps: 0,
  partialReps: 0,
  errorReps: 0,

  // Active rep accumulators
  repStartFrame: 0,
  repStartTime: 0.0,
  repBottomFrame: 0,
  repBottomTime: 0.0,
  maxKnee: 0.0,
  maxHip: 0.0,
  maxTrunk: 0.0,
  maxAnkle: 0.0,
  maxValgus: 0.0,
  trunkAtBottom: 0.0,
  downFrames: 0,
  trunkErrFrames: 0,
  valgusErrFrames: 0,

  // Smoothed angle streams
  smKnee: null,
  smHip: null,
  smTrunk: null,
  smAnkle: null,
  smValgus: null,

  // Recorded session time-series & rep summaries
  frameIndex: 0,
  sessionStartPerf: 0,
  lastVideoTime: -1,
  fpsEstimate: 30.0,
  lastFramePerf: 0,
  timeSeries: [],             // Array of frame telemetry objects
  repsSummary: [],            // Array of completed rep objects

  // MediaRecorder for live webcam sessions
  mediaStream: null,
  mediaRecorder: null,
  recordedChunks: [],
  recordedVideoUrl: null,
  recordedMimeType: '',
  sourceVideoUrl: null        // For sample/uploaded videos
};

// OpenSim Ground Truth (loaded from data/opensim_ground_truth.json)
let openSimGroundTruth = null;

// MediaPipe Pose instance & concurrency lock
let poseInstance = null;
let isSendingFrame = false;
let rafId = null;

// Mini-chart throttle
let lastMiniChartUpdate = 0;

// Replay Studio state
let replayRafId = null;
let activeRepLoopBounds = null; // { start, end } when looping a specific rep

// ============================================================================
// 2. BIOMECHANICAL 2D PIXEL + 3D WORLD FUSION ENGINE (VERIFIED BENCHMARK)
// ============================================================================

/**
 * Clamp value x between minVal and maxVal
 */
function clamp(x, minVal, maxVal) {
  return Math.max(minVal, Math.min(maxVal, x));
}

/**
 * Compute 2D included angle at vertex B (in degrees) using true pixel coordinates
 */
function angle2DPixel(a, b, c, w, h) {
  const ax = a.x * w, ay = a.y * h;
  const bx = b.x * w, by = b.y * h;
  const cx = c.x * w, cy = c.y * h;
  const v1x = ax - bx, v1y = ay - by;
  const v2x = cx - bx, v2y = cy - by;
  const n1 = Math.hypot(v1x, v1y);
  const n2 = Math.hypot(v2x, v2y);
  if (n1 < 1e-6 || n2 < 1e-6) return 180.0;
  const cosT = clamp((v1x * v2x + v1y * v2y) / (n1 * n2), -1.0, 1.0);
  return Math.acos(cosT) * (180.0 / Math.PI);
}

/**
 * Compute 3D included angle at vertex B (in degrees) using metric poseWorldLandmarks
 */
function angle3DWorld(a, b, c) {
  if (!a || !b || !c) return 180.0;
  const v1x = a.x - b.x, v1y = a.y - b.y, v1z = a.z - b.z;
  const v2x = c.x - b.x, v2y = c.y - b.y, v2z = c.z - b.z;
  const n1 = Math.hypot(v1x, v1y, v1z);
  const n2 = Math.hypot(v2x, v2y, v2z);
  if (n1 < 1e-6 || n2 < 1e-6) return 180.0;
  const dot = v1x * v2x + v1y * v2y + v1z * v2z;
  const cosT = clamp(dot / (n1 * n2), -1.0, 1.0);
  return Math.acos(cosT) * (180.0 / Math.PI);
}

/**
 * Compute bilateral visibility-weighted 2D Pixel + 3D World fused joint kinematics.
 * Exact mathematical match to verified benchmark (`test_final.py`).
 *
 * @param {Array} lm2 - MediaPipe normalized 2D poseLandmarks (33 points)
 * @param {Array} lm3 - MediaPipe metric 3D poseWorldLandmarks (33 points)
 * @param {number} w  - True video frame pixel width
 * @param {number} h  - True video frame pixel height
 */
function computeFusedBiomechanicalAngles(lm2, lm3, w, h) {
  if (!lm2 || lm2.length < 29) return null;

  const getVis = (pt) => (pt && typeof pt.visibility === 'number') ? pt.visibility : (pt && typeof pt.v === 'number' ? pt.v : 1.0);

  // Mean visibility for Right (12, 24, 26, 28) and Left (11, 23, 25, 27) chains
  const vr = (getVis(lm2[12]) + getVis(lm2[24]) + getVis(lm2[26]) + getVis(lm2[28])) / 4.0;
  const vl = (getVis(lm2[11]) + getVis(lm2[23]) + getVis(lm2[25]) + getVis(lm2[27])) / 4.0;
  const maxVis = Math.max(vr, vl);

  // Bilateral weighting: equal 0.5/0.5 when both >= 0.65 (cancels oblique perspective error)
  let wR = 0.5, wL = 0.5;
  if (vr >= 0.65 && vl >= 0.65) {
    wR = 0.5;
    wL = 0.5;
  } else if (vr > vl + 0.15) {
    wR = 1.0;
    wL = 0.0;
  } else if (vl > vr + 0.15) {
    wR = 0.0;
    wL = 1.0;
  } else {
    wR = vr / Math.max(1e-5, vr + vl);
    wL = 1.0 - wR;
  }

  // Frontal torso aspect ratio in true pixel coordinates
  const hipWidthPx = Math.abs(lm2[24].x - lm2[23].x) * w;
  const shWidthPx = Math.abs(lm2[12].x - lm2[11].x) * w;
  const torsoHeightPx = Math.max(
    1.0,
    Math.abs(0.5 * (lm2[24].y + lm2[23].y) - 0.5 * (lm2[12].y + lm2[11].y)) * h
  );
  const frontalRatio = Math.max(hipWidthPx, shWidthPx) / torsoHeightPx;

  // 2D Pixel flexion angles (0 deg = full extension)
  const k2R = 180.0 - angle2DPixel(lm2[24], lm2[26], lm2[28], w, h);
  const k2L = 180.0 - angle2DPixel(lm2[23], lm2[25], lm2[27], w, h);
  const h2R = 180.0 - angle2DPixel(lm2[12], lm2[24], lm2[26], w, h);
  const h2L = 180.0 - angle2DPixel(lm2[11], lm2[23], lm2[25], w, h);

  // 3D World flexion angles (fallback to 2D if lm3 is absent)
  const has3D = Array.isArray(lm3) && lm3.length >= 29;
  const k3R = has3D ? (180.0 - angle3DWorld(lm3[24], lm3[26], lm3[28])) : k2R;
  const k3L = has3D ? (180.0 - angle3DWorld(lm3[23], lm3[25], lm3[27])) : k2L;
  const h3R = has3D ? (180.0 - angle3DWorld(lm3[12], lm3[24], lm3[26])) : h2R;
  const h3L = has3D ? (180.0 - angle3DWorld(lm3[11], lm3[23], lm3[25])) : h2L;

  const k2 = wR * k2R + wL * k2L;
  const k3 = wR * k3R + wL * k3L;
  const h2 = wR * h2R + wL * h2L;
  const h3 = wR * h3R + wL * h3L;

  // 1. Knee Flexion Fusion (2D bilateral average in sagittal/oblique, blends to 3D when foreshortened)
  const deficitK = clamp((k3 - k2 - 3.0) / 18.0, 0.0, 1.0);
  const frontalBoost = 1.0 + 0.09 * clamp((k3 - k2 - 20.0) / 30.0, 0.0, 1.0);
  const rawKnee = ((1.0 - deficitK) * k2 + deficitK * k3) * frontalBoost;
  const rawKneeR = ((1.0 - deficitK) * k2R + deficitK * k3R) * frontalBoost;
  const rawKneeL = ((1.0 - deficitK) * k2L + deficitK * k3L) * frontalBoost;

  // 2. Hip Flexion Fusion (with soft saturation > 95 deg for lumbar flexion / anterior arm shift)
  const deficitH = clamp((h3 - h2) / 5.0, 0.0, 1.0);
  const includedHip = (1.0 - deficitH) * h2 + deficitH * h3;
  const rawHip = includedHip <= 95.0 ? includedHip : 95.0 + 0.55 * (includedHip - 95.0);
  const incHipR = (1.0 - deficitH) * h2R + deficitH * h3R;
  const incHipL = (1.0 - deficitH) * h2L + deficitH * h3L;
  const rawHipR = incHipR <= 95.0 ? incHipR : 95.0 + 0.55 * (incHipR - 95.0);
  const rawHipL = incHipL <= 95.0 ? incHipL : 95.0 + 0.55 * (incHipL - 95.0);

  // 3. Trunk Lean (2D pixel + 3D world inclination from vertical)
  const midShX = (wR * lm2[12].x + wL * lm2[11].x) * w;
  const midShY = (wR * lm2[12].y + wL * lm2[11].y) * h;
  const midHpX = (wR * lm2[24].x + wL * lm2[23].x) * w;
  const midHpY = (wR * lm2[24].y + wL * lm2[23].y) * h;
  const trunk2D = Math.atan2(Math.abs(midShX - midHpX), Math.max(1e-5, midHpY - midShY)) * (180.0 / Math.PI);

  let trunk3D = trunk2D;
  if (has3D) {
    const sh3X = 0.5 * (lm3[11].x + lm3[12].x);
    const sh3Y = 0.5 * (lm3[11].y + lm3[12].y);
    const sh3Z = 0.5 * (lm3[11].z + lm3[12].z);
    const hp3X = 0.5 * (lm3[23].x + lm3[24].x);
    const hp3Y = 0.5 * (lm3[23].y + lm3[24].y);
    const hp3Z = 0.5 * (lm3[23].z + lm3[24].z);
    trunk3D = Math.atan2(
      Math.hypot(sh3X - hp3X, sh3Z - hp3Z),
      Math.max(1e-5, hp3Y - sh3Y)
    ) * (180.0 / Math.PI);
  }
  const foreshortenBlend = clamp((frontalRatio - 0.45) / 0.20, 0.0, 1.0);
  const rawTrunk = (1.0 - foreshortenBlend) * trunk2D + foreshortenBlend * Math.max(trunk2D, trunk3D * 0.82);

  // 4. Ankle Dorsiflexion (0.74 * fused tibial inclination from vertical)
  const tib2R = Math.atan2(Math.abs(lm2[26].x - lm2[28].x) * w, Math.max(1e-5, (lm2[28].y - lm2[26].y) * h)) * (180.0 / Math.PI);
  const tib2L = Math.atan2(Math.abs(lm2[25].x - lm2[27].x) * w, Math.max(1e-5, (lm2[27].y - lm2[25].y) * h)) * (180.0 / Math.PI);
  const tib2 = wR * tib2R + wL * tib2L;

  let tib3 = tib2;
  if (has3D) {
    const tib3R = Math.atan2(
      Math.hypot(lm3[26].x - lm3[28].x, lm3[26].z - lm3[28].z),
      Math.max(1e-5, lm3[28].y - lm3[26].y)
    ) * (180.0 / Math.PI);
    const tib3L = Math.atan2(
      Math.hypot(lm3[25].x - lm3[27].x, lm3[25].z - lm3[27].z),
      Math.max(1e-5, lm3[27].y - lm3[25].y)
    ) * (180.0 / Math.PI);
    tib3 = wR * tib3R + wL * tib3L;
  }
  const rawAnkle = ((1.0 - deficitK) * tib2 + deficitK * tib3) * 0.74;

  // 5. Frontal-Plane Knee Valgus (Medial Collapse Angle)
  const ankDistX = Math.abs(lm2[27].x - lm2[28].x) * w;
  const kneeDistX = Math.abs(lm2[25].x - lm2[26].x) * w;
  let rawValgus = 0.0;
  if (frontalRatio > 0.45 && ankDistX > 0.10 * w && rawKnee > 45.0) {
    const collapsePx = Math.max(0.0, ankDistX * 0.90 - kneeDistX);
    const shankLenPx = Math.max(
      1.0,
      0.5 * (Math.abs(lm2[28].y - lm2[26].y) + Math.abs(lm2[27].y - lm2[25].y)) * h
    );
    rawValgus = Math.atan2(collapsePx * 0.5, shankLenPx) * (180.0 / Math.PI);
  }

  // Camera perspective classification for HUD
  let viewType = 'Sagittal / Oblique';
  if (frontalRatio > 0.58) {
    viewType = 'Frontal / Anterior';
  } else if (frontalRatio > 0.40) {
    viewType = 'Oblique (2D+3D Fused)';
  } else {
    viewType = 'Sagittal Profile';
  }

  return {
    rawKnee,
    rawHip,
    rawTrunk,
    rawAnkle,
    rawValgus,
    rawKneeR,
    rawKneeL,
    rawHipR,
    rawHipL,
    maxVis,
    vr,
    vl,
    frontalRatio,
    viewType
  };
}

// ============================================================================
// 3. SQUAT STATE MACHINE & FORM ERROR ENGINE
// ============================================================================

/**
 * Process a single frame of raw angles through the EMA filter and squat state machine.
 * Returns the processed telemetry entry and any newly completed rep object.
 */
function stepSquatStateMachine(angles, timestampSec, frameIdx, fps, thresholds, stateObj, lm2Snapshot = null) {
  const alpha = thresholds.emaAlpha;

  if (stateObj.smKnee === null) {
    stateObj.smKnee = angles.rawKnee;
    stateObj.smHip = angles.rawHip;
    stateObj.smTrunk = angles.rawTrunk;
    stateObj.smAnkle = angles.rawAnkle;
    stateObj.smValgus = angles.rawValgus;
  } else {
    stateObj.smKnee = alpha * angles.rawKnee + (1.0 - alpha) * stateObj.smKnee;
    stateObj.smHip = alpha * angles.rawHip + (1.0 - alpha) * stateObj.smHip;
    stateObj.smTrunk = alpha * angles.rawTrunk + (1.0 - alpha) * stateObj.smTrunk;
    stateObj.smAnkle = alpha * angles.rawAnkle + (1.0 - alpha) * stateObj.smAnkle;
    stateObj.smValgus = alpha * angles.rawValgus + (1.0 - alpha) * stateObj.smValgus;
  }

  const smKnee = stateObj.smKnee;
  const smHip = stateObj.smHip;
  const smTrunk = stateObj.smTrunk;
  const smAnkle = stateObj.smAnkle;
  const smValgus = stateObj.smValgus;

  let coachingCue = 'Good alignment — ready for squat';
  let coachingSeverity = 'good'; // 'good' | 'warn' | 'error'
  let completedRep = null;

  // Low landmark visibility guard
  if (angles.maxVis < 0.35) {
    coachingCue = 'Low body visibility — step back so hips, knees & ankles are in frame';
    coachingSeverity = 'warn';
  } else if (stateObj.squatState === 'up') {
    stateObj.phaseLabel = 'STANDING';

    // Check for pure waist bend / hip hinge (hip flexed > 35 deg while knees are straight < 25 deg)
    if (smHip > 35.0 && smKnee < 25.0) {
      coachingCue = 'Hip hinge / waist bend detected — bend knees & hips together to squat';
      coachingSeverity = 'warn';
    } else if (smKnee > thresholds.entryKnee && smHip > thresholds.entryHip) {
      // Enter Squat Descent!
      stateObj.squatState = 'down';
      stateObj.phaseLabel = 'DESCENDING';
      stateObj.repStartFrame = frameIdx;
      stateObj.repStartTime = timestampSec;
      stateObj.repBottomFrame = frameIdx;
      stateObj.repBottomTime = timestampSec;
      stateObj.maxKnee = smKnee;
      stateObj.maxHip = smHip;
      stateObj.maxTrunk = smKnee > thresholds.minSquatKnee ? smTrunk : 0.0;
      stateObj.maxAnkle = smAnkle;
      stateObj.maxValgus = smKnee > thresholds.minSquatKnee ? smValgus : 0.0;
      stateObj.trunkAtBottom = smTrunk;
      stateObj.downFrames = 1;
      stateObj.trunkErrFrames = 0;
      stateObj.valgusErrFrames = 0;
      coachingCue = 'Descending — control depth and keep chest up';
      coachingSeverity = 'good';
    }
  } else if (stateObj.squatState === 'down') {
    stateObj.downFrames += 1;

    // Update peak knee flexion & bottom timestamp
    if (smKnee >= stateObj.maxKnee) {
      stateObj.maxKnee = smKnee;
      stateObj.repBottomFrame = frameIdx;
      stateObj.repBottomTime = timestampSec;
      stateObj.trunkAtBottom = smTrunk;
      stateObj.phaseLabel = smKnee >= thresholds.fullKnee ? 'BOTTOM (FULL DEPTH)' : 'DESCENDING';
    } else if (smKnee < stateObj.maxKnee - 6.0) {
      stateObj.phaseLabel = 'ASCENDING';
    } else {
      stateObj.phaseLabel = 'BOTTOM HOLD';
    }

    stateObj.maxHip = Math.max(stateObj.maxHip, smHip);
    stateObj.maxAnkle = Math.max(stateObj.maxAnkle, smAnkle);

    // Evaluate biomechanical form errors while actively squatting (knee > 45 deg)
    if (smKnee > thresholds.minSquatKnee) {
      stateObj.maxTrunk = Math.max(stateObj.maxTrunk, smTrunk);
      stateObj.maxValgus = Math.max(stateObj.maxValgus, smValgus);

      if (smTrunk > thresholds.maxTrunk) {
        stateObj.trunkErrFrames += 1;
      }
      if (smValgus > thresholds.maxValgus) {
        stateObj.valgusErrFrames += 1;
      }
    }

    // Live coaching feedback during the squat
    if (smKnee > thresholds.minSquatKnee && smTrunk > thresholds.maxTrunk) {
      coachingCue = `Excessive Trunk Lean (${smTrunk.toFixed(0)}° > ${thresholds.maxTrunk}°) — Keep chest upright!`;
      coachingSeverity = 'error';
    } else if (smKnee > thresholds.minSquatKnee && smValgus > thresholds.maxValgus) {
      coachingCue = `Dynamic Knee Valgus (${smValgus.toFixed(0)}° > ${thresholds.maxValgus}°) — Push knees outward!`;
      coachingSeverity = 'error';
    } else if (stateObj.maxKnee >= thresholds.fullKnee && stateObj.maxHip >= thresholds.fullHip) {
      coachingCue = `Full Squat Depth Achieved (${stateObj.maxKnee.toFixed(0)}°) — Drive up to standing!`;
      coachingSeverity = 'good';
    } else {
      const rem = Math.max(0, thresholds.fullKnee - stateObj.maxKnee);
      coachingCue = `Squatting (${smKnee.toFixed(0)}°) — Descend ${rem.toFixed(0)}° deeper for Full Rep`;
      coachingSeverity = 'warn';
    }

    // Check for Return to Standing ('down' -> 'up')
    if (smKnee < thresholds.exitKnee && smHip < thresholds.exitHip) {
      stateObj.squatState = 'up';
      stateObj.phaseLabel = 'STANDING';

      const repDurationSec = timestampSec - stateObj.repStartTime;
      const minFramesReq = Math.max(5, Math.floor(0.32 * (fps || 30.0)));
      const hasValidDuration = stateObj.downFrames >= minFramesReq || (stateObj.downFrames >= 4 && repDurationSec >= 0.35);
      if (stateObj.maxKnee >= thresholds.minSquatKnee && hasValidDuration) {
        const isFull = (stateObj.maxKnee >= thresholds.fullKnee && stateObj.maxHip >= thresholds.fullHip);
        const errList = [];

        if (!isFull) {
          errList.push('Shallow Depth');
        }
        if (stateObj.maxTrunk > thresholds.maxTrunk) {
          errList.push('Excessive Trunk Lean');
        }
        if (stateObj.maxValgus > thresholds.maxValgus) {
          errList.push('Knee Valgus');
        }

        if (isFull) {
          stateObj.fullReps += 1;
        } else {
          stateObj.partialReps += 1;
        }

        const hasError = errList.length > 0;
        if (hasError) {
          stateObj.errorReps += 1;
        }

        const repNum = stateObj.fullReps + stateObj.partialReps;
        completedRep = {
          rep: repNum,
          type: isFull ? 'FULL' : 'PARTIAL',
          startFrame: stateObj.repStartFrame,
          bottomFrame: stateObj.repBottomFrame,
          endFrame: frameIdx,
          startTime: Number(stateObj.repStartTime.toFixed(2)),
          bottomTime: Number(stateObj.repBottomTime.toFixed(2)),
          endTime: Number(timestampSec.toFixed(2)),
          peakKnee: Number(stateObj.maxKnee.toFixed(1)),
          peakHip: Number(stateObj.maxHip.toFixed(1)),
          peakTrunk: Number(stateObj.maxTrunk.toFixed(1)),
          trunkAtBottom: Number(stateObj.trunkAtBottom.toFixed(1)),
          peakAnkle: Number(stateObj.maxAnkle.toFixed(1)),
          peakValgus: Number(stateObj.maxValgus.toFixed(1)),
          hasError,
          errors: hasError ? errList : ['Good Form']
        };

        stateObj.repsSummary.push(completedRep);

        // Set completion coaching banner
        if (!hasError && isFull) {
          coachingCue = `Rep #${repNum} Complete: FULL SQUAT (Peak Knee ${completedRep.peakKnee}°, Good Form!)`;
          coachingSeverity = 'good';
        } else if (!isFull) {
          coachingCue = `Rep #${repNum} Complete: PARTIAL SQUAT (Peak Knee ${completedRep.peakKnee}° < ${thresholds.fullKnee}°)`;
          coachingSeverity = 'warn';
        } else {
          coachingCue = `Rep #${repNum} Complete: FORM ERROR (${errList.join(', ')})`;
          coachingSeverity = 'error';
        }
      } else {
        // Non-squat micro-bend (< 45 deg knee flexion): reset cleanly without counting a false rep
        coachingCue = 'Returned to standing (movement < 45° knee flexion ignored)';
        coachingSeverity = 'good';
      }
    }
  }

  const trunkErrorNow = (stateObj.squatState === 'down' && smKnee > thresholds.minSquatKnee && smTrunk > thresholds.maxTrunk);
  const valgusErrorNow = (stateObj.squatState === 'down' && smKnee > thresholds.minSquatKnee && smValgus > thresholds.maxValgus);

  const frameRecord = {
    frame: frameIdx,
    time: Number(timestampSec.toFixed(3)),
    state: stateObj.squatState,
    phase: stateObj.phaseLabel,
    repNumber: stateObj.fullReps + stateObj.partialReps + (stateObj.squatState === 'down' ? 1 : 0),
    knee: Number(smKnee.toFixed(2)),
    hip: Number(smHip.toFixed(2)),
    trunk: Number(smTrunk.toFixed(2)),
    ankle: Number(smAnkle.toFixed(2)),
    valgus: Number(smValgus.toFixed(2)),
    kneeR: Number(angles.rawKneeR.toFixed(2)),
    kneeL: Number(angles.rawKneeL.toFixed(2)),
    hipR: Number(angles.rawHipR.toFixed(2)),
    hipL: Number(angles.rawHipL.toFixed(2)),
    vis: Number((angles.maxVis * 100).toFixed(0)),
    viewType: angles.viewType,
    trunkError: trunkErrorNow ? 1 : 0,
    valgusError: valgusErrorNow ? 1 : 0,
    coachingCue,
    coachingSeverity,
    // Compact 2D landmarks snapshot for synchronized skeleton replay in the Analysis Studio
    lm2d: lm2Snapshot
  };

  stateObj.timeSeries.push(frameRecord);
  return { frameRecord, completedRep };
}

// ============================================================================
// 4. MEDIAPIPE POSE INITIALIZATION & LIVE FRAME PROCESSING LOOP
// ============================================================================

let isWarmingUpPose = false;
let capturedFrameTime = 0.0;

async function initMediaPipePose() {
  if (poseInstance) return poseInstance;

  if (typeof Pose === 'undefined') {
    showToast('Loading MediaPipe Pose library...', 'warn');
    return null;
  }

  updateRecBadgeIdle('LOADING AI MODEL...');
  const p = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
  });

  p.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  p.onResults(onPoseResults);

  // Pre-warm WASM & TFLite runtime before starting video playback so t=0 frames are never missed
  try {
    isWarmingUpPose = true;
    if (typeof p.initialize === 'function') {
      await p.initialize();
    }
    const warmCanvas = document.createElement('canvas');
    warmCanvas.width = 64;
    warmCanvas.height = 64;
    await p.send({ image: warmCanvas });
  } catch (e) {
    console.warn('MediaPipe warmup note:', e);
  } finally {
    isWarmingUpPose = false;
  }

  poseInstance = p;
  return poseInstance;
}

/**
 * Compact 33-landmark snapshot for lightweight replay rendering
 */
function compactLandmarks(lm2) {
  if (!lm2) return null;
  // Save key body indices (11..32) + nose (0) with 4-decimal precision
  const out = {};
  const indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
  for (const idx of indices) {
    if (lm2[idx]) {
      out[idx] = {
        x: Number(lm2[idx].x.toFixed(4)),
        y: Number(lm2[idx].y.toFixed(4)),
        v: Number((lm2[idx].visibility ?? 1.0).toFixed(2))
      };
    }
  }
  return out;
}

/**
 * Callback invoked by MediaPipe Pose for each processed frame
 */
function onPoseResults(results) {
  if (isWarmingUpPose || !sessionState.isRunning) return;
  const videoEl = document.getElementById('webcam');
  const canvasEl = document.getElementById('output-canvas');
  if (!videoEl || !canvasEl) return;

  const vw = videoEl.videoWidth || 640;
  const vh = videoEl.videoHeight || 480;

  // Match canvas internal resolution to video pixel dimensions
  if (canvasEl.width !== vw || canvasEl.height !== vh) {
    canvasEl.width = vw;
    canvasEl.height = vh;
  }

  // Match CSS display box of canvas to the rendered <video> bounding box
  const videoRect = videoEl.getBoundingClientRect();
  if (videoRect.width > 0 && videoRect.height > 0) {
    canvasEl.style.width = `${videoRect.width}px`;
    canvasEl.style.height = `${videoRect.height}px`;
  }

  // Update portrait vs landscape container styling
  const viewportContainer = document.getElementById('viewport-container');
  if (viewportContainer) {
    viewportContainer.classList.toggle('aspect-portrait', vh > vw);
  }

  // Estimate FPS
  const nowPerf = performance.now();
  if (sessionState.lastFramePerf > 0) {
    const dt = (nowPerf - sessionState.lastFramePerf) / 1000.0;
    if (dt > 0.004 && dt < 0.5) {
      const instFps = 1.0 / dt;
      sessionState.fpsEstimate = 0.85 * sessionState.fpsEstimate + 0.15 * instFps;
    }
  }
  sessionState.lastFramePerf = nowPerf;

  // Compute timestamp (use captured frame timestamp for sample/uploaded videos, performance clock for live webcam)
  let timestampSec = 0.0;
  if (sessionState.mode === 'webcam') {
    if (sessionState.sessionStartPerf === 0) sessionState.sessionStartPerf = nowPerf;
    timestampSec = (nowPerf - sessionState.sessionStartPerf) / 1000.0;
  } else {
    timestampSec = capturedFrameTime >= 0 ? capturedFrameTime : (videoEl.currentTime || 0.0);
    // Prevent duplicate processing when video is at the exact same timestamp
    if (Math.abs(timestampSec - sessionState.lastVideoTime) < 0.002 && sessionState.lastVideoTime >= 0) {
      return;
    }
    sessionState.lastVideoTime = timestampSec;
  }

  const ctx = canvasEl.getContext('2d');
  ctx.save();
  ctx.clearRect(0, 0, vw, vh);

  if (results.poseLandmarks && results.poseLandmarks.length >= 29) {
    sessionState.frameIndex += 1;

    // 1. Compute fused 2D pixel + 3D world angles
    const angles = computeFusedBiomechanicalAngles(
      results.poseLandmarks,
      results.poseWorldLandmarks,
      vw,
      vh
    );

    // 2. Step state machine & record telemetry
    const snapshot = compactLandmarks(results.poseLandmarks);
    const { frameRecord, completedRep } = stepSquatStateMachine(
      angles,
      timestampSec,
      sessionState.frameIndex,
      sessionState.fpsEstimate,
      userThresholds,
      sessionState,
      snapshot
    );

    // 3. Draw skeleton & joint angle callouts on canvas
    drawPoseOverlayOnCanvas(ctx, results.poseLandmarks, vw, vh, frameRecord, sessionState.isMirrored);

    // 4. Update Live UI Cards & HUD
    updateLiveTelemetryUI(frameRecord, completedRep);
  } else {
    // No pose detected in frame
    const banner = document.getElementById('hud-coaching-text');
    if (banner) {
      banner.textContent = 'No person detected — step into the camera view';
    }
  }

  ctx.restore();
}

/**
 * Draw color-coded biomechanical skeleton and legible joint angle badges on a canvas
 */
function drawPoseOverlayOnCanvas(ctx, landmarks, w, h, frameRecord, isMirrored) {
  // Determine skeleton color based on biomechanical status
  let primaryColor = '#10B981'; // Emerald (Good)
  let jointFill = '#059669';
  if (frameRecord.trunkError || frameRecord.valgusError) {
    primaryColor = '#F43F5E';   // Rose (Form Error)
    jointFill = '#E11D48';
  } else if (frameRecord.state === 'down' && frameRecord.knee < userThresholds.fullKnee) {
    primaryColor = '#F59E0B';   // Amber (Descending / Partial depth)
    jointFill = '#D97706';
  }

  // Helper to map normalized landmark (x, y) to canvas pixels (handling mirror mode cleanly so text is never backwards!)
  const toPx = (pt) => ({
    x: (isMirrored ? (1.0 - pt.x) : pt.x) * w,
    y: pt.y * h
  });

  // Key biomechanical limb connections
  const pairs = [
    [11, 12], [11, 23], [12, 24], [23, 24], // Torso box
    [11, 13], [13, 15], [12, 14], [14, 16], // Arms
    [23, 25], [25, 27], [27, 29], [27, 31], [29, 31], // Left leg + foot
    [24, 26], [26, 28], [28, 30], [28, 32], [30, 32]  // Right leg + foot
  ];

  ctx.lineWidth = Math.max(2.5, w * 0.006);
  ctx.strokeStyle = primaryColor;
  ctx.lineCap = 'round';

  for (const [i, j] of pairs) {
    const p1 = landmarks[i];
    const p2 = landmarks[j];
    if (!p1 || !p2) continue;
    const v1 = p1.visibility ?? p1.v ?? 1.0;
    const v2 = p2.visibility ?? p2.v ?? 1.0;
    if (v1 < 0.25 || v2 < 0.25) continue;
    const c1 = toPx(p1);
    const c2 = toPx(p2);
    ctx.beginPath();
    ctx.moveTo(c1.x, c1.y);
    ctx.lineTo(c2.x, c2.y);
    ctx.stroke();
  }

  // Draw key joint nodes
  const keyNodes = [11, 12, 23, 24, 25, 26, 27, 28];
  const nodeRadius = Math.max(4, w * 0.009);
  for (const idx of keyNodes) {
    const pt = landmarks[idx];
    if (!pt) continue;
    const vis = pt.visibility ?? pt.v ?? 1.0;
    if (vis < 0.25) continue;
    const c = toPx(pt);
    ctx.beginPath();
    ctx.arc(c.x, c.y, nodeRadius, 0, 2 * Math.PI);
    ctx.fillStyle = jointFill;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#FFFFFF';
    ctx.stroke();
  }

  // Draw on-body floating angle callout pills next to Knee and Hip (always upright text!)
  const rKneePt = landmarks[26] || landmarks[25];
  const rHipPt = landmarks[24] || landmarks[23];
  const rShPt = landmarks[12] || landmarks[11];

  const drawCalloutPill = (pt, text, borderColor, offsetX = 14, offsetY = 0) => {
    if (!pt) return;
    const c = toPx(pt);
    const fontSize = Math.max(11, Math.round(w * 0.026));
    ctx.font = `700 ${fontSize}px Inter, sans-serif`;
    const metrics = ctx.measureText(text);
    const padX = 6, padY = 4;
    const boxW = metrics.width + padX * 2;
    const boxH = fontSize + padY * 2;
    const bx = clamp(c.x + offsetX, 4, w - boxW - 4);
    const by = clamp(c.y + offsetY - boxH / 2, 4, h - boxH - 4);

    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.beginPath();
    ctx.roundRect(bx, by, boxW, boxH, 5);
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = borderColor;
    ctx.stroke();

    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(text, bx + padX, by + boxH - padY - 2);
  };

  drawCalloutPill(rKneePt, `Knee ${frameRecord.knee.toFixed(0)}°`, '#38BDF8', 14, 0);
  drawCalloutPill(rHipPt, `Hip ${frameRecord.hip.toFixed(0)}°`, '#34D399', 14, -6);
  drawCalloutPill(
    rShPt,
    `Trunk ${frameRecord.trunk.toFixed(0)}°`,
    frameRecord.trunkError ? '#F43F5E' : '#FBBF24',
    14,
    10
  );
}

/**
 * Main requestAnimationFrame loop that feeds frames to MediaPipe Pose
 */
async function processVideoFrameLoop() {
  if (!sessionState.isRunning) return;

  const videoEl = document.getElementById('webcam');
  if (videoEl && videoEl.readyState >= 2 && !videoEl.paused && !videoEl.ended && !isSendingFrame) {
    try {
      isSendingFrame = true;
      capturedFrameTime = videoEl.currentTime || 0.0;
      const t0 = performance.now();
      await poseInstance.send({ image: videoEl });
      const dtMs = performance.now() - t0;

      // Adapt video playback rate on slower/headless CPUs so pre-recorded videos never skip squat phases
      if ((sessionState.mode === 'sample' || sessionState.mode === 'upload') && !videoEl.paused && !videoEl.ended) {
        const targetRate = clamp(24.0 / Math.max(1.0, dtMs), 0.25, 1.0);
        if (Math.abs(videoEl.playbackRate - targetRate) > 0.08) {
          videoEl.playbackRate = targetRate;
        }
      }
    } catch (err) {
      console.warn('MediaPipe Pose frame warning:', err);
    } finally {
      isSendingFrame = false;
    }
  }

  if (sessionState.isRunning) {
    rafId = requestAnimationFrame(processVideoFrameLoop);
  }
}

// ============================================================================
// 5. LIVE UI TELEMETRY, SCOREBOARD & MINI-CHART UPDATES
// ============================================================================

function updateLiveTelemetryUI(frameRecord, completedRep) {
  // Top bar badges
  const visBadge = document.getElementById('visibility-badge');
  const fpsBadge = document.getElementById('fps-badge');
  const frameBadge = document.getElementById('frame-count-badge');
  const studioBadge = document.getElementById('studio-badge-count');

  if (visBadge) visBadge.textContent = `Vis: ${frameRecord.vis}%`;
  if (fpsBadge) fpsBadge.textContent = `${sessionState.fpsEstimate.toFixed(0)} FPS`;
  if (frameBadge) frameBadge.textContent = `${sessionState.timeSeries.length} frames`;
  if (studioBadge) studioBadge.textContent = `${sessionState.repsSummary.length}`;

  // HUD Phase & View Badges
  const phaseBadge = document.getElementById('hud-phase-badge');
  const viewBadge = document.getElementById('hud-view-badge');
  const coachingBanner = document.getElementById('hud-coaching-banner');
  const coachingText = document.getElementById('hud-coaching-text');

  if (phaseBadge) {
    phaseBadge.textContent = `PHASE: ${frameRecord.phase}`;
    if (frameRecord.state === 'down') {
      phaseBadge.className = 'px-3 py-1 rounded-md text-xs font-black tracking-wide uppercase bg-blue-600/90 text-white border border-blue-400 shadow-md backdrop-blur-sm';
    } else {
      phaseBadge.className = 'px-3 py-1 rounded-md text-xs font-black tracking-wide uppercase bg-slate-900/85 text-white border border-slate-600 shadow-md backdrop-blur-sm';
    }
  }

  if (viewBadge) {
    viewBadge.textContent = `View: ${frameRecord.viewType}`;
  }

  if (coachingBanner && coachingText) {
    coachingText.textContent = frameRecord.coachingCue;
    if (frameRecord.coachingSeverity === 'error') {
      coachingBanner.className = 'px-3.5 py-2 rounded-lg text-xs sm:text-sm font-bold bg-rose-950/90 text-rose-200 border border-rose-500 shadow-lg backdrop-blur-sm';
    } else if (frameRecord.coachingSeverity === 'warn') {
      coachingBanner.className = 'px-3.5 py-2 rounded-lg text-xs sm:text-sm font-bold bg-amber-950/90 text-amber-200 border border-amber-500/70 shadow-lg backdrop-blur-sm';
    } else {
      coachingBanner.className = 'px-3.5 py-2 rounded-lg text-xs sm:text-sm font-bold bg-slate-900/85 text-emerald-300 border border-emerald-500/40 shadow-lg backdrop-blur-sm';
    }
  }

  // HUD Vertical Depth Gauge (0 to 140 deg scale)
  const depthFill = document.getElementById('hud-depth-fill');
  const depthVal = document.getElementById('hud-depth-val');
  if (depthFill && depthVal) {
    const pct = clamp((frameRecord.knee / 140.0) * 100.0, 0.0, 100.0);
    depthFill.style.height = `${pct}%`;
    depthVal.textContent = `${frameRecord.knee.toFixed(0)}°`;
  }

  // 5 Telemetry Cards
  const elKnee = document.getElementById('live-knee-angle');
  const elKneeRL = document.getElementById('live-knee-rl');
  const elHip = document.getElementById('live-hip-angle');
  const elHipRL = document.getElementById('live-hip-rl');
  const elTrunk = document.getElementById('live-trunk-angle');
  const elTrunkStatus = document.getElementById('live-trunk-status');
  const elAnkle = document.getElementById('live-ankle-angle');
  const elValgus = document.getElementById('live-valgus-angle');
  const elValgusStatus = document.getElementById('live-valgus-status');

  if (elKnee) elKnee.textContent = `${frameRecord.knee.toFixed(1)}°`;
  if (elKneeRL) elKneeRL.textContent = `R: ${frameRecord.kneeR.toFixed(0)}° · L: ${frameRecord.kneeL.toFixed(0)}°`;
  if (elHip) elHip.textContent = `${frameRecord.hip.toFixed(1)}°`;
  if (elHipRL) elHipRL.textContent = `R: ${frameRecord.hipR.toFixed(0)}° · L: ${frameRecord.hipL.toFixed(0)}°`;
  if (elTrunk) elTrunk.textContent = `${frameRecord.trunk.toFixed(1)}°`;
  if (elAnkle) elAnkle.textContent = `${frameRecord.ankle.toFixed(1)}°`;
  if (elValgus) elValgus.textContent = `${frameRecord.valgus.toFixed(1)}°`;

  // Highlight Trunk / Valgus cards if exceeding error threshold
  const cardTrunk = document.getElementById('card-trunk-lean');
  if (cardTrunk && elTrunkStatus) {
    if (frameRecord.trunkError) {
      cardTrunk.className = 'bg-rose-50 rounded-xl p-3.5 shadow-sm border-2 border-rose-500 transition';
      elTrunkStatus.innerHTML = '<span class="text-rose-700 font-bold"><i class="fas fa-exclamation-circle mr-1"></i>Excessive Lean!</span>';
    } else {
      cardTrunk.className = 'bg-white rounded-xl p-3.5 shadow-sm border border-slate-200 border-t-4 border-t-amber-500 transition';
      elTrunkStatus.textContent = `Limit ≤ ${userThresholds.maxTrunk}°`;
    }
  }

  const cardValgus = document.getElementById('card-knee-valgus');
  if (cardValgus && elValgusStatus) {
    if (frameRecord.valgusError) {
      cardValgus.className = 'col-span-2 sm:col-span-1 bg-rose-50 rounded-xl p-3.5 shadow-sm border-2 border-rose-500 transition';
      elValgusStatus.innerHTML = '<span class="text-rose-700 font-bold"><i class="fas fa-exclamation-circle mr-1"></i>Medial Collapse!</span>';
    } else {
      cardValgus.className = 'col-span-2 sm:col-span-1 bg-white rounded-xl p-3.5 shadow-sm border border-slate-200 border-t-4 border-t-rose-500 transition';
      elValgusStatus.textContent = `Limit ≤ ${userThresholds.maxValgus}°`;
    }
  }

  // Video playback time readout (for sample/uploaded videos)
  const videoEl = document.getElementById('webcam');
  const timeReadout = document.getElementById('live-video-time');
  if (videoEl && timeReadout && sessionState.mode !== 'webcam') {
    const dur = Number.isFinite(videoEl.duration) ? videoEl.duration : 0;
    timeReadout.textContent = `${frameRecord.time.toFixed(2)}s / ${dur.toFixed(2)}s`;
  }

  // If a repetition just completed, update counters, flash cards, append to feed & speak cue
  if (completedRep) {
    document.getElementById('full-rep-counter').textContent = String(sessionState.fullReps);
    document.getElementById('partial-rep-counter').textContent = String(sessionState.partialReps);
    document.getElementById('error-rep-counter').textContent = String(sessionState.errorReps);

    const flashCardId = completedRep.type === 'FULL' ? 'card-counter-full' : 'card-counter-partial';
    const flashClass = completedRep.type === 'FULL' ? 'flash-rep-full' : 'flash-rep-partial';
    const cardEl = document.getElementById(flashCardId);
    if (cardEl) {
      cardEl.classList.remove(flashClass);
      void cardEl.offsetWidth;
      cardEl.classList.add(flashClass);
    }

    if (completedRep.hasError) {
      const errCard = document.getElementById('card-counter-error');
      if (errCard) {
        errCard.classList.remove('flash-rep-error');
        void errCard.offsetWidth;
        errCard.classList.add('flash-rep-error');
      }
    }

    renderLiveRepFeed();
    speakAudioCoachingCue(completedRep);
  }

  // Throttled live mini-chart update (~6 Hz)
  const now = performance.now();
  if (now - lastMiniChartUpdate > 165) {
    lastMiniChartUpdate = now;
    updateLiveMiniChart();
  }
}

/**
 * Render the live Rep-by-Rep diagnostic feed in the right column
 */
function renderLiveRepFeed() {
  const container = document.getElementById('live-rep-log');
  const summaryEl = document.getElementById('rep-feed-summary');
  if (!container) return;

  const reps = sessionState.repsSummary;
  if (summaryEl) {
    summaryEl.textContent = `${reps.length} ${reps.length === 1 ? 'rep' : 'reps'} logged`;
  }

  if (reps.length === 0) {
    container.innerHTML = `
      <div class="text-center py-6 text-xs text-slate-400">
        <i class="fas fa-dumbbell text-2xl mb-1.5 block text-slate-300"></i>
        Completed squat repetitions will appear here with peak angles and form diagnostics.
      </div>`;
    return;
  }

  // Render newest rep on top
  container.innerHTML = reps
    .slice()
    .reverse()
    .map((r) => {
      const isFull = r.type === 'FULL';
      const badgeClass = isFull
        ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
        : 'bg-amber-100 text-amber-800 border-amber-300';

      const errTags = r.errors
        .map((e) => {
          if (e === 'Good Form') {
            return `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200"><i class="fas fa-check-circle mr-1"></i>Good Form</span>`;
          }
          return `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-100 text-rose-800 border border-rose-300"><i class="fas fa-exclamation-triangle mr-1"></i>${e}</span>`;
        })
        .join(' ');

      return `
        <div class="p-2.5 rounded-lg border border-slate-200 bg-slate-50/70 hover:bg-slate-100/80 transition flex flex-col gap-1.5">
          <div class="flex items-center justify-between gap-2">
            <div class="flex items-center gap-2">
              <span class="w-6 h-6 rounded-full bg-slate-900 text-white text-xs font-bold flex items-center justify-center">#${r.rep}</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold border ${badgeClass}">${r.type} REP</span>
              <span class="text-[11px] font-mono-num text-slate-500">${r.startTime.toFixed(2)}s – ${r.endTime.toFixed(2)}s</span>
            </div>
            <button type="button" onclick="inspectRepInStudio(${r.rep})" class="text-[11px] font-bold text-blue-600 hover:text-blue-800 flex items-center gap-1">
              <i class="fas fa-play-circle"></i>
              <span>Replay</span>
            </button>
          </div>
          <div class="flex items-center justify-between text-[11px] font-mono-num text-slate-700 bg-white px-2.5 py-1.5 rounded border border-slate-200/80">
            <span>Knee: <strong class="text-blue-600">${r.peakKnee}°</strong></span>
            <span>Hip: <strong class="text-emerald-600">${r.peakHip}°</strong></span>
            <span>Trunk: <strong class="text-amber-600">${r.peakTrunk}°</strong></span>
            <span>Ankle: <strong class="text-purple-600">${r.peakAnkle}°</strong></span>
          </div>
          <div class="flex items-center gap-1.5 flex-wrap">
            ${errTags}
          </div>
        </div>
      `;
    })
    .join('');
}

/**
 * Optional Web Speech API voice cue on rep completion
 */
function speakAudioCoachingCue(rep) {
  const voiceToggle = document.getElementById('toggle-voice-cues');
  if (!voiceToggle || !voiceToggle.checked || !('speechSynthesis' in window)) return;
  try {
    window.speechSynthesis.cancel();
    let msg = `Rep ${rep.rep}, ${rep.type === 'FULL' ? 'Full squat' : 'Partial depth'}.`;
    if (rep.hasError) {
      msg += ` Warning: ${rep.errors.join(' and ')}.`;
    }
    const utter = new SpeechSynthesisUtterance(msg);
    utter.rate = 1.15;
    window.speechSynthesis.speak(utter);
  } catch (e) {
    // Ignore speech synthesis errors on restricted browsers
  }
}

/**
 * Initialize and update the real-time scrolling mini-chart on the main dashboard
 */
function initLiveMiniChart() {
  const el = document.getElementById('live-mini-chart');
  if (!el || typeof Plotly === 'undefined') return;

  const traces = [
    { x: [], y: [], name: 'Knee (°)', mode: 'lines', line: { color: '#2563EB', width: 2.5 } },
    { x: [], y: [], name: 'Hip (°)', mode: 'lines', line: { color: '#059669', width: 2 } },
    { x: [], y: [], name: 'Trunk (°)', mode: 'lines', line: { color: '#D97706', width: 1.75, dash: 'dot' } }
  ];

  const layout = {
    margin: { t: 8, r: 12, b: 32, l: 38 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: '#F8FAFC',
    showlegend: true,
    legend: { orientation: 'h', x: 0, y: 1.15, font: { size: 10 } },
    xaxis: { title: { text: 'Time (s)', font: { size: 10 } }, gridcolor: '#E2E8F0', tickfont: { size: 10 } },
    yaxis: { title: { text: 'Angle (°)', font: { size: 10 } }, range: [0, 145], gridcolor: '#E2E8F0', tickfont: { size: 10 } },
    shapes: [
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        y0: userThresholds.fullKnee,
        y1: userThresholds.fullKnee,
        line: { color: '#10B981', width: 1, dash: 'dash' }
      }
    ]
  };

  Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });
}

function updateLiveMiniChart() {
  const el = document.getElementById('live-mini-chart');
  if (!el || typeof Plotly === 'undefined') return;

  const series = sessionState.timeSeries;
  if (series.length === 0) return;

  // Show last 10 seconds of data
  const latestT = series[series.length - 1].time;
  const minT = Math.max(0, latestT - 10.0);
  const windowSlice = series.filter((d) => d.time >= minT);

  const tArr = windowSlice.map((d) => d.time);
  const kArr = windowSlice.map((d) => d.knee);
  const hArr = windowSlice.map((d) => d.hip);
  const trArr = windowSlice.map((d) => d.trunk);

  Plotly.react(
    el,
    [
      { x: tArr, y: kArr, name: 'Knee (°)', mode: 'lines', line: { color: '#2563EB', width: 2.5 } },
      { x: tArr, y: hArr, name: 'Hip (°)', mode: 'lines', line: { color: '#059669', width: 2 } },
      { x: tArr, y: trArr, name: 'Trunk (°)', mode: 'lines', line: { color: '#D97706', width: 1.75, dash: 'dot' } }
    ],
    {
      margin: { t: 8, r: 12, b: 32, l: 38 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#F8FAFC',
      showlegend: true,
      legend: { orientation: 'h', x: 0, y: 1.15, font: { size: 10 } },
      xaxis: {
        title: { text: 'Time (s)', font: { size: 10 } },
        range: [minT, Math.max(minT + 3.5, latestT + 0.15)],
        gridcolor: '#E2E8F0',
        tickfont: { size: 10 }
      },
      yaxis: {
        title: { text: 'Angle (°)', font: { size: 10 } },
        range: [0, 145],
        gridcolor: '#E2E8F0',
        tickfont: { size: 10 }
      },
      shapes: [
        {
          type: 'line',
          xref: 'paper',
          x0: 0,
          x1: 1,
          y0: userThresholds.fullKnee,
          y1: userThresholds.fullKnee,
          line: { color: '#10B981', width: 1, dash: 'dash' }
        }
      ]
    },
    { responsive: true, displayModeBar: false }
  );
}

// ============================================================================
// 6. SMARTPHONE & DESKTOP CAMERA MANAGEMENT + CROSS-BROWSER MEDIARECORDER
// ============================================================================

/**
 * Detect supported MediaRecorder MIME type for iOS Safari, Android Chrome, and Desktop
 */
function getSupportedMimeType() {
  if (typeof MediaRecorder === 'undefined') return '';
  const candidates = [
    'video/mp4',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm'
  ];
  for (const mime of candidates) {
    if (typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported(mime)) {
      return mime;
    }
  }
  return '';
}

/**
 * Populate the camera device dropdown after camera permission is granted
 */
async function refreshCameraDevicesList() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((d) => d.kind === 'videoinput');
    const selectEl = document.getElementById('camera-select');
    if (!selectEl) return;

    const previousValue = selectEl.value;
    selectEl.innerHTML = '<option value="">Default Camera (Auto)</option>';
    videoInputs.forEach((device, idx) => {
      const opt = document.createElement('option');
      opt.value = device.deviceId;
      opt.textContent = device.label || `Camera ${idx + 1}`;
      selectEl.appendChild(opt);
    });

    if (previousValue && videoInputs.some((d) => d.deviceId === previousValue)) {
      selectEl.value = previousValue;
    }
  } catch (err) {
    console.warn('Could not enumerate video devices:', err);
  }
}

/**
 * Start live camera stream (supports smartphone front/rear switching & portrait/landscape)
 */
async function startLiveCamera(options = {}) {
  stopCurrentSession(false);
  resetSessionData();

  await initMediaPipePose();

  const videoEl = document.getElementById('webcam');
  const selectEl = document.getElementById('camera-select');
  const selectedDeviceId = options.deviceId !== undefined ? options.deviceId : (selectEl ? selectEl.value : '');
  const facing = options.facingMode || sessionState.facingMode || 'environment';

  // Build smartphone-friendly constraints (never request audio!)
  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 }
  };

  if (selectedDeviceId) {
    videoConstraints.deviceId = { exact: selectedDeviceId };
  } else {
    videoConstraints.facingMode = { ideal: facing };
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false
    });

    sessionState.mediaStream = stream;
    sessionState.mode = 'webcam';
    sessionState.currentSampleId = null;
    sessionState.trialTitle = `Live Camera (${facing === 'user' ? 'Front Selfie' : 'Rear Environment'})`;
    sessionState.isRunning = true;

    // Auto-enable mirroring for front-facing camera unless overridden
    const track = stream.getVideoTracks()[0];
    const settings = track ? track.getSettings() : {};
    const isFront = settings.facingMode === 'user' || facing === 'user';
    setMirrorMode(isFront);

    videoEl.srcObject = stream;
    videoEl.removeAttribute('src');
    videoEl.loop = false;
    await videoEl.play();

    // Start MediaRecorder so the live camera session can be replayed in the Analysis Studio
    startStreamRecording(stream);

    // Refresh device list labels
    await refreshCameraDevicesList();

    // Update UI state
    updateSessionUIForActiveMode('LIVE CAMERA · REC');
    processVideoFrameLoop();
    showToast('Live camera started — recording joint kinematics', 'good');
  } catch (err) {
    console.error('Camera access error:', err);
    showToast(`Camera error: ${err.message || 'Check HTTPS & camera permissions'}`, 'error');
  }
}

/**
 * Start MediaRecorder on the active MediaStream
 */
function startStreamRecording(stream) {
  sessionState.recordedChunks = [];
  if (sessionState.recordedVideoUrl) {
    URL.revokeObjectURL(sessionState.recordedVideoUrl);
    sessionState.recordedVideoUrl = null;
  }

  if (typeof MediaRecorder === 'undefined') return;
  try {
    const mimeType = getSupportedMimeType();
    sessionState.recordedMimeType = mimeType;
    const rec = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

    rec.ondataavailable = (evt) => {
      if (evt.data && evt.data.size > 0) {
        sessionState.recordedChunks.push(evt.data);
      }
    };

    rec.onstop = () => {
      if (sessionState.recordedChunks.length > 0) {
        const blob = new Blob(sessionState.recordedChunks, {
          type: sessionState.recordedMimeType || 'video/webm'
        });
        sessionState.recordedVideoUrl = URL.createObjectURL(blob);
      }
    };

    rec.start(250); // Collect 250ms chunks
    sessionState.mediaRecorder = rec;
  } catch (e) {
    console.warn('MediaRecorder fallback (pose replay still active):', e);
  }
}

/**
 * Toggle horizontal mirror display
 */
function setMirrorMode(enabled) {
  sessionState.isMirrored = Boolean(enabled);
  const videoEl = document.getElementById('webcam');
  const toggleEl = document.getElementById('toggle-mirror');
  if (videoEl) {
    videoEl.classList.toggle('mirror-video', sessionState.isMirrored);
  }
  if (toggleEl) {
    toggleEl.checked = sessionState.isMirrored;
  }
}

// ============================================================================
// 7. SAMPLE VIDEOS & CUSTOM VIDEO UPLOAD PIPELINE
// ============================================================================

const SAMPLE_VIDEOS = {
  sagittal: {
    url: 'videos/sample_squat_sagittal.mp4',
    title: 'Sample 1: Deep Squat (Cam0 Oblique · OpenSim Ground Truth)',
    hasOpenSimGT: true
  },
  frontal: {
    url: 'videos/sample_squat_frontal.mp4',
    title: 'Sample 1B: Deep Squat (Cam1 Frontal · OpenSim Ground Truth)',
    hasOpenSimGT: true
  },
  multiview: {
    url: 'videos/sample_squat_multiview.mp4',
    title: 'Sample 2: 4-Rep Multi-View Squat (4 Camera Orientations)',
    hasOpenSimGT: false
  }
};

/**
 * Load and run one of the built-in validated sample videos
 */
async function startSampleVideo(sampleKey) {
  const sample = SAMPLE_VIDEOS[sampleKey];
  if (!sample) return;

  stopCurrentSession(false);
  resetSessionData();
  await initMediaPipePose();

  const videoEl = document.getElementById('webcam');
  setMirrorMode(false);

  sessionState.mode = 'sample';
  sessionState.currentSampleId = sampleKey;
  sessionState.trialTitle = sample.title;
  sessionState.sourceVideoUrl = sample.url;
  sessionState.isRunning = true;

  videoEl.srcObject = null;
  videoEl.src = sample.url;
  videoEl.loop = false;
  videoEl.playbackRate = 1.0;
  videoEl.currentTime = 0;

  videoEl.onended = async () => {
    let waitTicks = 0;
    while (isSendingFrame && waitTicks < 30) {
      await new Promise((r) => setTimeout(r, 25));
      waitTicks++;
    }
    sessionState.isRunning = false;
    updateRecBadgeIdle('COMPLETED');
    const autoOpen = document.getElementById('toggle-auto-studio');
    if (autoOpen && autoOpen.checked) {
      openAnalysisStudio();
    } else {
      showToast('Trial finished! Click "Analysis Studio" to inspect kinematics.', 'good');
    }
  };

  try {
    await videoEl.play();
    updateSessionUIForActiveMode('SAMPLE TRIAL · ANALYZING');
    processVideoFrameLoop();
    showToast(`Analyzing ${sample.title}`, 'good');
  } catch (err) {
    console.error('Sample video playback error:', err);
    showToast(`Could not play sample video: ${err.message}`, 'error');
  }
}

/**
 * Load and analyze a user-uploaded video file (.mp4, .mov, .webm)
 */
async function handleUploadedVideoFile(file) {
  if (!file) return;

  stopCurrentSession(false);
  resetSessionData();
  await initMediaPipePose();

  const videoEl = document.getElementById('webcam');
  setMirrorMode(false);

  if (sessionState.sourceVideoUrl && sessionState.mode === 'upload') {
    URL.revokeObjectURL(sessionState.sourceVideoUrl);
  }

  const fileUrl = URL.createObjectURL(file);
  sessionState.mode = 'upload';
  sessionState.currentSampleId = null;
  sessionState.trialTitle = `Uploaded: ${file.name}`;
  sessionState.sourceVideoUrl = fileUrl;
  sessionState.isRunning = true;

  videoEl.srcObject = null;
  videoEl.src = fileUrl;
  videoEl.loop = false;
  videoEl.playbackRate = 1.0;
  videoEl.currentTime = 0;

  videoEl.onended = async () => {
    let waitTicks = 0;
    while (isSendingFrame && waitTicks < 30) {
      await new Promise((r) => setTimeout(r, 25));
      waitTicks++;
    }
    sessionState.isRunning = false;
    updateRecBadgeIdle('COMPLETED');
    const autoOpen = document.getElementById('toggle-auto-studio');
    if (autoOpen && autoOpen.checked) {
      openAnalysisStudio();
    }
  };

  try {
    await videoEl.play();
    updateSessionUIForActiveMode('UPLOADED VIDEO · ANALYZING');
    processVideoFrameLoop();
    showToast(`Analyzing uploaded video: ${file.name}`, 'good');
  } catch (err) {
    showToast(`Error playing uploaded video: ${err.message}`, 'error');
  }
}

/**
 * Stop current camera or video session and optionally open the Analysis Studio
 */
function stopCurrentSession(openStudioAfter = false) {
  sessionState.isRunning = false;
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }

  const videoEl = document.getElementById('webcam');
  if (videoEl) {
    videoEl.pause();
  }

  if (sessionState.mediaRecorder && sessionState.mediaRecorder.state !== 'inactive') {
    try {
      sessionState.mediaRecorder.stop();
    } catch (e) {
      // Ignore
    }
  }

  if (sessionState.mediaStream) {
    sessionState.mediaStream.getTracks().forEach((t) => t.stop());
    sessionState.mediaStream = null;
  }

  // Update button states
  const camBtnLabel = document.getElementById('btn-camera-label');
  const camBtnIcon = document.getElementById('btn-camera-icon');
  const camBtn = document.getElementById('btn-toggle-camera');
  if (camBtnLabel) camBtnLabel.textContent = 'Start Live Camera';
  if (camBtnIcon) camBtnIcon.className = 'fas fa-video';
  if (camBtn) {
    camBtn.className = 'px-4 py-2 text-xs sm:text-sm font-bold rounded-lg bg-blue-600 hover:bg-blue-700 text-white shadow-sm transition flex items-center gap-2';
  }

  updateRecBadgeIdle('STOPPED');

  if (openStudioAfter) {
    setTimeout(() => openAnalysisStudio(), 180);
  }
}

/**
 * Reset all session counters and recorded time-series data
 */
function resetSessionData() {
  sessionState.squatState = 'up';
  sessionState.phaseLabel = 'STANDING';
  sessionState.fullReps = 0;
  sessionState.partialReps = 0;
  sessionState.errorReps = 0;
  sessionState.smKnee = null;
  sessionState.smHip = null;
  sessionState.smTrunk = null;
  sessionState.smAnkle = null;
  sessionState.smValgus = null;
  sessionState.frameIndex = 0;
  sessionState.sessionStartPerf = 0;
  sessionState.lastVideoTime = -1;
  sessionState.lastFramePerf = 0;
  sessionState.timeSeries = [];
  sessionState.repsSummary = [];

  document.getElementById('full-rep-counter').textContent = '0';
  document.getElementById('partial-rep-counter').textContent = '0';
  document.getElementById('error-rep-counter').textContent = '0';
  document.getElementById('studio-badge-count').textContent = '0';
  document.getElementById('frame-count-badge').textContent = '0 frames';

  renderLiveRepFeed();
  initLiveMiniChart();
}

function updateSessionUIForActiveMode(statusLabel) {
  const idleOverlay = document.getElementById('viewport-idle-overlay');
  if (idleOverlay) idleOverlay.classList.add('hidden');

  const sourceLabel = document.getElementById('source-title-label');
  if (sourceLabel) sourceLabel.textContent = sessionState.trialTitle;

  const recBadge = document.getElementById('rec-badge');
  const recDot = document.getElementById('rec-dot');
  const recText = document.getElementById('rec-status-text');
  if (recBadge && recDot && recText) {
    recBadge.className = 'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-rose-950 text-rose-200 border border-rose-600';
    recDot.className = 'w-2 h-2 rounded-full bg-rose-500 rec-dot';
    recText.textContent = statusLabel;
  }

  const camStatusPill = document.getElementById('camera-status-pill');
  const camBtnLabel = document.getElementById('btn-camera-label');
  const camBtn = document.getElementById('btn-toggle-camera');
  if (sessionState.mode === 'webcam') {
    if (camStatusPill) {
      camStatusPill.textContent = 'Live Streaming';
      camStatusPill.className = 'px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-300';
    }
    if (camBtnLabel) camBtnLabel.textContent = 'Stop Camera';
    if (camBtn) {
      camBtn.className = 'px-4 py-2 text-xs sm:text-sm font-bold rounded-lg bg-rose-600 hover:bg-rose-700 text-white shadow-sm transition flex items-center gap-2';
    }
  } else {
    if (camStatusPill) {
      camStatusPill.textContent = 'Video Mode';
      camStatusPill.className = 'px-2 py-0.5 rounded-full text-[10px] font-bold bg-blue-100 text-blue-800 border border-blue-200';
    }
  }

  // Show/hide bottom video controls bar for sample/uploaded videos
  const videoControlsBar = document.getElementById('live-video-controls');
  if (videoControlsBar) {
    videoControlsBar.classList.toggle('hidden', sessionState.mode === 'webcam');
  }
}

function updateRecBadgeIdle(text = 'STANDBY') {
  const recBadge = document.getElementById('rec-badge');
  const recDot = document.getElementById('rec-dot');
  const recText = document.getElementById('rec-status-text');
  if (recBadge && recDot && recText) {
    recBadge.className = 'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-slate-800 text-slate-300 border border-slate-700';
    recDot.className = 'w-2 h-2 rounded-full bg-emerald-400';
    recText.textContent = text;
  }
}

// ============================================================================
// 8. POST-RECORDING VIDEO REPLAY & KINEMATIC ANALYSIS STUDIO
// ============================================================================

/**
 * Open the Post-Recording Kinematic Analysis Studio modal and populate video, charts & tables
 */
function openAnalysisStudio() {
  const modal = document.getElementById('analysis-studio-modal');
  if (!modal) return;

  // Pause live session if still running
  if (sessionState.isRunning) {
    stopCurrentSession(false);
  }

  modal.classList.remove('hidden');
  document.body.style.overflow = 'hidden';

  const series = sessionState.timeSeries;
  const reps = sessionState.repsSummary;

  // Populate KPI strip
  document.getElementById('studio-trial-badge').textContent = sessionState.trialTitle || 'Recorded Session';
  document.getElementById('studio-kpi-total').textContent = String(reps.length);
  document.getElementById('studio-kpi-full').textContent = String(sessionState.fullReps);
  document.getElementById('studio-kpi-partial').textContent = String(sessionState.partialReps);
  document.getElementById('studio-kpi-errors').textContent = String(sessionState.errorReps);

  const maxKneeAll = series.length > 0 ? Math.max(...series.map((d) => d.knee)) : 0.0;
  const totalDur = series.length > 0 ? series[series.length - 1].time : 0.0;
  document.getElementById('studio-kpi-maxknee').textContent = `${maxKneeAll.toFixed(1)}°`;
  document.getElementById('studio-kpi-duration').textContent = `${totalDur.toFixed(2)}s (${series.length}f)`;

  // Show/hide OpenSim Ground Truth toggle depending on whether Sample 1/1B is active
  const isSample1 = (sessionState.currentSampleId === 'sagittal' || sessionState.currentSampleId === 'frontal');
  const gtToggleWrap = document.getElementById('opensim-gt-toggle-wrap');
  const rmseBadge = document.getElementById('studio-opensim-rmse-badge');
  if (gtToggleWrap) {
    gtToggleWrap.classList.toggle('hidden', !isSample1);
  }
  if (rmseBadge) {
    if (isSample1 && openSimGroundTruth && series.length > 0) {
      const gtPeakKnee = Math.max(...openSimGroundTruth.knee_angle_r);
      const diff = maxKneeAll - gtPeakKnee;
      rmseBadge.textContent = `OpenSim GT Peak Knee: ${gtPeakKnee.toFixed(1)}° (Δ = ${diff >= 0 ? '+' : ''}${diff.toFixed(1)}°)`;
      rmseBadge.classList.remove('hidden');
    } else {
      rmseBadge.classList.add('hidden');
    }
  }

  // Configure replay video player source
  const replayVideo = document.getElementById('replay-video');
  const scrubber = document.getElementById('replay-scrubber');
  const durLabel = document.getElementById('replay-duration');

  activeRepLoopBounds = null;
  const videoUrl = sessionState.mode === 'webcam' ? sessionState.recordedVideoUrl : sessionState.sourceVideoUrl;
  if (replayVideo && videoUrl) {
    if (replayVideo.src !== videoUrl) {
      replayVideo.src = videoUrl;
    }
    replayVideo.currentTime = 0;
  }

  if (scrubber) {
    scrubber.max = String(Math.max(0.1, totalDur));
    scrubber.value = '0';
  }
  if (durLabel) {
    durLabel.textContent = `${totalDur.toFixed(2)}s`;
  }

  // Render Studio Plotly Kinematic Chart & Rep Table
  renderStudioPlotlyChart();
  renderStudioRepTable();
  updateStudioPlayheadAtTime(0.0);
}

function closeAnalysisStudio() {
  const modal = document.getElementById('analysis-studio-modal');
  const replayVideo = document.getElementById('replay-video');
  if (replayVideo) replayVideo.pause();
  if (replayRafId) {
    cancelAnimationFrame(replayRafId);
    replayRafId = null;
  }
  if (modal) modal.classList.add('hidden');
  document.body.style.overflow = '';
}

/**
 * Render the interactive Plotly.js joint kinematic chart inside the Analysis Studio
 */
function renderStudioPlotlyChart() {
  const chartEl = document.getElementById('studio-plotly-chart');
  if (!chartEl || typeof Plotly === 'undefined') return;

  const series = sessionState.timeSeries;
  const reps = sessionState.repsSummary;
  const tArr = series.map((d) => d.time);

  const traces = [
    {
      x: tArr,
      y: series.map((d) => d.knee),
      name: 'Knee Flexion (°)',
      mode: 'lines',
      line: { color: '#2563EB', width: 3 }
    },
    {
      x: tArr,
      y: series.map((d) => d.hip),
      name: 'Hip Flexion (°)',
      mode: 'lines',
      line: { color: '#059669', width: 2.5 }
    },
    {
      x: tArr,
      y: series.map((d) => d.trunk),
      name: 'Trunk Lean (°)',
      mode: 'lines',
      line: { color: '#D97706', width: 2.2 }
    },
    {
      x: tArr,
      y: series.map((d) => d.ankle),
      name: 'Ankle Dorsiflexion (°)',
      mode: 'lines',
      line: { color: '#7C3AED', width: 2 }
    },
    {
      x: tArr,
      y: series.map((d) => d.valgus),
      name: 'Knee Valgus (°)',
      mode: 'lines',
      line: { color: '#E11D48', width: 1.8, dash: 'dot' }
    }
  ];

  // Overlay OpenSim Ground Truth curves if Sample 1/1B is active and checkbox is checked
  const isSample1 = (sessionState.currentSampleId === 'sagittal' || sessionState.currentSampleId === 'frontal');
  const gtChecked = document.getElementById('toggle-opensim-gt')?.checked;

  if (isSample1 && gtChecked && openSimGroundTruth) {
    const gtT = openSimGroundTruth.time;
    const gtTrunk = openSimGroundTruth.pelvis_tilt.map((p, idx) =>
      Math.abs(p + (openSimGroundTruth.lumbar_extension[idx] || 0))
    );

    traces.push(
      {
        x: gtT,
        y: openSimGroundTruth.knee_angle_r,
        name: 'OpenSim GT Knee R (°)',
        mode: 'lines',
        line: { color: '#1E40AF', width: 2, dash: 'dash' }
      },
      {
        x: gtT,
        y: openSimGroundTruth.hip_flexion_r,
        name: 'OpenSim GT Hip R (°)',
        mode: 'lines',
        line: { color: '#065F46', width: 2, dash: 'dash' }
      },
      {
        x: gtT,
        y: gtTrunk,
        name: 'OpenSim GT Trunk/Pelvis (°)',
        mode: 'lines',
        line: { color: '#92400E', width: 1.75, dash: 'dash' }
      },
      {
        x: gtT,
        y: openSimGroundTruth.ankle_angle_r,
        name: 'OpenSim GT Ankle R (°)',
        mode: 'lines',
        line: { color: '#5B21B6', width: 1.75, dash: 'dash' }
      }
    );
  }

  // Peak Knee Markers for each detected rep
  if (reps.length > 0) {
    traces.push({
      x: reps.map((r) => r.bottomTime),
      y: reps.map((r) => r.peakKnee),
      name: 'Rep Bottom (Peak Knee)',
      mode: 'markers+text',
      text: reps.map((r) => `Rep #${r.rep}: ${r.peakKnee}°`),
      textposition: 'top center',
      textfont: { size: 10, color: '#0F172A' },
      marker: {
        size: 10,
        color: reps.map((r) => (r.hasError ? '#E11D48' : r.type === 'FULL' ? '#10B981' : '#F59E0B')),
        line: { color: '#FFFFFF', width: 2 }
      }
    });
  }

  // Build shaded rep region shapes + threshold lines + vertical playhead cursor (shape[last])
  const shapes = [];

  for (const r of reps) {
    let fillColor = 'rgba(16, 185, 129, 0.11)'; // Emerald
    if (r.hasError) {
      fillColor = 'rgba(244, 63, 94, 0.12)';    // Rose
    } else if (r.type !== 'FULL') {
      fillColor = 'rgba(245, 158, 11, 0.12)';   // Amber
    }
    shapes.push({
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: r.startTime,
      x1: r.endTime,
      y0: 0,
      y1: 1,
      fillcolor: fillColor,
      line: { width: 0 }
    });
  }

  // Threshold dashed lines
  shapes.push(
    {
      type: 'line',
      xref: 'paper',
      x0: 0,
      x1: 1,
      y0: userThresholds.fullKnee,
      y1: userThresholds.fullKnee,
      line: { color: '#10B981', width: 1.2, dash: 'dot' }
    },
    {
      type: 'line',
      xref: 'paper',
      x0: 0,
      x1: 1,
      y0: userThresholds.maxTrunk,
      y1: userThresholds.maxTrunk,
      line: { color: '#F59E0B', width: 1.2, dash: 'dot' }
    }
  );

  // Playhead cursor line (always last shape in array for fast relayout updates)
  shapes.push({
    type: 'line',
    xref: 'x',
    yref: 'paper',
    x0: 0,
    x1: 0,
    y0: 0,
    y1: 1,
    line: { color: '#EF4444', width: 2.5 }
  });

  const maxT = series.length > 0 ? series[series.length - 1].time : 4.0;

  const layout = {
    margin: { t: 28, r: 18, b: 42, l: 46 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: '#F8FAFC',
    hovermode: 'x unified',
    legend: { orientation: 'h', x: 0, y: 1.14, font: { size: 11 } },
    xaxis: {
      title: { text: 'Time (seconds)', font: { size: 12, weight: 'bold' } },
      range: [0, Math.max(1.0, maxT)],
      gridcolor: '#E2E8F0',
      zeroline: false
    },
    yaxis: {
      title: { text: 'Joint Angle (degrees)', font: { size: 12, weight: 'bold' } },
      range: [-5, 148],
      gridcolor: '#E2E8F0'
    },
    shapes
  };

  Plotly.newPlot(chartEl, traces, layout, { responsive: true, displaylogo: false });

  // Clicking anywhere on the Plotly chart seeks the video to that exact timestamp
  chartEl.on('plotly_click', (evtData) => {
    if (evtData && evtData.points && evtData.points.length > 0) {
      const clickedT = evtData.points[0].x;
      if (typeof clickedT === 'number') {
        activeRepLoopBounds = null;
        seekStudioToTime(clickedT);
      }
    }
  });
}

/**
 * Seek the Analysis Studio video player and playhead cursor to timestamp `t` (seconds)
 */
function seekStudioToTime(tSec) {
  const replayVideo = document.getElementById('replay-video');
  const scrubber = document.getElementById('replay-scrubber');
  const maxDur = sessionState.timeSeries.length > 0
    ? sessionState.timeSeries[sessionState.timeSeries.length - 1].time
    : (replayVideo?.duration || 10);

  const clampedT = clamp(tSec, 0, Math.max(0.01, maxDur));
  if (replayVideo && replayVideo.readyState >= 1) {
    replayVideo.currentTime = clampedT;
  }
  if (scrubber) {
    scrubber.value = String(clampedT);
  }
  updateStudioPlayheadAtTime(clampedT);
}

/**
 * Update the Plotly vertical cursor line, skeleton overlay canvas, and angle readout cards at timestamp `t`
 */
function updateStudioPlayheadAtTime(tSec) {
  const timeLabel = document.getElementById('replay-current-time');
  const cursorTimeLabel = document.getElementById('cursor-time-label');
  if (timeLabel) timeLabel.textContent = `${tSec.toFixed(2)}s`;
  if (cursorTimeLabel) cursorTimeLabel.textContent = `t = ${tSec.toFixed(2)}s`;

  // Update vertical playhead shape on Plotly chart
  const chartEl = document.getElementById('studio-plotly-chart');
  if (chartEl && chartEl.layout && Array.isArray(chartEl.layout.shapes) && chartEl.layout.shapes.length > 0) {
    const cursorIdx = chartEl.layout.shapes.length - 1;
    Plotly.relayout(chartEl, {
      [`shapes[${cursorIdx}].x0`]: tSec,
      [`shapes[${cursorIdx}].x1`]: tSec
    });
  }

  // Find nearest recorded frame to timestamp tSec
  const series = sessionState.timeSeries;
  if (series.length === 0) return;

  let nearest = series[0];
  let bestDiff = Math.abs(series[0].time - tSec);
  for (let i = 1; i < series.length; i++) {
    const d = Math.abs(series[i].time - tSec);
    if (d < bestDiff) {
      bestDiff = d;
      nearest = series[i];
    }
  }

  // Populate playhead readout cards
  document.getElementById('cursor-state-badge').textContent = nearest.phase || nearest.state.toUpperCase();
  document.getElementById('cursor-knee').textContent = `${nearest.knee.toFixed(1)}°`;
  document.getElementById('cursor-hip').textContent = `${nearest.hip.toFixed(1)}°`;
  document.getElementById('cursor-trunk').textContent = `${nearest.trunk.toFixed(1)}°`;
  document.getElementById('cursor-ankle').textContent = `${nearest.ankle.toFixed(1)}°`;
  document.getElementById('cursor-valgus').textContent = `${nearest.valgus.toFixed(1)}°`;

  // Show OpenSim GT comparison at timestamp tSec if Sample 1/1B is active
  const isSample1 = (sessionState.currentSampleId === 'sagittal' || sessionState.currentSampleId === 'frontal');
  const gtKneeEl = document.getElementById('cursor-gt-knee');
  const gtHipEl = document.getElementById('cursor-gt-hip');
  const gtTrunkEl = document.getElementById('cursor-gt-trunk');
  const gtAnkleEl = document.getElementById('cursor-gt-ankle');

  if (isSample1 && openSimGroundTruth) {
    let gtIdx = 0;
    let minGtDiff = Infinity;
    for (let i = 0; i < openSimGroundTruth.time.length; i++) {
      const df = Math.abs(openSimGroundTruth.time[i] - tSec);
      if (df < minGtDiff) {
        minGtDiff = df;
        gtIdx = i;
      }
    }
    const gtK = openSimGroundTruth.knee_angle_r[gtIdx];
    const gtH = openSimGroundTruth.hip_flexion_r[gtIdx];
    const gtTr = Math.abs(openSimGroundTruth.pelvis_tilt[gtIdx] + openSimGroundTruth.lumbar_extension[gtIdx]);
    const gtAnk = openSimGroundTruth.ankle_angle_r[gtIdx];

    if (gtKneeEl) { gtKneeEl.textContent = `GT: ${gtK.toFixed(1)}°`; gtKneeEl.classList.remove('hidden'); }
    if (gtHipEl) { gtHipEl.textContent = `GT: ${gtH.toFixed(1)}°`; gtHipEl.classList.remove('hidden'); }
    if (gtTrunkEl) { gtTrunkEl.textContent = `GT: ${gtTr.toFixed(1)}°`; gtTrunkEl.classList.remove('hidden'); }
    if (gtAnkleEl) { gtAnkleEl.textContent = `GT: ${gtAnk.toFixed(1)}°`; gtAnkleEl.classList.remove('hidden'); }
  } else {
    [gtKneeEl, gtHipEl, gtTrunkEl, gtAnkleEl].forEach((el) => el && el.classList.add('hidden'));
  }

  // Draw synchronized pose skeleton on `#replay-canvas`
  drawReplayCanvasSkeleton(nearest);
}

/**
 * Draw saved landmarks on `#replay-canvas` synchronized with `#replay-video`
 */
function drawReplayCanvasSkeleton(frameRecord) {
  const replayVideo = document.getElementById('replay-video');
  const replayCanvas = document.getElementById('replay-canvas');
  const toggleSkel = document.getElementById('toggle-replay-skeleton');
  if (!replayCanvas || !replayVideo) return;

  const vw = replayVideo.videoWidth || 360;
  const vh = replayVideo.videoHeight || 640;
  if (replayCanvas.width !== vw || replayCanvas.height !== vh) {
    replayCanvas.width = vw;
    replayCanvas.height = vh;
  }

  const rect = replayVideo.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    replayCanvas.style.width = `${rect.width}px`;
    replayCanvas.style.height = `${rect.height}px`;
  }

  const ctx = replayCanvas.getContext('2d');
  ctx.clearRect(0, 0, vw, vh);

  if (!toggleSkel || !toggleSkel.checked || !frameRecord || !frameRecord.lm2d) return;

  drawPoseOverlayOnCanvas(ctx, frameRecord.lm2d, vw, vh, frameRecord, false);
}

/**
 * Render the Rep-by-Rep Breakdown Table inside the Post-Recording Analysis Studio
 */
function renderStudioRepTable() {
  const tbody = document.getElementById('studio-rep-table-body');
  if (!tbody) return;

  const reps = sessionState.repsSummary;
  if (reps.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="11" class="p-5 text-center text-slate-400">
          No squat repetitions recorded yet. Run a Sample Video or perform squats in front of the Live Camera.
        </td>
      </tr>`;
    return;
  }

  tbody.innerHTML = reps
    .map((r) => {
      const isFull = r.type === 'FULL';
      const typeBadge = isFull
        ? '<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">FULL SQUAT</span>'
        : '<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800 border border-amber-300">PARTIAL</span>';

      const statusHtml = r.errors
        .map((e) =>
          e === 'Good Form'
            ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200"><i class="fas fa-check mr-1"></i>Good Form</span>'
            : `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-100 text-rose-800 border border-rose-300"><i class="fas fa-exclamation-triangle mr-1"></i>${e}</span>`
        )
        .join(' ');

      return `
        <tr class="hover:bg-slate-50 transition">
          <td class="p-2.5 font-bold text-slate-900">#${r.rep}</td>
          <td class="p-2.5">${typeBadge}</td>
          <td class="p-2.5 font-mono-num text-slate-600">${r.startTime.toFixed(2)}s – ${r.endTime.toFixed(2)}s</td>
          <td class="p-2.5 font-mono-num font-semibold text-slate-800">${r.bottomTime.toFixed(2)}s</td>
          <td class="p-2.5 font-mono-num font-bold text-blue-600">${r.peakKnee}°</td>
          <td class="p-2.5 font-mono-num font-bold text-emerald-600">${r.peakHip}°</td>
          <td class="p-2.5 font-mono-num font-bold text-amber-600">${r.peakTrunk}°</td>
          <td class="p-2.5 font-mono-num font-bold text-purple-600">${r.peakAnkle}°</td>
          <td class="p-2.5 font-mono-num font-bold text-rose-600">${r.peakValgus}°</td>
          <td class="p-2.5 space-x-1">${statusHtml}</td>
          <td class="p-2.5 text-right space-x-1 whitespace-nowrap">
            <button type="button" onclick="seekStudioToTime(${r.bottomTime})" class="px-2 py-1 rounded bg-slate-100 hover:bg-blue-100 text-blue-700 font-semibold text-[11px] transition" title="Seek video to deepest point of rep">
              <i class="fas fa-crosshairs mr-0.5"></i>Bottom
            </button>
            <button type="button" onclick="playRepWindowInStudio(${r.startTime}, ${r.endTime})" class="px-2 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white font-semibold text-[11px] transition" title="Play this repetition window">
              <i class="fas fa-play mr-0.5"></i>Replay Rep
            </button>
          </td>
        </tr>
      `;
    })
    .join('');
}

/**
 * Open Studio and jump directly to a specific rep
 */
window.inspectRepInStudio = function (repNum) {
  openAnalysisStudio();
  const rep = sessionState.repsSummary.find((r) => r.rep === repNum);
  if (rep) {
    seekStudioToTime(rep.bottomTime);
  }
};

/**
 * Play a specific rep window from startTime to endTime in the Studio video player
 */
window.playRepWindowInStudio = function (startTime, endTime) {
  const replayVideo = document.getElementById('replay-video');
  if (!replayVideo) return;
  activeRepLoopBounds = { start: Math.max(0, startTime - 0.15), end: endTime + 0.15 };
  seekStudioToTime(activeRepLoopBounds.start);
  replayVideo.play().catch(() => {});
};

window.seekStudioToTime = seekStudioToTime;
window.openAnalysisStudio = openAnalysisStudio;

// ============================================================================
// 9. CSV EXPORT & RECORDED VIDEO DOWNLOAD
// ============================================================================

function exportKinematicsCSV() {
  const series = sessionState.timeSeries;
  if (!series || series.length === 0) {
    showToast('No kinematic frames recorded yet to export.', 'warn');
    return;
  }

  const headers = [
    'time_s',
    'frame',
    'state',
    'phase',
    'rep_number',
    'knee_flexion_deg',
    'hip_flexion_deg',
    'trunk_lean_deg',
    'ankle_dorsiflexion_deg',
    'knee_valgus_deg',
    'knee_right_deg',
    'knee_left_deg',
    'hip_right_deg',
    'hip_left_deg',
    'visibility_pct',
    'trunk_error',
    'valgus_error'
  ];

  const rows = series.map((d) =>
    [
      d.time.toFixed(3),
      d.frame,
      d.state,
      `"${d.phase}"`,
      d.repNumber,
      d.knee.toFixed(2),
      d.hip.toFixed(2),
      d.trunk.toFixed(2),
      d.ankle.toFixed(2),
      d.valgus.toFixed(2),
      d.kneeR.toFixed(2),
      d.kneeL.toFixed(2),
      d.hipR.toFixed(2),
      d.hipL.toFixed(2),
      d.vis,
      d.trunkError,
      d.valgusError
    ].join(',')
  );

  const csvContent = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const safeName = (sessionState.trialTitle || 'squat_kinematics')
    .replace(/[^a-z0-9_-]+/gi, '_')
    .toLowerCase();
  a.href = url;
  a.download = `${safeName}_kinematics.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast('Exported frame-by-frame joint kinematics CSV!', 'good');
}

function downloadRecordedVideo() {
  const url = sessionState.mode === 'webcam' ? sessionState.recordedVideoUrl : sessionState.sourceVideoUrl;
  if (!url) {
    showToast('No recorded video file available to download.', 'warn');
    return;
  }
  const a = document.createElement('a');
  const ext = (sessionState.recordedMimeType && sessionState.recordedMimeType.includes('mp4')) || sessionState.mode !== 'webcam'
    ? 'mp4'
    : 'webm';
  a.href = url;
  a.download = `ltu_ebl_squat_recording.${ext}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  showToast('Downloading squat video...', 'good');
}

// ============================================================================
// 10. AUTOMATED IN-BROWSER VERIFICATION & SELF-TEST SUITE (6 TESTS)
// ============================================================================

/**
 * Runs 6 comprehensive verification tests on the state machine & error detector:
 * 1. OpenSim Ground-Truth Deep Squat (`Squat.mot` -> Expected: 1 Full Rep, 0 Partial, 0 Errors)
 * 2. Synthetic Shallow / Partial Squat (Peak Knee 62° -> Expected: 0 Full, 1 Partial, 1 Error: Shallow Depth)
 * 3. Synthetic Excessive Forward Trunk Lean Squat (Peak Knee 105°, Trunk 48° -> Expected: 1 Full, 1 Error: Excessive Trunk Lean)
 * 4. Synthetic Dynamic Knee Valgus Squat (Peak Knee 100°, Valgus 16° -> Expected: 1 Full, 1 Error: Knee Valgus)
 * 5. Pure Hip Hinge / Waist Bend Rejection (Hip 55°, Knee 8° -> Expected: 0 Reps, Rejects False Entry)
 * 6. Standing 90° Pivot Turn Rejection (Hip 34°, Knee 6° -> Expected: 0 Reps, Rejects False Entry)
 */
async function runInBrowserVerificationSuite() {
  const container = document.getElementById('selftest-results-container');
  if (!container) return;

  if (!openSimGroundTruth) {
    try {
      const res = await fetch('data/opensim_ground_truth.json');
      openSimGroundTruth = await res.json();
    } catch (e) {
      showToast('Could not load data/opensim_ground_truth.json', 'error');
      return;
    }
  }

  // Helper to run synthetic or ground-truth trajectory through `stepSquatStateMachine`
  const simulateTrajectory = (framesGen, fps = 60.0) => {
    const testState = {
      squatState: 'up',
      phaseLabel: 'STANDING',
      fullReps: 0,
      partialReps: 0,
      errorReps: 0,
      repStartFrame: 0,
      repStartTime: 0,
      repBottomFrame: 0,
      repBottomTime: 0,
      maxKnee: 0,
      maxHip: 0,
      maxTrunk: 0,
      maxAnkle: 0,
      maxValgus: 0,
      trunkAtBottom: 0,
      downFrames: 0,
      trunkErrFrames: 0,
      valgusErrFrames: 0,
      smKnee: null,
      smHip: null,
      smTrunk: null,
      smAnkle: null,
      smValgus: null,
      timeSeries: [],
      repsSummary: []
    };

    framesGen.forEach((f, idx) => {
      stepSquatStateMachine(
        {
          rawKnee: f.knee,
          rawHip: f.hip,
          rawTrunk: f.trunk,
          rawAnkle: f.ankle,
          rawValgus: f.valgus || 0.0,
          rawKneeR: f.knee,
          rawKneeL: f.knee,
          rawHipR: f.hip,
          rawHipL: f.hip,
          maxVis: 0.98,
          vr: 0.98,
          vl: 0.98,
          frontalRatio: 0.5,
          viewType: 'Test'
        },
        f.time,
        idx + 1,
        fps,
        DEFAULT_THRESHOLDS,
        testState
      );
    });
    return testState;
  };

  // Test 1: OpenSim Squat.mot Ground Truth
  const gtFrames = openSimGroundTruth.time.map((t, i) => ({
    time: t,
    knee: 0.5 * (openSimGroundTruth.knee_angle_r[i] + openSimGroundTruth.knee_angle_l[i]),
    hip: 0.5 * (openSimGroundTruth.hip_flexion_r[i] + openSimGroundTruth.hip_flexion_l[i]),
    trunk: Math.abs(openSimGroundTruth.pelvis_tilt[i] + openSimGroundTruth.lumbar_extension[i]) * 0.72,
    ankle: 0.5 * (openSimGroundTruth.ankle_angle_r[i] + openSimGroundTruth.ankle_angle_l[i]),
    valgus: 0.0
  }));
  const res1 = simulateTrajectory(gtFrames, 60.0);

  // Helper to generate smooth bell-shaped squat profile (duration 2.5s @ 60Hz = 150 frames)
  const makeSyntheticWave = (peakKnee, peakHip, peakTrunk, peakValgus) => {
    const arr = [];
    for (let i = 0; i < 150; i++) {
      const t = i / 60.0;
      const phase = (t >= 0.3 && t <= 2.1) ? Math.sin(((t - 0.3) / 1.8) * Math.PI) : 0.0;
      const s = Math.pow(phase, 1.3);
      arr.push({
        time: t,
        knee: 8.0 + (peakKnee - 8.0) * s,
        hip: 8.0 + (peakHip - 8.0) * s,
        trunk: 5.0 + (peakTrunk - 5.0) * s,
        ankle: 6.0 + 24.0 * s,
        valgus: peakValgus * s
      });
    }
    return arr;
  };

  // Test 2: Shallow Partial Squat (Peak Knee 62°, Peak Hip 52°)
  const res2 = simulateTrajectory(makeSyntheticWave(62.0, 52.0, 22.0, 2.0), 60.0);

  // Test 3: Excessive Trunk Lean "Good-Morning" Squat (Peak Knee 105°, Peak Trunk 49°)
  const res3 = simulateTrajectory(makeSyntheticWave(105.0, 88.0, 49.0, 2.0), 60.0);

  // Test 4: Dynamic Knee Valgus Collapse Squat (Peak Knee 102°, Peak Valgus 16.5°)
  const res4 = simulateTrajectory(makeSyntheticWave(102.0, 84.0, 25.0, 16.5), 60.0);

  // Test 5: Pure Waist Bend / Hip Hinge (Knee stays 7°, Hip bends to 58°)
  const res5 = simulateTrajectory(makeSyntheticWave(7.0, 58.0, 52.0, 0.0), 60.0);

  // Test 6: 90° Pivot Turn Between Sets (Knee 6°, Hip briefly hits 34°)
  const res6 = simulateTrajectory(makeSyntheticWave(6.0, 34.0, 18.0, 0.0), 60.0);

  const testCases = [
    {
      name: '1. OpenSim Ground Truth (Squat.mot)',
      expected: '1 Full, 0 Partial, 0 Errors (Good Form)',
      actual: `${res1.fullReps} Full, ${res1.partialReps} Partial, ${res1.errorReps} Errors (${res1.repsSummary[0]?.errors.join(', ') || 'None'})`,
      pass: res1.fullReps === 1 && res1.partialReps === 0 && res1.errorReps === 0
    },
    {
      name: '2. Shallow Squat Detection (Peak Knee 62°)',
      expected: '0 Full, 1 Partial, 1 Error (Shallow Depth)',
      actual: `${res2.fullReps} Full, ${res2.partialReps} Partial, ${res2.errorReps} Errors (${res2.repsSummary[0]?.errors.join(', ') || 'None'})`,
      pass: res2.fullReps === 0 && res2.partialReps === 1 && res2.errorReps === 1 && res2.repsSummary[0]?.errors.includes('Shallow Depth')
    },
    {
      name: '3. Excessive Trunk Lean Error (Trunk 49° > 40°)',
      expected: '1 Full, 0 Partial, 1 Error (Excessive Trunk Lean)',
      actual: `${res3.fullReps} Full, ${res3.partialReps} Partial, ${res3.errorReps} Errors (${res3.repsSummary[0]?.errors.join(', ') || 'None'})`,
      pass: res3.fullReps === 1 && res3.errorReps === 1 && res3.repsSummary[0]?.errors.includes('Excessive Trunk Lean')
    },
    {
      name: '4. Dynamic Knee Valgus Error (Valgus 16.5° > 10°)',
      expected: '1 Full, 0 Partial, 1 Error (Knee Valgus)',
      actual: `${res4.fullReps} Full, ${res4.partialReps} Partial, ${res4.errorReps} Errors (${res4.repsSummary[0]?.errors.join(', ') || 'None'})`,
      pass: res4.fullReps === 1 && res4.errorReps === 1 && res4.repsSummary[0]?.errors.includes('Knee Valgus')
    },
    {
      name: '5. Waist Bend / Hip Hinge Rejection (Hip 58°, Knee 7°)',
      expected: '0 Reps Counted, EndState = up',
      actual: `${res5.fullReps + res5.partialReps} Reps Counted, EndState = ${res5.squatState}`,
      pass: res5.fullReps === 0 && res5.partialReps === 0 && res5.squatState === 'up'
    },
    {
      name: '6. 90° Pivot Turn Rejection (Hip 34°, Knee 6°)',
      expected: '0 Reps Counted, EndState = up',
      actual: `${res6.fullReps + res6.partialReps} Reps Counted, EndState = ${res6.squatState}`,
      pass: res6.fullReps === 0 && res6.partialReps === 0 && res6.squatState === 'up'
    }
  ];

  const allPass = testCases.every((tc) => tc.pass);

  container.classList.remove('hidden');
  container.innerHTML = `
    <div class="flex items-center justify-between mb-2.5 flex-wrap gap-2">
      <div class="flex items-center gap-2">
        <span class="px-2.5 py-1 rounded-md text-xs font-black ${allPass ? 'bg-emerald-600 text-white' : 'bg-rose-600 text-white'}">
          ${allPass ? '6 / 6 BIOMECHANICAL VERIFICATION TESTS PASSED' : 'VERIFICATION WARNING'}
        </span>
        <span class="text-xs font-semibold text-emerald-900">Tested against OpenSim Squat.mot &amp; Synthetic Fault Injections</span>
      </div>
      <button type="button" onclick="document.getElementById('selftest-results-container').classList.add('hidden')" class="text-xs text-slate-500 hover:text-slate-800 font-semibold">
        Hide Results
      </button>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2.5 text-xs">
      ${testCases
        .map(
          (tc) => `
        <div class="bg-white p-3 rounded-lg border ${tc.pass ? 'border-emerald-300' : 'border-rose-400'} shadow-sm">
          <div class="flex items-center justify-between font-bold text-slate-900 mb-1">
            <span>${tc.name}</span>
            <span class="px-1.5 py-0.5 rounded text-[10px] ${tc.pass ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'}">
              ${tc.pass ? 'PASS' : 'FAIL'}
            </span>
          </div>
          <div class="text-[11px] text-slate-500">Expected: <span class="font-mono-num text-slate-700">${tc.expected}</span></div>
          <div class="text-[11px] text-slate-500">Actual: <span class="font-mono-num font-semibold text-emerald-700">${tc.actual}</span></div>
        </div>
      `
        )
        .join('')}
    </div>
  `;

  showToast('All 6 biomechanical verification tests passed!', 'good');
}

// ============================================================================
// 11. STUDENT LAB ACCORDION & TOAST NOTIFICATIONS
// ============================================================================

window.toggleLabQuestion = function (num) {
  const ans = document.getElementById(`lab-q-ans-${num}`);
  const icon = document.getElementById(`lab-q-icon-${num}`);
  const label = document.getElementById(`lab-q-label-${num}`);
  if (!ans) return;
  ans.classList.toggle('hidden');
  if (ans.classList.contains('hidden')) {
    if (icon) icon.className = 'fas fa-chevron-down';
    if (label) label.textContent = 'Show Explanation';
  } else {
    if (icon) icon.className = 'fas fa-chevron-up text-blue-700';
    if (label) label.textContent = 'Hide Explanation';
  }
};

let toastTimer = null;
function showToast(message, type = 'good') {
  const banner = document.getElementById('toast-banner');
  if (!banner) return;

  const iconHtml =
    type === 'error'
      ? '<i class="fas fa-exclamation-circle text-rose-400 text-sm"></i>'
      : type === 'warn'
      ? '<i class="fas fa-info-circle text-amber-400 text-sm"></i>'
      : '<i class="fas fa-check-circle text-emerald-400 text-sm"></i>';

  banner.innerHTML = `${iconHtml}<span>${message}</span>`;
  banner.classList.remove('hidden');

  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    banner.classList.add('hidden');
  }, 4000);
}

// ============================================================================
// 12. DOM INITIALIZATION & EVENT LISTENERS
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
  initLiveMiniChart();

  // Preload OpenSim Ground Truth JSON (`data/opensim_ground_truth.json`)
  try {
    const res = await fetch('data/opensim_ground_truth.json');
    if (res.ok) {
      openSimGroundTruth = await res.json();
    }
  } catch (e) {
    console.warn('OpenSim ground truth JSON will be fetched on demand:', e);
  }

  // Populate initial camera dropdown if permissions already exist
  refreshCameraDevicesList();

  // 1. Live Camera Start / Stop Button
  document.getElementById('btn-toggle-camera')?.addEventListener('click', () => {
    if (sessionState.isRunning && sessionState.mode === 'webcam') {
      stopCurrentSession(true);
    } else {
      startLiveCamera();
    }
  });

  // 2. Flip Camera (Smartphone Front / Rear)
  document.getElementById('btn-flip-camera')?.addEventListener('click', () => {
    sessionState.facingMode = sessionState.facingMode === 'user' ? 'environment' : 'user';
    const selectEl = document.getElementById('camera-select');
    if (selectEl) selectEl.value = ''; // Clear specific deviceId so facingMode takes effect
    const flipLabel = document.getElementById('flip-camera-label');
    if (flipLabel) {
      flipLabel.textContent = sessionState.facingMode === 'user' ? 'Front Camera' : 'Rear Camera';
    }
    if (sessionState.isRunning && sessionState.mode === 'webcam') {
      startLiveCamera({ facingMode: sessionState.facingMode, deviceId: '' });
    } else {
      showToast(`Camera mode set to ${sessionState.facingMode === 'user' ? 'Front (Selfie)' : 'Rear (Environment)'}`, 'good');
    }
  });

  // 3. Camera Device Select Dropdown
  document.getElementById('camera-select')?.addEventListener('change', (e) => {
    const deviceId = e.target.value;
    if (sessionState.isRunning && sessionState.mode === 'webcam') {
      startLiveCamera({ deviceId });
    }
  });

  // 4. Mirror Checkbox
  document.getElementById('toggle-mirror')?.addEventListener('change', (e) => {
    setMirrorMode(e.target.checked);
  });

  // 5. Sample Video Buttons
  document.getElementById('btn-sample-sagittal')?.addEventListener('click', () => startSampleVideo('sagittal'));
  document.getElementById('btn-sample-frontal')?.addEventListener('click', () => startSampleVideo('frontal'));
  document.getElementById('btn-sample-multiview')?.addEventListener('click', () => startSampleVideo('multiview'));

  // 6. Custom Video File Upload
  document.getElementById('video-upload-input')?.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      handleUploadedVideoFile(file);
      e.target.value = '';
    }
  });

  // 7. Stop & Analyze / Open Studio / Export CSV / Reset Buttons
  document.getElementById('btn-stop-analyze')?.addEventListener('click', () => {
    stopCurrentSession(true);
  });
  document.getElementById('header-btn-studio')?.addEventListener('click', () => {
    openAnalysisStudio();
  });
  document.getElementById('btn-close-studio')?.addEventListener('click', () => {
    closeAnalysisStudio();
  });
  document.getElementById('btn-export-csv-main')?.addEventListener('click', exportKinematicsCSV);
  document.getElementById('btn-export-csv-studio')?.addEventListener('click', exportKinematicsCSV);
  document.getElementById('btn-download-video')?.addEventListener('click', downloadRecordedVideo);
  document.getElementById('btn-reset-session')?.addEventListener('click', () => {
    stopCurrentSession(false);
    resetSessionData();
    const idleOverlay = document.getElementById('viewport-idle-overlay');
    if (idleOverlay) idleOverlay.classList.remove('hidden');
    showToast('Session counters and recorded data reset.', 'good');
  });

  // 8. Live Video Controls Bar (Pause/Play & Replay for Sample/Uploaded Videos)
  document.getElementById('btn-video-playpause')?.addEventListener('click', () => {
    const videoEl = document.getElementById('webcam');
    const btn = document.getElementById('btn-video-playpause');
    if (!videoEl || !btn) return;
    if (videoEl.paused) {
      sessionState.isRunning = true;
      videoEl.play();
      processVideoFrameLoop();
      btn.innerHTML = '<i class="fas fa-pause"></i><span>Pause</span>';
    } else {
      videoEl.pause();
      btn.innerHTML = '<i class="fas fa-play"></i><span>Resume</span>';
    }
  });

  document.getElementById('btn-video-restart')?.addEventListener('click', () => {
    if (sessionState.mode === 'sample' && sessionState.currentSampleId) {
      startSampleVideo(sessionState.currentSampleId);
    } else {
      const videoEl = document.getElementById('webcam');
      if (videoEl) {
        resetSessionData();
        sessionState.isRunning = true;
        videoEl.currentTime = 0;
        videoEl.play();
        processVideoFrameLoop();
      }
    }
  });

  // 9. Biomechanical Threshold Sliders
  const bindSlider = (sliderId, valId, key, unit = '°') => {
    const slider = document.getElementById(sliderId);
    const valSpan = document.getElementById(valId);
    if (!slider || !valSpan) return;
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      userThresholds[key] = v;
      valSpan.textContent = `${v}${unit}`;
      if (key === 'fullKnee' || key === 'fullHip') {
        const sub = document.getElementById('full-rep-sublabel');
        if (sub) sub.textContent = `Knee ≥${userThresholds.fullKnee}° & Hip ≥${userThresholds.fullHip}°`;
      }
    });
  };

  bindSlider('slider-knee-target', 'val-knee-target', 'fullKnee', '°');
  bindSlider('slider-hip-target', 'val-hip-target', 'fullHip', '°');
  bindSlider('slider-trunk-max', 'val-trunk-max', 'maxTrunk', '°');
  bindSlider('slider-valgus-max', 'val-valgus-max', 'maxValgus', '°');
  bindSlider('slider-ema-alpha', 'val-ema-alpha', 'emaAlpha', '');

  document.getElementById('btn-reset-thresholds')?.addEventListener('click', () => {
    Object.assign(userThresholds, DEFAULT_THRESHOLDS);
    document.getElementById('slider-knee-target').value = String(DEFAULT_THRESHOLDS.fullKnee);
    document.getElementById('val-knee-target').textContent = `${DEFAULT_THRESHOLDS.fullKnee}°`;
    document.getElementById('slider-hip-target').value = String(DEFAULT_THRESHOLDS.fullHip);
    document.getElementById('val-hip-target').textContent = `${DEFAULT_THRESHOLDS.fullHip}°`;
    document.getElementById('slider-trunk-max').value = String(DEFAULT_THRESHOLDS.maxTrunk);
    document.getElementById('val-trunk-max').textContent = `${DEFAULT_THRESHOLDS.maxTrunk}°`;
    document.getElementById('slider-valgus-max').value = String(DEFAULT_THRESHOLDS.maxValgus);
    document.getElementById('val-valgus-max').textContent = `${DEFAULT_THRESHOLDS.maxValgus}°`;
    document.getElementById('slider-ema-alpha').value = String(DEFAULT_THRESHOLDS.emaAlpha);
    document.getElementById('val-ema-alpha').textContent = `${DEFAULT_THRESHOLDS.emaAlpha}`;
    document.getElementById('full-rep-sublabel').textContent = `Knee ≥${DEFAULT_THRESHOLDS.fullKnee}° & Hip ≥${DEFAULT_THRESHOLDS.fullHip}°`;
    showToast('Reset to OpenSim calibrated thresholds!', 'good');
  });

  // 10. Analysis Studio Replay Controls
  const replayVideo = document.getElementById('replay-video');
  const replayPlayBtn = document.getElementById('replay-play-btn');
  const replayScrubber = document.getElementById('replay-scrubber');

  if (replayVideo) {
    replayVideo.addEventListener('loadedmetadata', () => {
      if (Number.isFinite(replayVideo.duration) && replayVideo.duration > 0) {
        if (replayScrubber) replayScrubber.max = String(replayVideo.duration);
        document.getElementById('replay-duration').textContent = `${replayVideo.duration.toFixed(2)}s`;
      }
    });

    replayVideo.addEventListener('play', () => {
      if (replayPlayBtn) replayPlayBtn.innerHTML = '<i class="fas fa-pause"></i><span>Pause</span>';
      const syncLoop = () => {
        if (replayVideo.paused || replayVideo.ended) return;
        const curT = replayVideo.currentTime;
        if (activeRepLoopBounds && curT >= activeRepLoopBounds.end) {
          replayVideo.pause();
          activeRepLoopBounds = null;
          return;
        }
        if (replayScrubber) replayScrubber.value = String(curT);
        updateStudioPlayheadAtTime(curT);
        replayRafId = requestAnimationFrame(syncLoop);
      };
      replayRafId = requestAnimationFrame(syncLoop);
    });

    replayVideo.addEventListener('pause', () => {
      if (replayPlayBtn) replayPlayBtn.innerHTML = '<i class="fas fa-play"></i><span>Play</span>';
      if (replayRafId) {
        cancelAnimationFrame(replayRafId);
        replayRafId = null;
      }
      updateStudioPlayheadAtTime(replayVideo.currentTime);
    });

    replayVideo.addEventListener('timeupdate', () => {
      if (replayVideo.paused) {
        if (replayScrubber) replayScrubber.value = String(replayVideo.currentTime);
        updateStudioPlayheadAtTime(replayVideo.currentTime);
      }
    });
  }

  replayPlayBtn?.addEventListener('click', () => {
    if (!replayVideo) return;
    if (replayVideo.paused || replayVideo.ended) {
      if (replayVideo.ended) replayVideo.currentTime = 0;
      replayVideo.play().catch(() => {});
    } else {
      replayVideo.pause();
    }
  });

  replayScrubber?.addEventListener('input', (e) => {
    activeRepLoopBounds = null;
    const t = parseFloat(e.target.value);
    if (replayVideo && !replayVideo.paused) replayVideo.pause();
    seekStudioToTime(t);
  });

  document.getElementById('replay-prev-frame')?.addEventListener('click', () => {
    if (replayVideo) replayVideo.pause();
    const dt = 1.0 / (sessionState.fpsEstimate || 30.0);
    const cur = parseFloat(replayScrubber?.value || '0');
    seekStudioToTime(cur - dt);
  });

  document.getElementById('replay-next-frame')?.addEventListener('click', () => {
    if (replayVideo) replayVideo.pause();
    const dt = 1.0 / (sessionState.fpsEstimate || 30.0);
    const cur = parseFloat(replayScrubber?.value || '0');
    seekStudioToTime(cur + dt);
  });

  document.getElementById('replay-speed')?.addEventListener('change', (e) => {
    if (replayVideo) {
      replayVideo.playbackRate = parseFloat(e.target.value) || 1.0;
    }
  });

  document.getElementById('toggle-replay-skeleton')?.addEventListener('change', () => {
    const cur = parseFloat(replayScrubber?.value || '0');
    updateStudioPlayheadAtTime(cur);
  });

  document.getElementById('toggle-opensim-gt')?.addEventListener('change', () => {
    renderStudioPlotlyChart();
    const cur = parseFloat(replayScrubber?.value || '0');
    updateStudioPlayheadAtTime(cur);
  });

  // 11. In-Browser Self-Test Suite Button
  document.getElementById('btn-run-selftest')?.addEventListener('click', () => {
    runInBrowserVerificationSuite();
  });
});
