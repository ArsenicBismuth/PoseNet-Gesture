/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';

const color = 'aqua';
const boundingBoxColor = 'red';
const lineWidth = 2;

export const tryResNetButtonName = 'tryResNetButton';
export const tryResNetButtonText = '[New] Try ResNet50';
const tryResNetButtonTextCss = 'width:100%;text-decoration:underline;';
const tryResNetButtonBackgroundCss = 'background:#e61d5f;';

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export function isMobile() {
  return isAndroid() || isiOS();
}

function setDatGuiPropertyCss(propertyText, liCssString, spanCssString = '') {
  var spans = document.getElementsByClassName('property-name');
  for (var i = 0; i < spans.length; i++) {
    var text = spans[i].textContent || spans[i].innerText;
    if (text == propertyText) {
      spans[i].parentNode.parentNode.style = liCssString;
      if (spanCssString !== '') {
        spans[i].style = spanCssString;
      }
    }
  }
}

export function updateTryResNetButtonDatGuiCss() {
  setDatGuiPropertyCss(
      tryResNetButtonText, tryResNetButtonBackgroundCss,
      tryResNetButtonTextCss);
}

/**
 * Toggles between the loading UI and the main canvas UI.
 */
export function toggleLoadingUI(
    showLoadingUI, loadingDivId = 'loading', mainDivId = 'main') {
  if (showLoadingUI) {
    document.getElementById(loadingDivId).style.display = 'block';
    document.getElementById(mainDivId).style.display = 'none';
  } else {
    document.getElementById(loadingDivId).style.display = 'none';
    document.getElementById(mainDivId).style.display = 'block';
  }
}

function toTuple({y, x}) {
  return [y, x];
}

export function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

/**
 * Draws a line on a canvas, i.e. a joint
 */
export function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.stroke();
}

/**
 * Draws a pose skeleton by looking up all adjacent keypoints/joints
 */
export function drawSkeleton(keypoints, minConfidence, ctx, color = color, scale = 1) {
  const adjacentKeyPoints =
      posenet.getAdjacentKeyPoints(keypoints, minConfidence);

  adjacentKeyPoints.forEach((keypoints) => {
    drawSegment(
        toTuple(keypoints[0].position), toTuple(keypoints[1].position), color,
        scale, ctx);
  });
}

/**
 * Draw pose keypoints onto a canvas
 */
export function drawKeypoints(keypoints, minConfidence, ctx, color = color, scale = 1) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];

    if (keypoint.score < minConfidence) {
      continue;
    }

    const {y, x} = keypoint.position;
    drawPoint(ctx, y * scale, x * scale, 3, color);
  }
}

/**
 * Draw the bounding box of a pose. For example, for a whole person standing
 * in an image, the bounding box will begin at the nose and extend to one of
 * ankles
 */
export function drawBoundingBox(keypoints, ctx) {
  const boundingBox = posenet.getBoundingBox(keypoints);

  ctx.rect(
      boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
      boundingBox.maxY - boundingBox.minY);

  ctx.strokeStyle = boundingBoxColor;
  ctx.stroke();
}

/**
 * Converts an arary of pixel data into an ImageData object
 */
export async function renderToCanvas(a, ctx) {
  const [height, width] = a.shape;
  const imageData = new ImageData(width, height);

  const data = await a.data();

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * 3;

    imageData.data[j + 0] = data[k + 0];
    imageData.data[j + 1] = data[k + 1];
    imageData.data[j + 2] = data[k + 2];
    imageData.data[j + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw an image on a canvas
 */
export function renderImageToCanvas(image, size, canvas) {
  canvas.width = size[0];
  canvas.height = size[1];
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
}

/**
 * Draw heatmap values, one of the model outputs, on to the canvas
 * Read our blog post for a description of PoseNet's heatmap outputs
 * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
 */
export function drawHeatMapValues(heatMapValues, outputStride, canvas) {
  const ctx = canvas.getContext('2d');
  const radius = 5;
  const scaledValues = heatMapValues.mul(tf.scalar(outputStride, 'int32'));

  drawPoints(ctx, scaledValues, radius, color);
}

/**
 * Used by the drawHeatMapValues method to draw heatmap points on to
 * the canvas
 */
function drawPoints(ctx, points, radius, color) {
  const data = points.buffer().values;

  for (let i = 0; i < data.length; i += 2) {
    const pointY = data[i];
    const pointX = data[i + 1];

    if (pointX !== 0 && pointY !== 0) {
      ctx.beginPath();
      ctx.arc(pointX, pointY, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }
}

/**
 * Draw offset vector values, one of the model outputs, on to the canvas
 * Read our blog post for a description of PoseNet's offset vector outputs
 * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
 */
export function drawOffsetVectors(
    heatMapValues, offsets, outputStride, scale = 1, ctx) {
  const offsetPoints =
      posenet.singlePose.getOffsetPoints(heatMapValues, outputStride, offsets);

  const heatmapData = heatMapValues.buffer().values;
  const offsetPointsData = offsetPoints.buffer().values;

  for (let i = 0; i < heatmapData.length; i += 2) {
    const heatmapY = heatmapData[i] * outputStride;
    const heatmapX = heatmapData[i + 1] * outputStride;
    const offsetPointY = offsetPointsData[i];
    const offsetPointX = offsetPointsData[i + 1];

    drawSegment(
        [heatmapY, heatmapX], [offsetPointY, offsetPointX], color, scale, ctx);
  }
}


//// Data processing

const distTh = 100; // Minimum gesture distance
const gestN = 5;    // Max poses in a gesture
const nullPose = {
  "score":0, 
  "center":{"x":0,"y":0},
  "keypoints":[
    {"score":0,"part":"nose","position":{"x":0,"y":0}},
    {"score":0,"part":"leftEye","position":{"x":0,"y":0}},
    {"score":0,"part":"rightEye","position":{"x":0,"y":0}},
    {"score":0,"part":"leftEar","position":{"x":0,"y":0}},
    {"score":0,"part":"rightEar","position":{"x":0,"y":0}},
    {"score":0,"part":"leftShoulder","position":{"x":0,"y":0}},
    {"score":0,"part":"rightShoulder","position":{"x":0,"y":0}},
    {"score":0,"part":"leftElbow","position":{"x":0,"y":0}},
    {"score":0,"part":"rightElbow","position":{"x":0,"y":0}},
    {"score":0,"part":"leftWrist","position":{"x":0,"y":0}},
    {"score":0,"part":"rightWrist","position":{"x":0,"y":0}},
    {"score":0,"part":"leftHip","position":{"x":0,"y":0}},
    {"score":0,"part":"rightHip","position":{"x":0,"y":0}},
    {"score":0,"part":"leftKnee","position":{"x":0,"y":0}},
    {"score":0,"part":"rightKnee","position":{"x":0,"y":0}},
    {"score":0,"part":"leftAnkle","position":{"x":0,"y":0}},
    {"score":0,"part":"rightAnkle","position":{"x":0,"y":0}}
  ]
}

/**
 * Tracking - Match poses into respective gestures.
 * 1 gesture can only pair with 1 pose max. Two methods: complete and shallow.
 * - Complete : check every combinations, filter by threshold, sort thru the list, get nearest P-G, remove pairs with claimed components (P or G), repeat.
 * - Shallow  : for a G find nearest P, filter by threshold, remove claimed P from candidacy, repeat.
 */
export function matchPoses(gestures, poses) {
  // Store pairing results
  let dists = [];
  // To track of which is claimed
  let done = { g:[], p:[] };
  let p = 0;
  
  // Calculate distance between every pose (p) and gest (g)
  poses.forEach(pose => {
    let g = 0;
    
    gestures.forEach(gest => {
      // Reference: latest value (head)
      const prev = gest.a[0];
      
      // Matching pose - Distance
      const x = pose.center.x - prev.center.x;
      const y = pose.center.y - prev.center.y;
      const dist = Math.sqrt(x*x + y*y);
      
      // Minimum distance
      if (dist < distTh) dists.push([dist, g, p]);
      g++;
    });
    
    p++;
  });
  
  // Pair pose and gesture based on result, start from closest pair
  dists.sort((a, b) => {
    return a[0] - b[0];
  });
  
  console.log(dists);
  
  dists.forEach(d => {
    // Skip through matched components (eq to continue)
    if (done.g.includes(d[1])) return;
    if (done.p.includes(d[2])) return;
    
    // Pair found, add to head
    gestures[d[1]].a.unshift(poses[d[2]]);
    gestures[d[1]].a.pop();  // Remove last element
    done.g.push(d[1]);
    done.p.push(d[2]);
  });
  
  // No pair found
  // Gesture delete
  for (let i=0; i<gestures.length; i++) {
    if (!done.g.includes(i)) gestures.splice(i,1);
  }
  
  // Pose create new gesture
  for (let i=0; i<poses.length; i++) {
    if (!done.p.includes(i)) {
      // Latest data is head, so add fillers to tail
      let a = [poses[i]];
      for (let n=0; n<gestN-1; n++) a.push(nullPose);
      
      gestures.push({
        color: getColor(Math.random()),
        a: a  // Array of poses
      });
      
    }
  }
  
}

/**
 * Select random color using golden angle & hsl, to distinguish different poses.
 */
function getColor(rand) {
  const hue = rand * 137.508; // use golden angle approximation
  return `hsl(${hue},100%,50%)`;
}

/**
 * Process the pose directly after being inferred
 */
export function preprocPose(pose, minConfidence, remove = []) {
  // Modify object, not copy. Must not do any reassigment (pose = new object).
  removeKeypoints(pose, remove);
  zeroKeypoints(pose, minConfidence);
  centerPose(pose);
}

/**
 * Remove unused keypoints by setting them to 0
 */
function removeKeypoints(pose, remove = []) {
  remove.forEach(i => {
    pose.keypoints[i].score = 0;
  });
}

/**
 * Zero low-confidence keypoints by setting them to (0,0)
 */
function zeroKeypoints(pose, minConfidence) {
  pose.keypoints.forEach(({score, position}) => {
    if (score < minConfidence) {
      position.x = 0;
      position.y = 0;
    }
  });
}

/**
 * Find single-point representation of a pose - Between shoulders
 */
function centerPose(pose) {
  let center = {x:0, y:0};
  let points = [pose.keypoints[5], pose.keypoints[6]]; // Left & right shoulders
    
  if (points[0].score > 0 && points[1].score > 0) {
    center.x = points[0].position.x + (points[1].position.x - points[0].position.x)/2;
    center.y = points[0].position.y + (points[1].position.y - points[0].position.y)/2;
  } else {
    // Through median
    center = medianPose(pose);
  }
  
  pose.center = center;
}

/**
 * Find single-point representation of a pose - Median
 */
function medianPose(pose) {
  let center = {x:0, y:0};
  let x = [];
  let y = [];
  
  pose.keypoints.forEach(({score, position}) => {
    if (score > 0) {
      x.push(position.x);
      y.push(position.y);
    }
  });
  
  // Ascending sort
  x.sort((a, b) => {return a - b});
  y.sort((a, b) => {return a - b});
  
  // Median
  const i = x.length;
  if (i % 2 == 0) {
    center.x = (x[i/2] + x[(i+2)/2])/2;
    center.y = (y[i/2] + y[(i+2)/2])/2;
  } else {
    center.x = x[(i+1)/2];
    center.y = y[(i+1)/2];
  }
  
  return center;
}