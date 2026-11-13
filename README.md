# X-Ray Mirror

Webcam X-ray mirror: a real 3D skull, jaw and brain follow your head; hand bones follow your hands. Video never leaves the device.

## Run
```
npm install
npm run dev        # open the printed https URL on your phone/laptop
```
Camera access needs https or localhost. Deploy on Vercel: import the repo, framework preset **Vite**, no settings needed.

## Structure
- `src/main.js` – tracking (MediaPipe Tasks Vision), head pose, Three.js scene, labels, hand overlay
- `public/models/anatomy.glb` – all anatomy in one file (skull, jaw, teeth, brain regions, eyes, eye/palate muscles, tongue, glands, neck bones)
- `public/models/labels.json` – label names with 3D anchor points
- `public/models/torso.glb` + `torso.json` – ribs, spine, shoulders, pelvis, heart, vessels, airways, liver, stomach, kidneys, diaphragm (same head space as anatomy.glb); torso.json holds the labels and the shoulder/hip anchor points
- `tools/build_torso.py` – rebuilds torso.glb/torso.json from the torso .obj files (`--lists`, `--obj`, `--out`; needs python3 + numpy)
- `tools/build_models.py` – rebuilds both from the BodyParts3D .obj files (`--lists`, `--obj`, `--out`; needs python3 + numpy)
- Models are in head space (millimetres, nasion at origin, +Y up, +Z forward). Jaw parts ride on a hinge pivot.

## Torso
The body is tracked with MediaPipe Pose (lite): shoulders set size, position and tilt; hips (when visible) set torso length. Stand back so your shoulders are in frame. Use Torso size / length / up-down to fine-tune. The pose tracker only runs while Torso bones or Torso organs is on. The release has no solid lung or spleen models, so the lungs show as the bronchial tree.

## Tuning the fit
Fit size / Up-down / In-out sliders. Once you find values you like, change the defaults in `S` (top of `main.js`).

## Notes
- MediaPipe wasm loads from jsDelivr and the .task models from Google's storage. To self-host, download them into `public/mediapipe/` and change the URLs at the top of `main.js`.
- Anatomy: BodyParts3D, © DBCLS, CC BY-SA 2.1 Japan. Keep this credit; derived models must stay CC BY-SA.

## Muscles, arms and 3D hands
- `public/models/muscles.glb` + `muscles.json` – chest/back/rib/neck muscles, upper-arm and forearm muscles, humerus/radius/ulna (built by `tools/build_muscles.py`; `--ids` writes the list of .obj files to collect).
- `public/models/hands.glb` + `hands.json` – 3D hand bones (carpals, metacarpals, phalanges), posed bone by bone from the MediaPipe hand landmarks (`tools/build_hands.py`).
- Arms hang off the shoulder joints from the pose tracker (shoulder → elbow → wrist). If hands.glb fails to load the app falls back to the flat 2D hands.
- This BodyParts3D release lacks the trapezius transverse part, rhomboids, scalenes, SCM and deep back muscles, the trapezoid carpal, face muscles, rectus abdominis and latissimus dorsi.

## Tracking, fit and capture
- Landmarks go through a One-Euro filter (steady at rest, quick when moving). The Smoothing slider trades steadiness for speed; Tracking: Smooth runs the body and hand trackers every frame, Fast runs them every second frame.
- Arm direction comes from the image x/y; depth comes from how foreshortened each segment is (the tracker's own depth is used only for its sign). Forearms end at the hand tracker's wrist when a hand is visible.
- Neck join pulls the torso toward the head's neck base so the separately tracked head and torso stay joined. Torso width/size/length/up-down fit the torso to your body.
- `public/models/face_muscles.glb` + `face_muscles.json`: approximate facial muscles built by `tools/build_face_muscles.py` (BodyParts3D has none).
- Up to 3 people, front/back camera, Fit/Fill screen, snap with timer (saves a JPEG and offers sharing).
