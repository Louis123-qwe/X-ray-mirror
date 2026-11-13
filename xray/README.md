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
