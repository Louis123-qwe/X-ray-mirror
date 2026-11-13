import './style.css';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { FilesetResolver, FaceLandmarker, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODELS = 'https://storage.googleapis.com/mediapipe-models';
const FACE_MODEL = `${MODELS}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`;
const HAND_MODEL = `${MODELS}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`;
const POSE_MODEL = `${MODELS}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`;

const $ = (id) => document.getElementById(id);
const V3 = THREE.Vector3;
let W = 640, H = 480;

// ---------- State & UI ----------
const S = {
  skull: 1, brain: 1, muscles: 1, eyes: 1, teeth: 1, neck: 1, tbones: 1, organs: 1, hands: 1, outline: 1, labels: 1, tracking: 0,
  xray: true, bright: 0.8, real: 0.45, nlabels: 16, ntlabels: 8, tsize: 1, tht: 1, tup: 0, size: 1, wd: 1, ht: 1, dp: 1, lr: 0, up: 0, fw: 0,
};
const checks = [...document.querySelectorAll('[data-k]')];
checks.forEach((e) => (e.onchange = () => ((S[e.dataset.k] = e.checked ? 1 : 0), $('presets').querySelectorAll('.chip').forEach((b) => b.classList.remove('on')))));
const PRESETS = {
  skull: ['skull', 'teeth', 'neck', 'tbones'], brain: ['brain'], muscles: ['muscles', 'eyes'], organs: ['organs'],
  all: ['skull', 'brain', 'muscles', 'eyes', 'teeth', 'neck', 'tbones', 'organs'],
};
document.querySelectorAll('[data-p]').forEach((b) => (b.onclick = () => {
  document.querySelectorAll('[data-p]').forEach((x) => x.classList.toggle('on', x === b));
  for (const k of PRESETS.all) { S[k] = PRESETS[b.dataset.p].includes(k) ? 1 : 0; document.querySelector(`[data-k=${k}]`).checked = !!S[k]; }
}));
const sliders = [...document.querySelectorAll('[data-s]')];
sliders.forEach((el) => {
  const upd = () => {
    const v = +el.value;
    S[el.dataset.s] = v * (+el.dataset.scale || 1);
    el.parentElement.querySelector('b').textContent = el.dataset.unit === 'cm' ? (v / 10).toFixed(1) + ' cm' : el.dataset.unit === 'n' ? v : v + '%';
  };
  el.oninput = upd; el._upd = upd; upd();
});
$('reset').onclick = () => sliders.filter((e) => e.classList.contains('fit')).forEach((e) => { e.value = e.defaultValue; e._upd(); });
const setX = (x) => { S.xray = x; $('mx').classList.toggle('on', x); $('ms').classList.toggle('on', !x); };
$('mx').onclick = () => setX(true);
$('ms').onclick = () => setX(false);

// ---------- Three.js scene ----------
const renderer = new THREE.WebGLRenderer({ canvas: $('gl'), alpha: true, antialias: true });
renderer.setPixelRatio(1);
renderer.setClearColor(0x000000, 0);
const scene = new THREE.Scene();
const cam = new THREE.OrthographicCamera(0, 640, 0, -480, -3000, 3000);
cam.position.z = 1000;
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(-200, 300, 600);
scene.add(sun);

const glow = (rgb, base, power) => new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, uniforms: { k: { value: 0.8 } },
  vertexShader: 'varying vec3 vN;void main(){vN=normalize(normalMatrix*normal);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
  fragmentShader: `uniform float k;varying vec3 vN;void main(){float f=1.-abs(normalize(vN).z);float a=(${base}+.85*pow(f,${power}))*k;gl_FragColor=vec4(vec3(${rgb})*a,a);}`,
});
const solid = (hex) => new THREE.MeshStandardMaterial({ color: hex, roughness: 0.6, side: THREE.DoubleSide });
// node: [x-ray colour, base glow, rim power, solid colour]
const STYLE = {
  skull: ['.55,.82,1.', '.07', '2.', 0xe9e1cc], jaw_bone: ['.55,.82,1.', '.07', '2.', 0xe9e1cc], neck: ['.55,.82,1.', '.07', '2.', 0xe3dac4],
  teeth_upper: ['.8,.95,1.', '.12', '1.6', 0xfffdf2], teeth_lower: ['.8,.95,1.', '.12', '1.6', 0xfffdf2],
  cerebrum: ['1.,.55,.75', '.16', '1.5', 0xe8a0b4], deep: ['1.,.78,.85', '.2', '1.3', 0xf1c0cb], cerebellum: ['.85,.5,.95', '.16', '1.5', 0xc58fb5], brainstem: ['1.,.7,.6', '.18', '1.4', 0xe6b199],
  eye_muscles: ['1.,.35,.35', '.14', '1.6', 0xb8433f], palate: ['1.,.35,.35', '.14', '1.6', 0xb8433f], tongue: ['1.,.45,.5', '.14', '1.5', 0xc9606a], glands: ['1.,.75,.3', '.14', '1.5', 0xd9a441],
  eyes: ['.75,.9,1.', '.2', '1.2', 0xf4f6ff], iris: ['.35,.85,1.', '.4', '1.', 0x3b6fa0],
};
const BONE = ['.55,.82,1.', '.07', '2.', 0xe9e1cc];
const TSTYLE = {
  ribs: BONE, spine_t: BONE, shoulder: BONE, pelvis: BONE,
  heart: ['1.,.25,.3', '.2', '1.4', 0xc0392b], vessels: ['1.,.4,.45', '.16', '1.5', 0xb03a4e], airways: ['.5,1.,.9', '.18', '1.5', 0xdfe9ee],
  liver: ['1.,.6,.25', '.14', '1.5', 0x8a4b2a], gi: ['1.,.7,.5', '.14', '1.5', 0xd9917a], kidneys: ['.9,.4,.45', '.16', '1.5', 0x9b3d3d], tmuscle: ['1.,.35,.35', '.1', '1.6', 0xb8433f],
};
Object.assign(STYLE, TSTYLE);
const mats = Object.fromEntries(Object.entries(STYLE).map(([n, [rgb, b, p, hex]]) => [n, { x: glow(rgb, b, p), s: solid(hex) }]));
const LAYERS = {
  skull: ['skull', 'jaw_bone'], teeth: ['teeth_upper', 'teeth_lower'], neck: ['neck'], eyes: ['eyes', 'iris'],
  brain: ['cerebrum', 'deep', 'cerebellum', 'brainstem'], muscles: ['eye_muscles', 'palate', 'tongue', 'glands'],
  tbones: ['ribs', 'spine_t', 'shoulder', 'pelvis'], organs: ['heart', 'vessels', 'airways', 'liver', 'gi', 'kidneys', 'tmuscle'],
};
const JAW = ['jaw_bone', 'teeth_lower', 'tongue']; // these ride on the jaw pivot

const root = new THREE.Group();
const pivot = new THREE.Group();
pivot.position.set(0, -30, -90); // jaw hinge in head space
root.add(pivot);
scene.add(root);
const troot = new THREE.Group(); // torso, tracked separately from the pose landmarks
troot.visible = false;
scene.add(troot);
let tAnchor = null, torsoReady = false, torsoLabels = [];
const nodes = {};
let labelDefs = [];

async function loadModels() {
  const base = import.meta.env.BASE_URL + 'models/';
  const [gltf, labels] = await Promise.all([new GLTFLoader().loadAsync(base + 'anatomy.glb'), fetch(base + 'labels.json').then((r) => r.json())]);
  for (const name of Object.keys(STYLE).filter((n) => !(n in TSTYLE))) {
    const o = gltf.scene.getObjectByName(name);
    if (!o) continue;
    o.geometry.computeVertexNormals();
    (JAW.includes(name) ? pivot : root).add(o);
    nodes[name] = o;
  }
  labelDefs = labels.map((l) => ({ ...l, a: l.a.map((p) => new V3(...p)) }));
}

async function loadTorso() {
  try {
    const base = import.meta.env.BASE_URL + 'models/';
    const [gltf, info] = await Promise.all([new GLTFLoader().loadAsync(base + 'torso.glb'), fetch(base + 'torso.json').then((r) => r.json())]);
    for (const name of Object.keys(TSTYLE)) {
      const o = gltf.scene.getObjectByName(name);
      if (!o) continue;
      o.geometry.computeVertexNormals();
      troot.add(o); nodes[name] = o; styleX = null;
    }
    const a = info.anchors, v = (k) => new V3(...a[k]);
    tAnchor = { shMid: v('shoulderPlus').add(v('shoulderMinus')).multiplyScalar(0.5), sw: v('shoulderPlus').distanceTo(v('shoulderMinus')) };
    tAnchor.len = tAnchor.shMid.distanceTo(v('hipPlus').add(v('hipMinus')).multiplyScalar(0.5));
    torsoLabels = info.labels.map((l) => ({ ...l, a: l.a.map((p) => new V3(...p)) }));
    torsoReady = true;
  } catch (e) { console.warn('torso models failed', e); }
}

let styleX = null;
function applyLook() {
  for (const [n, o] of Object.entries(nodes)) {
    if (styleX !== S.xray) o.material = S.xray ? mats[n].x : mats[n].s;
    if (S.xray) mats[n].x.uniforms.k.value = S.bright;
  }
  styleX = S.xray;
  for (const [layer, names] of Object.entries(LAYERS)) for (const n of names) if (nodes[n]) nodes[n].visible = !!S[layer];
}

// ---------- Head pose from face landmarks ----------
function poseHead(l) {
  const p = (i) => new V3(l[i].x * W, -l[i].y * H, -l[i].z * W);
  const a = p(234), b = p(454), nose = p(168), chin = p(152), top = p(10), m1 = p(13), m2 = p(14);
  const r = b.clone().sub(a);
  const w = r.length();
  r.normalize();
  const u0 = top.clone().sub(chin);
  const u = u0.sub(r.clone().multiplyScalar(u0.dot(r))).normalize();
  const f = new V3().crossVectors(r, u);
  const s = (w / 140) * S.size;
  root.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, u, f));
  root.scale.set(s * S.wd, s * S.ht, s * S.dp);
  root.position.copy(nose).addScaledVector(f, -4 * s + S.fw * s).addScaledVector(u, S.up * s).addScaledVector(r, S.lr * s);
  pivot.rotation.x = Math.min(0.6, Math.max(0, (m1.distanceTo(m2) / w - 0.01) * 1.4));
  root.updateMatrixWorld(true);
  return w;
}

// ---------- Torso pose from body landmarks ----------
const vis = (l, i) => (l[i].visibility ?? 1);
function poseTorso(l, headZ) {
  const p = (i) => new V3(l[i].x * W, -l[i].y * H, -l[i].z * W * 0.5);
  const Ls = p(11), Rs = p(12);
  if (!torsoReady || Math.min(vis(l, 11), vis(l, 12)) < 0.5) return false;
  const r = Ls.clone().sub(Rs), sw = r.length();
  if (sw < 20) return false;
  r.normalize();
  const sh = Ls.clone().add(Rs).multiplyScalar(0.5);
  const hipsOk = Math.min(vis(l, 23), vis(l, 24)) > 0.5;
  const hip = p(23).add(p(24)).multiplyScalar(0.5);
  const u0 = hipsOk ? sh.clone().sub(hip) : new V3(-r.y, r.x, 0);
  const u = u0.sub(r.clone().multiplyScalar(u0.dot(r))).normalize();
  const f = new V3().crossVectors(r, u);
  const s = (sw / tAnchor.sw) * S.tsize;
  let ht = 1;
  if (hipsOk) ht = Math.min(1.3, Math.max(0.75, sh.distanceTo(hip) / (s * tAnchor.len)));
  const sc = new V3(s, s * ht * S.tht, s);
  troot.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, u, f));
  troot.scale.copy(sc);
  const off = tAnchor.shMid.clone().multiply(sc).applyQuaternion(troot.quaternion);
  troot.position.copy(sh).sub(off).addScaledVector(u, S.tup * s);
  troot.position.z = (headZ ?? 0) + tAnchor.shMid.z * s - off.z; // put the torso at the head's depth, behind it
  troot.updateMatrixWorld(true);
  return true;
}

// ---------- 2D overlays ----------
const ov = $('ov').getContext('2d');
const lab = $('labels').getContext('2d');
const FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];

function drawOutline(l) {
  ov.save();
  ov.strokeStyle = `rgba(150,210,255,${0.75 * S.bright})`; ov.lineWidth = 2; ov.shadowBlur = 8; ov.shadowColor = 'rgba(110,190,255,.9)';
  ov.beginPath();
  for (const { start, end } of FaceLandmarker.FACE_LANDMARKS_FACE_OVAL) { ov.moveTo(l[start].x * W, l[start].y * H); ov.lineTo(l[end].x * W, l[end].y * H); }
  ov.stroke(); ov.restore();
}

function drawHands(hands) {
  ov.lineCap = 'round';
  ov.shadowBlur = S.xray ? 7 : 3;
  ov.shadowColor = S.xray ? 'rgba(110,190,255,.9)' : 'rgba(0,0,0,.5)';
  for (const h of hands) {
    const q = (i) => [h[i].x * W, h[i].y * H];
    const ps = Math.hypot(q(0)[0] - q(9)[0], q(0)[1] - q(9)[1]);
    const edge = S.xray ? `rgba(215,238,255,${0.95 * S.bright})` : '#8f8466';
    const core = S.xray ? 'rgba(20,50,90,.6)' : '#cfc5a8';
    for (const f of FINGERS) for (let i = 0; i < 4; i++) {
      const a = q(f[i]), b = q(f[i + 1]), wd = ps * (0.2 - 0.035 * i);
      for (const [col, k] of [[edge, 1], [core, 0.55]]) {
        ov.strokeStyle = col; ov.lineWidth = wd * k;
        ov.beginPath(); ov.moveTo(a[0], a[1]); ov.lineTo(b[0], b[1]); ov.stroke();
      }
      ov.fillStyle = S.xray ? 'rgba(225,242,255,.8)' : '#f3ecd8';
      ov.beginPath(); ov.arc(b[0], b[1], wd * 0.6, 0, 7); ov.fill();
    }
  }
  ov.shadowBlur = 0;
}

// Labels are drawn on an un-mirrored canvas, so image x is flipped to screen x here.
function collectLabels(lm, hands) {
  const items = [];
  if (troot.visible && S.labels) {
    let c = 0;
    for (const d of torsoLabels) {
      if (c >= S.ntlabels) break;
      if (!S[d.l]) continue;
      let best = null;
      for (const a of d.a) { const w = troot.localToWorld(a.clone()); if (!best || w.z > best.z) best = w; }
      items.push({ t: d.t, x: best.x, y: -best.y }); c++;
    }
  }
  if (lm && S.labels) {
    let c = 0;
    for (const d of labelDefs) {
      if (c >= S.nlabels) break;
      if (!S[d.l]) continue;
      const parent = d.f === 'HG' ? pivot : root;
      let best = null;
      for (const a of d.a) { const w = parent.localToWorld(a.clone()); if (!best || w.z > best.z) best = w; }
      items.push({ t: d.t, x: best.x, y: -best.y }); c++;
    }
  }
  if (S.hands && S.labels) {
    const h = hands[0];
    if (h) {
      const q = (i) => [h[i].x * W, h[i].y * H], mid = (a, b) => [(q(a)[0] + q(b)[0]) / 2, (q(a)[1] + q(b)[1]) / 2];
      for (const [t, [x, y]] of [['Metacarpal', mid(5, 0)], ['Proximal phalanx', mid(9, 10)], ['Middle phalanx', mid(10, 11)], ['Distal phalanx', q(8)]]) items.push({ t, x, y });
    }
  }
  return items;
}

function drawLabels(items, info) {
  lab.clearRect(0, 0, W, H);
  const fs = Math.max(10, Math.round(W / 52));
  lab.font = `${fs}px system-ui, sans-serif`; lab.textBaseline = 'middle';
  const sides = [[], []];
  for (const it of items) { it.sx = W - it.x; sides[it.sx < W / 2 ? 0 : 1].push(it); }
  sides.forEach((side, k) => {
    side.sort((a, b) => a.y - b.y);
    let prev = -1e9;
    for (const it of side) { it.ly = Math.min(H - fs, Math.max(fs, it.y, prev + fs + 3)); prev = it.ly; }
    for (const it of side) {
      const tw = lab.measureText(it.t).width, ex = k === 0 ? 6 + tw + 4 : W - 6 - tw - 4;
      lab.strokeStyle = 'rgba(255,255,255,.55)'; lab.lineWidth = 1;
      lab.beginPath(); lab.moveTo(it.sx, it.y); lab.lineTo(ex, it.ly); lab.stroke();
      lab.fillStyle = '#fff'; lab.beginPath(); lab.arc(it.sx, it.y, 2, 0, 7); lab.fill();
      lab.textAlign = k === 0 ? 'left' : 'right'; lab.shadowColor = '#000'; lab.shadowBlur = 3;
      lab.fillText(it.t, k === 0 ? 6 : W - 6, it.ly); lab.shadowBlur = 0;
    }
  });
  if (S.tracking) { lab.textAlign = 'left'; lab.fillStyle = '#9fe'; lab.fillText(info, 6, fs); }
}

// ---------- Tracking ----------
const ema3 = (prev, next, a) => (prev ? next.map((q, i) => ({ x: prev[i].x + (q.x - prev[i].x) * a, y: prev[i].y + (q.y - prev[i].y) * a, z: prev[i].z + (q.z - prev[i].z) * a, visibility: q.visibility })) : next);
const ema = (prev, next, a) => (prev ? next.map((q, i) => ({ x: prev[i].x + (q.x - prev[i].x) * a, y: prev[i].y + (q.y - prev[i].y) * a, z: q.z })) : next);

async function makeLandmarkers() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const mk = async (Cls, url, extra) => {
    for (const delegate of ['GPU', 'CPU']) {
      try { return await Cls.createFromOptions(fileset, { baseOptions: { modelAssetPath: url, delegate }, runningMode: 'VIDEO', ...extra }); }
      catch (e) { console.warn(Cls.name, delegate, e); }
    }
    throw new Error('Could not load ' + Cls.name);
  };
  const pose = await mk(PoseLandmarker, POSE_MODEL, { numPoses: 1 }).catch((e) => (console.warn('pose unavailable', e), null));
  return [await mk(FaceLandmarker, FACE_MODEL, { numFaces: 1 }), await mk(HandLandmarker, HAND_MODEL, { numHands: 2 }), pose];
}

$('go').onclick = async () => {
  const st = $('st'), go = $('go'), vid = $('vid');
  go.style.display = 'none';
  try {
    st.textContent = 'Starting camera…';
    vid.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 }, audio: false });
    await vid.play();
    W = vid.videoWidth; H = vid.videoHeight;
    $('stage').style.aspectRatio = `${W}/${H}`;
    for (const id of ['ov', 'labels']) { $(id).width = W; $(id).height = H; }
    renderer.setSize(W, H, false);
    Object.assign(cam, { left: 0, right: W, top: 0, bottom: -H }); cam.updateProjectionMatrix();
    st.textContent = 'Loading anatomy models…';
    await loadModels();
    st.textContent = 'Loading tracking models…';
    const [face, hand, pose] = await makeLandmarkers();
    $('msg').style.display = 'none';
    loadTorso();

    let lm = null, hands = [], pl = null, last = -1, n = 0, fps = 0, tPrev = performance.now(), w = 0, torsoOn = false;
    const loop = () => {
      requestAnimationFrame(loop);
      if (vid.readyState < 2 || vid.currentTime === last) return;
      last = vid.currentTime;
      const t = performance.now();
      fps = Math.round(0.9 * fps + 0.1 * (1000 / (t - tPrev))); tPrev = t;
      const fr = face.detectForVideo(vid, t);
      lm = fr.faceLandmarks[0] ? ema(lm, fr.faceLandmarks[0], 0.55) : null;
      n++;
      if (S.hands && n % 2 === 0) { const hr = hand.detectForVideo(vid, t); hands = hr.landmarks.map((h, k) => ema(hands[k], h, 0.6)); }
      if (!S.hands) hands = [];
      if (pose && (S.tbones || S.organs)) {
        if (n % 2 === 1) { const pr = pose.detectForVideo(vid, t); pl = pr.landmarks[0] ? ema3(pl, pr.landmarks[0], 0.5) : null; }
      } else pl = null;
      $('dim').style.opacity = 1 - S.real;
      ov.clearRect(0, 0, W, H);
      if (lm && S.outline) drawOutline(lm);
      if (S.hands) drawHands(hands);
      root.visible = !!lm;
      if (lm) w = poseHead(lm);
      torsoOn = !!pl && poseTorso(pl, lm ? root.position.z : 0);
      troot.visible = torsoOn;
      applyLook();
      renderer.render(scene, cam);
      drawLabels(collectLabels(lm, hands), `${fps} fps · face ${lm ? 'yes' : 'no'} · head ${Math.round(w)}px · hands ${hands.length} · torso ${torsoOn ? 'yes' : torsoReady ? 'no' : 'loading'}`);
    };
    loop();
  } catch (e) {
    console.error(e);
    st.textContent = 'Could not start: ' + (e.message || e);
    go.style.display = ''; go.textContent = 'Try again';
    $('msg').style.display = '';
  }
};
