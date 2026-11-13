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
  skull: 1, brain: 1, muscles: 1, eyes: 1, teeth: 1, neck: 1, tbones: 1, organs: 1, tmusc: 1, abones: 1, amusc: 1, hands: 1, outline: 1, labels: 1, tracking: 0,
  xray: true, bright: 0.8, real: 0.45, nlabels: 16, ntlabels: 8, tsize: 1, tht: 1, tw: 1, tup: 0, neckjoin: 0.3, smooth: 0.6, size: 1, wd: 1, ht: 1, dp: 1, lr: 0, up: 0, fw: 0,
};
const checks = [...document.querySelectorAll('[data-k]')];
checks.forEach((e) => (e.onchange = () => ((S[e.dataset.k] = e.checked ? 1 : 0), $('presets').querySelectorAll('.chip').forEach((b) => b.classList.remove('on')))));
const PRESETS = {
  skull: ['skull', 'teeth', 'neck', 'tbones', 'abones'], brain: ['brain'], muscles: ['muscles', 'eyes', 'tmusc', 'amusc'], organs: ['organs'],
  all: ['skull', 'brain', 'muscles', 'eyes', 'teeth', 'neck', 'tbones', 'organs', 'tmusc', 'abones', 'amusc'],
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
$('reset').addEventListener('click', (e) => e.preventDefault(), true); // don't toggle the section
$('reset').onclick = () => sliders.filter((e) => e.classList.contains('fit')).forEach((e) => { e.value = e.defaultValue; e._upd(); });
const setX = (x) => { S.xray = x; $('mx').classList.toggle('on', x); $('ms').classList.toggle('on', !x); };
$('mx').onclick = () => setX(true);
$('ms').onclick = () => setX(false);

// ---------- Layout & collapsible UI ----------
let fill = false;
const fit = () => {
  const fr = $('frame'), ar = W / H, vw = window.innerWidth, vh = window.innerHeight;
  const w = fill ? Math.max(vw, vh * ar) : Math.min(vw, vh * ar);
  fr.style.width = w + 'px'; fr.style.height = w / ar + 'px';
};
const store = { get: (k) => { try { return localStorage.getItem('xr_' + k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem('xr_' + k, v); } catch {} } };
fill = store.get('fill') === '1';
window.addEventListener('resize', fit); fit();
const sheet = $('sheet');
const openSheet = (o) => { sheet.classList.toggle('open', o); store.set('open', o ? '1' : '0'); };
if (store.get('open') === '0') sheet.classList.remove('open');
$('grab').onclick = () => openSheet(!sheet.classList.contains('open'));
$('bCtl').onclick = () => openSheet(!sheet.classList.contains('open'));
document.querySelectorAll('details').forEach((d, i) => {
  const v = store.get('d' + i); if (v !== null) d.open = v === '1';
  d.ontoggle = () => store.set('d' + i, d.open ? '1' : '0');
});
const setClean = (c) => {
  document.body.classList.toggle('clean', c);
  const el = document.documentElement;
  if (c && el.requestFullscreen) el.requestFullscreen().catch(() => {});
  if (!c && document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {});
  setTimeout(fit, 150);
};
$('bClean').onclick = () => setClean(true);
$('bShow').onclick = () => setClean(false);
document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement) document.body.classList.remove('clean'); setTimeout(fit, 150); });

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
const BONE = ['.55,.82,1.', '.07', '2.', 0xe9e1cc];
const MUSC = ['1.,.35,.35', '.12', '1.6', 0xb8433f];
const HEAD_STYLE = {
  skull: BONE, jaw_bone: BONE, neck: ['.55,.82,1.', '.07', '2.', 0xe3dac4],
  teeth_upper: ['.8,.95,1.', '.12', '1.6', 0xfffdf2], teeth_lower: ['.8,.95,1.', '.12', '1.6', 0xfffdf2],
  cerebrum: ['1.,.55,.75', '.16', '1.5', 0xe8a0b4], deep: ['1.,.78,.85', '.2', '1.3', 0xf1c0cb], cerebellum: ['.85,.5,.95', '.16', '1.5', 0xc58fb5], brainstem: ['1.,.7,.6', '.18', '1.4', 0xe6b199],
  eye_muscles: ['1.,.35,.35', '.14', '1.6', 0xb8433f], palate: ['1.,.35,.35', '.14', '1.6', 0xb8433f], tongue: ['1.,.45,.5', '.14', '1.5', 0xc9606a], glands: ['1.,.75,.3', '.14', '1.5', 0xd9a441],
  eyes: ['.75,.9,1.', '.2', '1.2', 0xf4f6ff], iris: ['.35,.85,1.', '.4', '1.', 0x3b6fa0],
};
const TORSO_STYLE = {
  ribs: BONE, spine_t: BONE, shoulder: BONE, pelvis: BONE,
  heart: ['1.,.25,.3', '.2', '1.4', 0xc0392b], vessels: ['1.,.4,.45', '.16', '1.5', 0xb03a4e], airways: ['.5,1.,.9', '.18', '1.5', 0xdfe9ee],
  liver: ['1.,.6,.25', '.14', '1.5', 0x8a4b2a], gi: ['1.,.7,.5', '.14', '1.5', 0xd9917a], kidneys: ['.9,.4,.45', '.16', '1.5', 0x9b3d3d], tmuscle: ['1.,.35,.35', '.1', '1.6', 0xb8433f],
};
const MUSC_STYLE = { neckm: MUSC, chest: MUSC, back: MUSC, inter: MUSC };
for (const sd of ['P', 'M']) Object.assign(MUSC_STYLE, { ['uarm_m_' + sd]: MUSC, ['farm_m_' + sd]: MUSC, ['uarm_b_' + sd]: BONE, ['farm_b_' + sd]: BONE });
const STYLE = { ...HEAD_STYLE, ...TORSO_STYLE, ...MUSC_STYLE, face_m: MUSC, hbone: BONE };
const mats = Object.fromEntries(Object.entries(STYLE).map(([n, [rgb, b, p, hex]]) => [n, { x: glow(rgb, b, p), s: solid(hex) }]));
const matOf = (n) => mats[n] || mats.hbone;
const LAYERS = {
  skull: ['skull', 'jaw_bone'], teeth: ['teeth_upper', 'teeth_lower'], neck: ['neck'], eyes: ['eyes', 'iris'],
  brain: ['cerebrum', 'deep', 'cerebellum', 'brainstem'], muscles: ['eye_muscles', 'palate', 'tongue', 'glands', 'face_m'],
  tmusc: ['neckm', 'chest', 'back', 'inter'], amusc: ['uarm_m_P', 'uarm_m_M', 'farm_m_P', 'farm_m_M'], abones: ['uarm_b_P', 'uarm_b_M', 'farm_b_P', 'farm_b_M'],
  tbones: ['ribs', 'spine_t', 'shoulder', 'pelvis'], organs: ['heart', 'vessels', 'airways', 'liver', 'gi', 'kidneys', 'tmuscle'],
};
const JAW = ['jaw_bone', 'teeth_lower', 'tongue']; // these ride on the jaw pivot
const SEG = { carp: [0, 9], met1: [1, 2], met2: [0, 5], met3: [0, 9], met4: [0, 13], met5: [0, 17], prox_thumb: [2, 3], dist_thumb: [3, 4] };
[['index', 5], ['middle', 9], ['ring', 13], ['little', 17]].forEach(([f, a]) => { SEG['prox_' + f] = [a, a + 1]; SEG['mid_' + f] = [a + 1, a + 2]; SEG['dist_' + f] = [a + 2, a + 3]; });

// ---------- Model templates (loaded once, shared by every person) ----------
const TPL = { head: null, torso: null, musc: null, hands: null, face: null };
const base = () => import.meta.env.BASE_URL + 'models/';
const pickGeoms = (gltf, names) => {
  const g = {};
  for (const n of names) { const o = gltf.scene.getObjectByName(n); if (o) { o.geometry.computeVertexNormals(); g[n] = o.geometry; } }
  return g;
};
const lbl = (arr) => arr.map((l) => ({ ...l, a: l.a.map((p) => new V3(...p)) }));
async function loadModels() {
  const [gltf, labels] = await Promise.all([new GLTFLoader().loadAsync(base() + 'anatomy.glb'), fetch(base() + 'labels.json').then((r) => r.json())]);
  TPL.head = { geoms: pickGeoms(gltf, Object.keys(HEAD_STYLE)), labels: lbl(labels) };
}
async function loadFace() {
  try {
    const [gltf, labels] = await Promise.all([new GLTFLoader().loadAsync(base() + 'face_muscles.glb'), fetch(base() + 'face_muscles.json').then((r) => r.json())]);
    TPL.face = { geoms: pickGeoms(gltf, ['face_m']) };
    if (TPL.head) TPL.head.labels.unshift(...lbl(labels));
  } catch (e) { console.warn('face muscles failed', e); }
}
async function loadTorso() {
  try {
    const [gltf, info] = await Promise.all([new GLTFLoader().loadAsync(base() + 'torso.glb'), fetch(base() + 'torso.json').then((r) => r.json())]);
    const a = info.anchors, v = (k) => new V3(...a[k]);
    const anchor = { shMid: v('shoulderPlus').add(v('shoulderMinus')).multiplyScalar(0.5), sw: v('shoulderPlus').distanceTo(v('shoulderMinus')) };
    anchor.len = anchor.shMid.distanceTo(v('hipPlus').add(v('hipMinus')).multiplyScalar(0.5));
    TPL.torso = { geoms: pickGeoms(gltf, Object.keys(TORSO_STYLE)), anchor, labels: lbl(info.labels) };
  } catch (e) { console.warn('torso models failed', e); }
}
async function loadMuscles() {
  try {
    const [gltf, info] = await Promise.all([new GLTFLoader().loadAsync(base() + 'muscles.glb'), fetch(base() + 'muscles.json').then((r) => r.json())]);
    TPL.musc = { geoms: pickGeoms(gltf, Object.keys(MUSC_STYLE)), joints: info.joints, labels: lbl(info.labels) };
  } catch (e) { console.warn('muscle models failed', e); }
}
async function loadHands() {
  try {
    const [gltf, info] = await Promise.all([new GLTFLoader().loadAsync(base() + 'hands.glb'), fetch(base() + 'hands.json').then((r) => r.json())]);
    const geoms = {};
    for (const sd of ['P', 'M']) for (const key of Object.keys(info[sd] || {})) {
      const o = gltf.scene.getObjectByName(key + '_' + sd);
      if (o) { o.geometry.computeVertexNormals(); geoms[key + '_' + sd] = o.geometry; }
    }
    TPL.hands = { geoms, info };
  } catch (e) { console.warn('3D hand models failed, using flat hands', e); }
}

// ---------- One Person = one full anatomy rig ----------
const vis = (l, i) => (l[i].visibility ?? 1);
class Person {
  constructor() {
    this.root = new THREE.Group(); this.pivot = new THREE.Group(); this.pivot.position.set(0, -30, -90); this.root.add(this.pivot);
    this.troot = new THREE.Group();
    this.rig = {}; this.hrig = { P: {}, M: {} }; this.nodes = {}; this.built = {};
    this.tScale = 1; this.tZoff = 0; this.sRef = 0; this.torsoOn = false; this.hands3d = false; this.neckBase = null; this.t1 = null;
    for (const k of ['UP', 'UM', 'FP', 'FM']) { const g = new THREE.Group(), inner = new THREE.Group(); g.add(inner); g.visible = false; scene.add(g); this.rig[k] = { g, inner, hold: 0, k: 1, sign: 1, init: false }; }
    this.root.visible = this.troot.visible = false;
    scene.add(this.root, this.troot);
  }
  mesh(name, geom, parent) { const o = new THREE.Mesh(geom, matOf(name).x); parent.add(o); this.nodes[name] = o; return o; }
  sync() {
    if (!this.built.head && TPL.head) {
      this.built.head = 1;
      for (const [n, g] of Object.entries(TPL.head.geoms)) this.mesh(n, g, JAW.includes(n) ? this.pivot : this.root);
    }
    if (!this.built.torso && TPL.torso) {
      this.built.torso = 1;
      for (const [n, g] of Object.entries(TPL.torso.geoms)) this.mesh(n, g, this.troot);
    }
    if (!this.built.face && TPL.face) { this.built.face = 1; this.mesh('face_m', TPL.face.geoms.face_m, this.root); }
    if (!this.neckBase && TPL.head && TPL.torso && TPL.head.geoms.neck && TPL.torso.geoms.spine_t) {
      const end = (geom, top) => {
        const a = geom.attributes.position, v = []; let ext = top ? -1e9 : 1e9;
        for (let i = 0; i < a.count; i++) ext = top ? Math.max(ext, a.getY(i)) : Math.min(ext, a.getY(i));
        const c = new V3(); let n = 0;
        for (let i = 0; i < a.count; i++) if (Math.abs(a.getY(i) - ext) < 6) { c.x += a.getX(i); c.y += a.getY(i); c.z += a.getZ(i); n++; }
        return c.multiplyScalar(1 / Math.max(1, n));
      };
      this.neckBase = end(TPL.head.geoms.neck, false); this.t1 = end(TPL.torso.geoms.spine_t, true);
    }
    if (!this.built.musc && TPL.musc) {
      this.built.musc = 1;
      const jt = (sd, k) => new V3(...TPL.musc.joints[sd][k]);
      for (const sd of ['P', 'M']) {
        const sh = jt(sd, 'shoulder'), el = jt(sd, 'elbow'), wr = jt(sd, 'wrist');
        Object.assign(this.rig['U' + sd], { dir: el.clone().sub(sh).normalize(), len: sh.distanceTo(el) });
        Object.assign(this.rig['F' + sd], { dir: wr.clone().sub(el).normalize(), len: el.distanceTo(wr) });
        this.rig['U' + sd].inner.position.copy(sh).negate(); this.rig['F' + sd].inner.position.copy(el).negate();
      }
      for (const [n, g] of Object.entries(TPL.musc.geoms)) {
        const m = n.match(/^(uarm|farm)_[mb]_([PM])$/);
        this.mesh(n, g, m ? this.rig[(m[1] === 'uarm' ? 'U' : 'F') + m[2]].inner : n === 'neckm' ? this.root : this.troot);
      }
    }
    if (!this.built.hands && TPL.hands) {
      this.built.hands = 1;
      for (const sd of ['P', 'M']) for (const [key, b] of Object.entries(TPL.hands.info[sd] || {})) {
        const geom = TPL.hands.geoms[key + '_' + sd];
        if (!geom) continue;
        const g = new THREE.Group(), inner = new THREE.Group();
        inner.position.set(...b.pivot).negate(); g.add(inner); g.visible = false; scene.add(g);
        this.mesh(key + '_' + sd, geom, inner);
        this.hrig[sd][key] = { g, axis: new V3(...b.axis), len: b.len, anchor: b.anchor };
      }
      this.hands3d = Object.keys(this.hrig.P).length > 0;
    }
  }
  look() {
    for (const [n, o] of Object.entries(this.nodes)) {
      const m = matOf(n); o.material = S.xray ? m.x : m.s;
    }
    for (const [layer, names] of Object.entries(LAYERS)) for (const n of names) if (this.nodes[n]) this.nodes[n].visible = !!S[layer];
  }
  hide() {
    this.root.visible = this.troot.visible = false; this.torsoOn = false;
    for (const r of Object.values(this.rig)) r.g.visible = false;
    for (const sd of ['P', 'M']) for (const r of Object.values(this.hrig[sd])) r.g.visible = false;
  }
  remove() { this.hide(); scene.remove(this.root, this.troot); for (const r of Object.values(this.rig)) scene.remove(r.g); for (const sd of ['P', 'M']) for (const r of Object.values(this.hrig[sd])) scene.remove(r.g); }

  // Head pose from face landmarks
  poseHead(l) {
    const p = (i) => new V3(l[i].x * W, -l[i].y * H, -l[i].z * W);
    const a = p(234), b = p(454), nose = p(168), chin = p(152), top = p(10), m1 = p(13), m2 = p(14);
    const r = b.clone().sub(a), w = r.length();
    r.normalize();
    const u0 = top.clone().sub(chin);
    const u = u0.sub(r.clone().multiplyScalar(u0.dot(r))).normalize();
    const f = new V3().crossVectors(r, u);
    const s = (w / 140) * S.size;
    this.root.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, u, f));
    this.root.scale.set(s * S.wd, s * S.ht, s * S.dp);
    this.root.position.copy(nose).addScaledVector(f, -4 * s + S.fw * s).addScaledVector(u, S.up * s).addScaledVector(r, S.lr * s);
    this.pivot.rotation.x = Math.min(0.6, Math.max(0, (m1.distanceTo(m2) / w - 0.01) * 1.4));
    this.root.updateMatrixWorld(true);
    return w;
  }
  // Torso pose from body landmarks
  poseTorso(l, headZ) {
    const T = TPL.torso && this.built.torso ? TPL.torso.anchor : null;
    if (!T || Math.min(vis(l, 11), vis(l, 12)) < 0.5) return false;
    const p = (i) => new V3(l[i].x * W, -l[i].y * H, -l[i].z * W * 0.5);
    const Ls = p(11), Rs = p(12), r = Ls.clone().sub(Rs), sw = r.length();
    if (sw < 20) return false;
    r.normalize();
    const sh = Ls.clone().add(Rs).multiplyScalar(0.5);
    const hipsOk = Math.min(vis(l, 23), vis(l, 24)) > 0.5;
    const hip = p(23).add(p(24)).multiplyScalar(0.5);
    const u0 = hipsOk ? sh.clone().sub(hip) : new V3(-r.y, r.x, 0);
    const u = u0.sub(r.clone().multiplyScalar(u0.dot(r))).normalize();
    const f = new V3().crossVectors(r, u);
    const s = (sw / T.sw) * S.tsize;
    let ht = 1;
    if (hipsOk) ht = Math.min(1.3, Math.max(0.75, sh.distanceTo(hip) / (s * T.len)));
    this.sRef = Math.max(s, (this.sRef || s) * 0.97); // slowly-decaying peak: stays right when the torso turns sideways
    const sc = new V3(s * S.tw, s * ht * S.tht, s * S.tw), t = this.troot;
    t.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, u, f));
    t.scale.copy(sc);
    const off = T.shMid.clone().multiply(sc).applyQuaternion(t.quaternion);
    t.position.copy(sh).sub(off).addScaledVector(u, S.tup * s);
    t.position.z = (headZ ?? 0) + T.shMid.z * s - off.z; // torso at the head's depth, behind it
    this.tScale = s; this.tZoff = t.position.z + off.z - sh.z;
    t.updateMatrixWorld(true);
    return true;
  }
  // Which tracked hand belongs to which arm (P = +x side = image right)
  handSides(hands, pl) {
    const out = { P: null, M: null }, wr = pl && this.torsoOn ? { P: pl[15], M: pl[16] } : null;
    for (const h of hands) {
      let sd = h[0].x > 0.5 ? 'P' : 'M';
      if (wr) sd = Math.hypot(wr.P.x - h[0].x, wr.P.y - h[0].y) < Math.hypot(wr.M.x - h[0].x, wr.M.y - h[0].y) ? 'P' : 'M';
      if (out[sd]) sd = sd === 'P' ? 'M' : 'P';
      if (!out[sd]) out[sd] = h;
    }
    return out;
  }
  // Arms: image x/y give the direction; depth comes from how foreshortened the segment is (the tracker's own depth is only used for its sign)
  poseArms(l, hs) {
    const on = S.abones || S.amusc, sA = this.sRef || this.tScale;
    for (const [sd, a, b, c] of [['P', 11, 13, 15], ['M', 12, 14, 16]]) {
      const U = this.rig['U' + sd], F = this.rig['F' + sd];
      if (!on || !this.built.musc || !U.dir) { U.g.visible = F.g.visible = false; continue; }
      const P = (i) => new V3(l[i].x * W, -l[i].y * H, -l[i].z * W * 0.5 + this.tZoff);
      const fit = (R, A, B, raw0, raw1) => {
        const d = B.clone().sub(A), dxy = Math.hypot(d.x, d.y), unit = R.len * sA;
        if (dxy < 6) return null;
        R.k = Math.min(1.3, Math.max(0.85, Math.max(dxy / unit, R.k * 0.997)));
        const dz = Math.sqrt(Math.max(0, (unit * R.k) ** 2 - dxy * dxy)), dr = raw1 - raw0;
        if (Math.abs(dr) > 12) R.sign = Math.sign(dr);
        d.z = R.sign * dz;
        const q = new THREE.Quaternion().setFromUnitVectors(R.dir, d.clone().normalize());
        if (R.init) R.g.quaternion.slerp(q, 0.5); else { R.g.quaternion.copy(q); R.init = true; }
        R.g.position.copy(A); R.g.scale.setScalar(sA * R.k); R.g.visible = true; R.hold = 8; R.g.updateMatrixWorld(true);
        return A.clone().add(d.setLength(unit * R.k)); // where this segment actually ends
      };
      const pa = P(a), pb = P(b), pc = P(c);
      let elbow = null;
      if (Math.min(vis(l, a), vis(l, b)) >= 0.4) elbow = fit(U, pa, pb, pa.z, pb.z);
      if (!elbow) { if (U.hold > 0) { U.hold--; } else U.g.visible = false; }
      let wrist = pc;
      if (hs && hs[sd]) wrist = new V3(hs[sd][0].x * W, -hs[sd][0].y * H, pc.z); // the hand tracker knows the wrist better
      if (Math.min(vis(l, b), vis(l, c)) >= 0.4 || (hs && hs[sd])) { if (!(elbow && fit(F, elbow, wrist, pb.z, pc.z))) { if (F.hold > 0) F.hold--; else F.g.visible = false; } }
      else if (F.hold > 0) F.hold--; else F.g.visible = false;
    }
  }
  // 3D hand bones from hand landmarks
  poseHands(hs, pl) {
    for (const sd of ['P', 'M']) for (const r of Object.values(this.hrig[sd])) r.g.visible = false;
    if (!this.hands3d || !S.hands) return;
    const wr = pl && this.torsoOn ? { P: pl[15], M: pl[16] } : null;
    for (const sd of ['P', 'M']) {
      const h = hs[sd];
      if (!h) continue;
      const R = this.hrig[sd], z0 = wr ? -wr[sd].z * W * 0.5 + this.tZoff : 0;
      const p = (k) => new V3(h[k].x * W, -h[k].y * H, -(h[k].z - h[0].z) * W + z0);
      const k = p(9).distanceTo(p(0)) / R.carp.len;
      if (!(k > 0)) continue;
      for (const [key, r] of Object.entries(R)) {
        const [a, b] = SEG[key], A = p(a), B = p(b), d = B.clone().sub(A), L = d.length();
        if (L < 1) continue;
        const sc = key === 'carp' ? k : Math.min(1.35 * k, Math.max(0.75 * k, L / r.len));
        r.g.position.copy(r.anchor === 'dist' ? B : A);
        r.g.quaternion.setFromUnitVectors(r.axis, d.normalize());
        r.g.scale.setScalar(sc); r.g.visible = true;
      }
    }
  }
  // Pull the torso toward the head's neck base so the two separately-tracked parts stay joined
  joinNeck() {
    if (!this.neckBase || !this.t1 || !this.root.visible || !this.troot.visible || S.neckjoin <= 0) return;
    const d = this.root.localToWorld(this.neckBase.clone()).sub(this.troot.localToWorld(this.t1.clone()));
    d.z = 0;
    const lim = 60 * this.tScale;
    if (d.length() > lim) d.setLength(lim);
    this.troot.position.addScaledVector(d, S.neckjoin);
    this.troot.updateMatrixWorld(true);
  }
  // Labels (screen-space anchors, image coordinates)
  labels(hasFace, hands) {
    const items = [];
    if (this.troot.visible && S.labels) {
      const parentOf = (f) => (f === 'T' ? this.troot : f === 'N' ? (hasFace ? this.root : null) : this.rig[f]?.g.visible ? this.rig[f].inner : null);
      const pools = [TPL.torso?.labels || [], TPL.musc?.labels || []];
      let c = 0; const idx = [0, 0];
      while (c < S.ntlabels && (idx[0] < pools[0].length || idx[1] < pools[1].length)) {
        for (let k = 0; k < 2 && c < S.ntlabels; k++) {
          const d = pools[k][idx[k]++];
          if (!d || !S[d.l]) continue;
          const par = parentOf(d.f);
          if (!par) continue;
          let best = null;
          for (const a of d.a) { const w = par.localToWorld(a.clone()); if (!best || w.z > best.z) best = w; }
          items.push({ t: d.t, x: best.x, y: -best.y }); c++;
        }
      }
    }
    if (hasFace && S.labels && TPL.head) {
      let c = 0;
      for (const d of TPL.head.labels) {
        if (c >= S.nlabels) break;
        if (!S[d.l]) continue;
        const parent = d.f === 'HG' ? this.pivot : this.root;
        let best = null;
        for (const a of d.a) { const w = parent.localToWorld(a.clone()); if (!best || w.z > best.z) best = w; }
        items.push({ t: d.t, x: best.x, y: -best.y }); c++;
      }
    }
    if (S.hands && S.labels && hands[0]) {
      const h = hands[0], q = (i) => [h[i].x * W, h[i].y * H], mid = (a, b) => [(q(a)[0] + q(b)[0]) / 2, (q(a)[1] + q(b)[1]) / 2];
      for (const [t, [x, y]] of [['Metacarpal', mid(5, 0)], ['Proximal phalanx', mid(9, 10)], ['Middle phalanx', mid(10, 11)], ['Distal phalanx', q(8)]]) items.push({ t, x, y });
    }
    return items;
  }
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

// Labels are drawn on an un-mirrored canvas, so image x is flipped to screen x for the selfie camera.
function drawLabels(items, info) {
  lab.clearRect(0, 0, W, H);
  const fs = Math.max(10, Math.round(W / 52));
  lab.font = `${fs}px system-ui, sans-serif`; lab.textBaseline = 'middle';
  const sides = [[], []];
  for (const it of items) { it.sx = mirrored ? W - it.x : it.x; sides[it.sx < W / 2 ? 0 : 1].push(it); }
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

// ---------- Tracking: several people ----------
// One-Euro filter: steady when you hold still, quick when you move (less shake AND less lag than plain smoothing)
class OneEuro {
  constructor() { this.x = null; this.dx = 0; this.t = null; }
  f(v, t, mc, beta) {
    if (this.t === null) { this.x = v; this.t = t; return v; }
    const dt = Math.max(1e-3, t - this.t); this.t = t;
    const al = (c) => 1 / (1 + 1 / (2 * Math.PI * c) / dt);
    this.dx += al(1) * ((v - this.x) / dt - this.dx);
    this.x += al(mc + beta * Math.abs(this.dx)) * (v - this.x);
    return this.x;
  }
}
class LmFilter {
  constructor() { this.f = []; }
  run(lms, t, beta = 12) {
    const mc = 4.5 - 3.6 * S.smooth;
    return lms.map((q, i) => { const F = (this.f[i] ||= [new OneEuro(), new OneEuro(), new OneEuro()]); return { x: F[0].f(q.x, t, mc, beta), y: F[1].f(q.y, t, mc, beta), z: F[2].f(q.z, t, mc, beta), visibility: q.visibility }; });
  }
}

let NP = 1;
const slots = [];
const newTrack = () => ({ anchor: null, lm: null, pl: null, hands: [], ff: new LmFilter(), pf: new LmFilter(), hf: [new LmFilter(), new LmFilter()] });
const getSlot = (i) => (slots[i] ||= Object.assign({ p: new Person(), w: 0 }, newTrack()));
const dist2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// Group the detected faces, bodies and hands into people.
function associate(faces, poses, hl) {
  const dets = poses.map((pl) => ({ pl, face: null, hands: [], a: [pl[0].x, pl[0].y] }));
  for (const f of faces) {
    let best = null, bd = 0.3;
    for (const d of dets) if (!d.face) { const x = dist2([f[1].x, f[1].y], d.a); if (x < bd) { bd = x; best = d; } }
    if (best) best.face = f; else dets.push({ pl: null, face: f, hands: [], a: [f[1].x, f[1].y] });
  }
  for (const h of hl) {
    let best = null, bd = 1e9;
    for (const d of dets) {
      const x = d.pl ? Math.min(dist2([d.pl[15].x, d.pl[15].y], [h[0].x, h[0].y]), dist2([d.pl[16].x, d.pl[16].y], [h[0].x, h[0].y])) : Math.abs(d.a[0] - h[0].x) + 0.15;
      if (x < bd) { bd = x; best = d; }
    }
    if (best && best.hands.length < 2) best.hands.push(h);
  }
  return dets;
}
// Keep each person in the same slot from frame to frame (so smoothing and models don't swap).
function assign(dets) {
  const pairs = [];
  dets.forEach((d, i) => slots.forEach((s, j) => { if (s && s.anchor && j < NP) pairs.push([dist2(d.a, s.anchor), i, j]); }));
  pairs.sort((a, b) => a[0] - b[0]);
  const di = new Map(), sj = new Set();
  for (const [dd, i, j] of pairs) { if (dd > 0.35 || di.has(i) || sj.has(j)) continue; di.set(i, j); sj.add(j); }
  dets.forEach((d, i) => { if (di.has(i)) return; for (let j = 0; j < NP; j++) if (!sj.has(j)) { di.set(i, j); sj.add(j); break; } });
  return di;
}

async function makeLandmarkers(np) {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const mk = async (Cls, url, extra) => {
    for (const delegate of ['GPU', 'CPU']) {
      try { return await Cls.createFromOptions(fileset, { baseOptions: { modelAssetPath: url, delegate }, runningMode: 'VIDEO', ...extra }); }
      catch (e) { console.warn(Cls.name, delegate, e); }
    }
    throw new Error('Could not load ' + Cls.name);
  };
  const pose = await mk(PoseLandmarker, POSE_MODEL, { numPoses: np }).catch((e) => (console.warn('pose unavailable', e), null));
  return { face: await mk(FaceLandmarker, FACE_MODEL, { numFaces: np }), hand: await mk(HandLandmarker, HAND_MODEL, { numHands: 2 * np }), pose };
}

// ---------- Camera ----------
let mirrored = true, facing = 'user', stream = null, trackers = null, busy = false, resetTrack = false;
const vid = $('vid');
function applySize() {
  W = vid.videoWidth; H = vid.videoHeight;
  for (const id of ['ov', 'labels']) { $(id).width = W; $(id).height = H; }
  renderer.setSize(W, H, false);
  Object.assign(cam, { left: 0, right: W, top: 0, bottom: -H }); cam.updateProjectionMatrix();
  fit();
}
async function startCamera(face) {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: face }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  vid.srcObject = stream; await vid.play();
  facing = face; mirrored = face === 'user';
  $('mirror').classList.toggle('nm', !mirrored);
  applySize(); resetTrack = true;
  document.querySelectorAll('[data-cam]').forEach((b) => b.classList.toggle('on', b.dataset.cam === face));
}
async function setPeople(n) {
  NP = n;
  document.querySelectorAll('[data-ppl]').forEach((b) => b.classList.toggle('on', +b.dataset.ppl === n));
  if (!trackers) return;
  busy = true;
  try {
    for (const t of Object.values(trackers)) t?.close?.();
    trackers = await makeLandmarkers(n);
  } catch (e) { console.error(e); }
  for (let i = n; i < slots.length; i++) if (slots[i]) { slots[i].p.remove(); slots[i] = null; }
  resetTrack = true; busy = false;
}

// ---------- Snap ----------
let snapReq = false, lastBlobUrl = null, lastBlob = null;
const toast = (t) => { const el = $('toast'); el.textContent = t; el.classList.add('show'); clearTimeout(toast.t); toast.t = setTimeout(() => el.classList.remove('show'), 2200); };
function composeSnap() {
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const x = c.getContext('2d');
  x.save();
  if (mirrored) { x.translate(W, 0); x.scale(-1, 1); }
  x.drawImage(vid, 0, 0, W, H);
  x.fillStyle = `rgba(5,10,20,${1 - S.real})`; x.fillRect(0, 0, W, H);
  x.drawImage($('ov'), 0, 0); x.drawImage($('gl'), 0, 0, W, H);
  x.restore();
  x.drawImage($('labels'), 0, 0);
  c.toBlob((blob) => {
    if (!blob) return toast('Could not save photo');
    lastBlob = blob;
    if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
    lastBlobUrl = URL.createObjectURL(blob);
    $('thumb').src = lastBlobUrl; $('thumbBtn').classList.add('show');
    const name = `xray-mirror-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.jpg`;
    lastBlob.fname = name;
    const a = document.createElement('a'); a.href = lastBlobUrl; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    toast('Photo saved — tap the thumbnail to share');
  }, 'image/jpeg', 0.92);
}
$('thumbBtn').onclick = async () => {
  if (!lastBlob) return;
  const file = new File([lastBlob], lastBlob.fname || 'xray-mirror.jpg', { type: 'image/jpeg' });
  try { if (navigator.canShare?.({ files: [file] })) { await navigator.share({ files: [file], title: 'X-Ray Mirror' }); return; } } catch { return; }
  const a = document.createElement('a'); a.href = lastBlobUrl; a.download = file.name; a.click();
};
let timerOn = 0;
function doSnap() {
  if (timerOn) return;
  const sec = +(store.get('timer') || 0);
  if (!sec) { $('flash').classList.add('go'); setTimeout(() => $('flash').classList.remove('go'), 180); snapReq = true; return; }
  let n = sec; timerOn = 1;
  const c = $('count'); c.textContent = n; c.classList.add('show');
  const iv = setInterval(() => {
    n--;
    if (n > 0) c.textContent = n;
    else { clearInterval(iv); c.classList.remove('show'); timerOn = 0; $('flash').classList.add('go'); setTimeout(() => $('flash').classList.remove('go'), 180); snapReq = true; }
  }, 1000);
}
$('snap').onclick = doSnap;
$('bFlip').onclick = async () => { try { toast('Switching camera…'); await startCamera(facing === 'user' ? 'environment' : 'user'); } catch (e) { toast('Could not switch camera'); console.warn(e); } };
document.querySelectorAll('[data-cam]').forEach((b) => (b.onclick = async () => { if (b.dataset.cam !== facing) { try { await startCamera(b.dataset.cam); } catch (e) { toast('Camera not available'); } } }));
document.querySelectorAll('[data-ppl]').forEach((b) => (b.onclick = () => setPeople(+b.dataset.ppl)));
document.querySelectorAll('[data-timer]').forEach((b) => {
  b.classList.toggle('on', +b.dataset.timer === +(store.get('timer') || 0));
  b.onclick = () => { store.set('timer', b.dataset.timer); document.querySelectorAll('[data-timer]').forEach((x) => x.classList.toggle('on', x === b)); };
});
document.querySelectorAll('[data-trk]').forEach((b) => {
  S.fast = store.get('fast') === '1' ? 1 : 0;
  b.classList.toggle('on', (b.dataset.trk === 'fast') === !!S.fast);
  b.onclick = () => { S.fast = b.dataset.trk === 'fast' ? 1 : 0; store.set('fast', S.fast); document.querySelectorAll('[data-trk]').forEach((x) => x.classList.toggle('on', x === b)); };
});
document.querySelectorAll('[data-fit]').forEach((b) => {
  b.classList.toggle('on', (store.get('fill') === '1') === (b.dataset.fit === 'fill'));
  b.onclick = () => { fill = b.dataset.fit === 'fill'; store.set('fill', fill ? '1' : '0'); document.querySelectorAll('[data-fit]').forEach((x) => x.classList.toggle('on', x === b)); fit(); };
});

$('go').onclick = async () => {
  const st = $('st'), go = $('go');
  go.style.display = 'none';
  try {
    st.textContent = 'Starting camera…';
    await startCamera('user');
    st.textContent = 'Loading anatomy models…';
    await loadModels();
    st.textContent = 'Loading tracking models…';
    trackers = await makeLandmarkers(NP);
    $('msg').style.display = 'none';
    loadTorso(); loadMuscles(); loadHands(); loadFace();

    let faces = [], poses = [], handsRaw = [], last = -1, n = 0, fps = 0, tPrev = performance.now();
    const loop = () => {
      requestAnimationFrame(loop);
      if (busy || !trackers || vid.readyState < 2 || vid.currentTime === last) return;
      last = vid.currentTime;
      const t = performance.now();
      fps = Math.round(0.9 * fps + 0.1 * (1000 / (t - tPrev))); tPrev = t;
      if (resetTrack) { slots.forEach((s) => s && Object.assign(s, newTrack())); faces = []; poses = []; handsRaw = []; resetTrack = false; }
      const { face, hand, pose } = trackers;
      faces = face.detectForVideo(vid, t).faceLandmarks;
      n++;
      let freshH = false, freshP = false;
      const poseOn = pose && (S.tbones || S.organs || S.tmusc || S.abones || S.amusc);
      const every = S.fast ? 2 : 1;
      if (S.hands && n % every === 0) { handsRaw = hand.detectForVideo(vid, t).landmarks; freshH = true; }
      if (!S.hands) handsRaw = [];
      if (poseOn && (S.fast ? n % 2 === 1 : true)) { poses = pose.detectForVideo(vid, t).landmarks; freshP = true; }
      if (!poseOn) poses = [];

      $('dim').style.opacity = 1 - S.real;
      ov.clearRect(0, 0, W, H);
      const dets = associate(faces, poses, handsRaw), map = assign(dets), active = new Set();
      let lead = null;
      dets.forEach((d, i) => {
        const j = map.get(i);
        if (j === undefined) return;
        active.add(j);
        const s = getSlot(j), p = s.p;
        s.anchor = d.a;
        const ts = t / 1000;
        s.lm = d.face ? s.ff.run(d.face, ts, 14) : (s.ff = new LmFilter(), null);
        s.pl = d.pl ? (freshP || !s.pl ? s.pf.run(d.pl, ts, 8) : s.pl) : (s.pf = new LmFilter(), null);
        const hs = d.hands.slice().sort((a, b) => a[0].x - b[0].x);
        s.hands = hs.length ? (freshH || !s.hands.length ? hs.map((h, k) => s.hf[k].run(h, ts, 14)) : s.hands) : [];
        p.sync();
        if (s.lm && S.outline) drawOutline(s.lm);
        p.root.visible = !!s.lm;
        s.w = s.lm ? p.poseHead(s.lm) : 0;
        p.torsoOn = !!s.pl && p.poseTorso(s.pl, s.lm ? p.root.position.z : 0);
        p.troot.visible = p.torsoOn;
        p.joinNeck();
        const hsd = p.handSides(s.hands, s.pl);
        if (p.torsoOn) p.poseArms(s.pl, S.hands && p.hands3d ? hsd : null); else for (const r of Object.values(p.rig)) r.g.visible = false;
        p.poseHands(hsd, s.pl);
        p.look();
        if (!lead || s.w > lead.w) lead = s;
      });
      slots.forEach((s, j) => { if (s && !active.has(j)) { s.p.hide(); Object.assign(s, newTrack()); } });
      if (S.hands && !slots.some((s) => s && s.p.hands3d)) drawHands(handsRaw);
      for (const m of Object.values(mats)) m.x.uniforms.k.value = S.bright;
      renderer.render(scene, cam);
      const items = lead ? lead.p.labels(!!lead.lm, lead.hands) : [];
      drawLabels(items, `${fps} fps · people ${active.size}/${NP} · face ${lead?.lm ? 'yes' : 'no'} · hands ${handsRaw.length} · torso ${lead?.p.torsoOn ? 'yes' : TPL.torso ? 'no' : 'loading'} · arms ${lead ? ['UP', 'UM'].filter((k) => lead.p.rig[k].g.visible).length : 0}`);
      if (snapReq) { snapReq = false; composeSnap(); }
    };
    loop();
  } catch (e) {
    console.error(e);
    st.textContent = 'Could not start: ' + (e.message || e);
    go.style.display = ''; go.textContent = 'Try again';
    $('msg').style.display = '';
  }
};
