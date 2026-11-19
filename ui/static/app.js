"use strict";

// ---------------- state ----------------
const SR_IN = 16000;
let ws = null;
let audioCtx = null;
let mediaStream = null;
let processor = null;
let analyser = null;
let micData = null;      // Float32Array for the level meter
let running = false;
let state = "idle";      // idle | listening | thinking | speaking
let botAudioCtx = null;  // AudioContext for playback (created on gesture)
let botQueue = [];       // PCM chunks waiting for the audio context to unlock
let botPlaying = false;
let nextStartTime = 0;   // AudioContext clock time the next chunk starts at
let scheduledSrcs = new Set(); // scheduled-but-unfinished BufferSources
let chunkSr = null;      // sample rate of the utterance being scheduled
let heardVoice = false;  // one-time confirm beep on first audio
let reconnectTries = 0;  // self-heal after a server restart (see onclose)
let reconnectTimer = null; // pending setTimeout(connect) so it can be cancelled
let sessionId = 0;       // bumped on every connect/cancel; stale startMic runs bail out
// Stable per-tab identity (survives the idle-timeout disconnect) so the
// server can resume the same conversation when the orb is tapped again.
let client_id = (() => {
  try {
    let id = localStorage.getItem("lava_cid");
    if (!id) {
      id = "c_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
      localStorage.setItem("lava_cid", id);
    }
    return id;
  } catch (e) {
    return "c_anon";
  }
})();
let botGain = null;
let botAnalyser = null;
let botMicData = null;

const orbWrap = document.getElementById("orbWrap");
const stateLabel = document.getElementById("stateLabel");
const chat = document.getElementById("chat");
const soundBanner = document.getElementById("soundBanner");
const micBanner = document.getElementById("micBanner");
const micBannerMsg = document.getElementById("micBannerMsg");
const micRetryBtn = document.getElementById("micRetryBtn");

// States:
//   idle      - slow spin, few soft blobs, low glow
//   listening - mic level feeds a bulge aimed at the camera
//   thinking  - fast spin + longitude shear (vortex), hot core
//   speaking  - radius swells with the TTS playback level
const canvas = document.getElementById("wave");
const ctx2d = canvas.getContext("2d");
const DPR = Math.min(2.5, window.devicePixelRatio || 1);
let SIZE = 600, ORB_R = 300;
function fitCanvas() {
  const rect = canvas.getBoundingClientRect();
  SIZE = Math.max(64, Math.round(rect.width * DPR)) || 600;
  canvas.width = SIZE; canvas.height = SIZE;
  ORB_R = SIZE * 0.335;
}
fitCanvas();
window.addEventListener("resize", fitCanvas);

// Low-res render target, upscaled when drawn. 96 (down from 128) cuts the
// per-frame pixel work by ~44% - the orb is soft anyway, and the Pi needs
// that main-thread time for audio scheduling.
const RES = 96;
const buf = document.createElement("canvas");
buf.width = RES; buf.height = RES;
const bctx = buf.getContext("2d");
const img = bctx.createImageData(RES, RES);
const px = img.data;

// duotone ramp: deep maroon -> burnt orange -> vivid orange -> peach -> cream
const STOPS = [
  [0.00, 26, 11, 6],
  [0.28, 74, 24, 8],
  [0.46, 142, 44, 10],
  [0.62, 208, 74, 14],
  [0.76, 240, 116, 34],
  [0.87, 251, 162, 82],
  [0.95, 255, 208, 152],
  [1.00, 255, 240, 220],
];
const LUT = new Uint8ClampedArray(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  let k = 0;
  while (k < STOPS.length - 2 && t > STOPS[k + 1][0]) k++;
  const a = STOPS[k], b = STOPS[k + 1];
  const f = Math.min(1, Math.max(0, (t - a[0]) / (b[0] - a[0])));
  LUT[i * 3] = a[1] + (b[1] - a[1]) * f;
  LUT[i * 3 + 1] = a[2] + (b[2] - a[2]) * f;
  LUT[i * 3 + 2] = a[3] + (b[3] - a[3]) * f;
}

// per-state motion targets (interpolated smoothly every frame)
const TARGETS = {
  idle: { rot: 0.12, energy: 0.32, swirl: 0.10, bright: 0.85, blobs: 4 },
  listening: { rot: 0.28, energy: 0.55, swirl: 0.22, bright: 0.97, blobs: 5 },
  thinking: { rot: 1.10, energy: 1.00, swirl: 1.00, bright: 1.12, blobs: 7 },
  speaking: { rot: 0.42, energy: 0.70, swirl: 0.38, bright: 1.00, blobs: 6 },
};
const P = { rot: 0.12, energy: 0.32, swirl: 0.10, bright: 0.85, blobs: 4 };

function frac(x) { return x - Math.floor(x); }
// metaballs drifting in the planet frame: each rides a slow great-circle
// orbit (two angular frequencies), living slightly above the surface.
const BLOBS = [];
for (let i = 0; i < 8; i++) {
  BLOBS.push({
    th0: frac(i * 0.618) * Math.PI * 2,          // start longitude
    ph0: (frac(i * 0.383) - 0.5) * Math.PI * 0.9, // start latitude
    w1: 0.10 + 0.22 * frac(i * 0.382),            // longitude drift
    w2: 0.06 + 0.16 * frac(i * 0.723),            // latitude wobble
    lift: 1.04 + 0.22 * frac(i * 0.517),          // shell radius
    sig: 0.30 + 0.22 * frac(i * 0.291),
    amp: 0.50 + 0.50 * frac(i * 0.769),
  });
}

// axial tilt (scene leans slightly, like the reference planet)
const TILT = 0.38;
const cosT = Math.cos(TILT), sinT = Math.sin(TILT);
// key light from upper-left-front (canvas y grows downward, so LY is
// negative to light from above); half vector for the specular sheen
const LX = -0.42, LY = -0.50, LZ = 0.76;
let HX = LX, HY = LY, HZ = LZ + 1;
{ const hl = Math.hypot(HX, HY, HZ); HX /= hl; HY /= hl; HZ /= hl; }

// Per-pixel static geometry and shading, precomputed once: the old loop
// recomputed sqrt/pow chains (limb darkening, ^12 specular, tilt transform)
// for all pixels on every frame - ~half the render cost for pixels that
// never change. The per-frame loop below only evaluates the dynamic field.
const N_PX = RES * RES;
const A = new Uint8Array(N_PX);            // alpha incl. limb feather (0 = outside)
const VX = new Float32Array(N_PX), VY = new Float32Array(N_PX);
const TY = new Float32Array(N_PX), TZP = new Float32Array(N_PX);
const SWK = new Float32Array(N_PX);        // (1 - tz*0.5) swirl factor
const LIT = new Float32Array(N_PX);        // 0.42 + 0.58*diffuse
const SPEC = new Float32Array(N_PX);       // specular^12 * 0.5
const LIMB = new Float32Array(N_PX);       // z^0.42
{
  const R0 = RES / 2;
  let i = 0;
  for (let y = 0; y < RES; y++) {
    const vy = (y - R0 + 0.5) / R0;
    for (let x = 0; x < RES; x++, i++) {
      const vx = (x - R0 + 0.5) / R0;
      const r2 = vx * vx + vy * vy;
      if (r2 > 1) continue;              // alpha stays 0 (outside the disc)
      const rr = Math.sqrt(r2);
      const z = Math.sqrt(1 - r2);
      A[i] = rr > 0.965 ? 255 * ((1 - rr) / 0.035) : 255;
      VX[i] = vx; VY[i] = vy;
      const ty = vy * cosT - z * sinT;
      const tz = vy * sinT + z * cosT;
      TY[i] = ty; TZP[i] = tz;
      SWK[i] = 1 - tz * 0.5;
      let diff = LX * vx + LY * vy + LZ * z;
      if (diff < 0) diff = 0;
      LIT[i] = 0.42 + 0.58 * diff;
      let spec = HX * vx + HY * vy + HZ * z;
      spec = spec <= 0 ? 0 : spec * spec;
      spec = spec * spec * spec;
      SPEC[i] = spec * spec * 0.5;
      LIMB[i] = Math.pow(z, 0.42);
    }
  }
}

// ---------- face ----------
// The expression is folded into the same metaball field as the lava: two
// small almond glows for the eyes and one longer molten seam for the mouth.
// Geometry lives in view space (vx/vy), so the face keeps facing the camera
// while the lava spins under it, and each feature is weighted by the field
// already beneath it (see renderPlanet) so the face ignites where the
// surface is hot - part of the planet, never a sticker on it.
const FACE_MY = 0.21;                      // mouth rest height (disc spans [-1, 1])
const FACE_MW = 0.34;                      // mouth half-width
const FACE_MINV = 1 / (2 * 0.046 * 0.046); // seam half-thickness
const EYE_X = 0.255, EYE_Y = -0.15;
const EYE_WI = 1 / (2 * 0.088 * 0.088);
const EYE_HI = 1 / (2 * 0.054 * 0.054);
const EYE_HIn = 1 / (2 * 0.014 * 0.014);   // squinted: eyes almost shut
const FMASK = new Uint8Array(N_PX);        // 1 inside the face's bounding box
const MSH = new Float32Array(N_PX);        // mouth envelope (tapers at the corners)
const EYEA = new Float32Array(N_PX);       // eyes open
const EYEB = new Float32Array(N_PX);       // eyes squinted (blend target)
{
  const R0 = RES / 2;
  let i = 0;
  for (let y = 0; y < RES; y++) {
    const vy = (y - R0 + 0.5) / R0;
    for (let x = 0; x < RES; x++, i++) {
      const vx = (x - R0 + 0.5) / R0;
      if (vx * vx + vy * vy > 1) continue;
      if (vx > 0.42 || vx < -0.42 || vy < -0.32 || vy > 0.44) continue;
      FMASK[i] = 1;
      const u = 1 - (vx / FACE_MW) * (vx / FACE_MW);
      MSH[i] = u > 0 ? u : 0;
      const dy = vy - EYE_Y;
      const dR = (vx - EYE_X) * (vx - EYE_X), dL = (vx + EYE_X) * (vx + EYE_X);
      EYEA[i] = Math.max(Math.exp(-(dR * EYE_WI + dy * dy * EYE_HI)),
        Math.exp(-(dL * EYE_WI + dy * dy * EYE_HI)));
      EYEB[i] = Math.max(Math.exp(-(dR * EYE_WI + dy * dy * EYE_HIn)),
        Math.exp(-(dL * EYE_WI + dy * dy * EYE_HIn)));
    }
  }
}

// Expression per connection phase: curve > 0 dips the seam's center into a
// smile, < 0 arches it into a frown; amp is the seam's heat, eye the glow of
// the eyes, squint narrows them (concern; blinks ride on top of it).
const FACE_EXPR = {
  online: { curve: 0.078, amp: 0.205, eye: 0.130, squint: 0.08 },
  retrying: { curve: 0.012, amp: 0.125, eye: 0.100, squint: 0.48 },
  offline: { curve: -0.055, amp: 0.105, eye: 0.088, squint: 0.20 },
};
let connPhase = "offline";  // offline | retrying | online (drives the face)
const FP = { curve: -0.055, amp: 0.105, eye: 0.088, squint: 0.20 };
let blinkAt = -9, nextBlink = 2 + Math.random() * 3;

let rotAngle = 0;
let speechLevel = 0;  // smoothed TTS playback RMS -> speaking swells
let micLevel = 0;     // smoothed mic RMS -> listening bulge
let lastT = performance.now();

function renderPlanet(t, dt) {
  // --- interpolate state params ---
  // Under prefers-reduced-motion the states still differ (state feedback
  // stays legible) but spin and swirl are damped to a drift.
  const T0 = TARGETS[state] || TARGETS.idle;
  const calm = REDUCED_MOTION.matches ? 0.25 : 1;
  const T = calm === 1 ? T0
    : {
      rot: T0.rot * calm, energy: T0.energy, swirl: T0.swirl * calm,
      bright: T0.bright, blobs: T0.blobs
    };
  const s = Math.min(1, dt * 2.5);
  P.rot += (T.rot - P.rot) * s;
  P.energy += (T.energy - P.energy) * s;
  P.swirl += (T.swirl - P.swirl) * s;
  P.bright += (T.bright - P.bright) * s;
  P.blobs += (T.blobs - P.blobs) * s;
  rotAngle += P.rot * dt * (1 + 0.6 * P.energy);

  // --- audio levels ---
  let sl = 0;
  if (botAnalyser && botPlaying && botMicData) {
    botAnalyser.getFloatTimeDomainData(botMicData);
    let sum = 0;
    for (let i = 0; i < botMicData.length; i++) sum += botMicData[i] * botMicData[i];
    sl = Math.sqrt(sum / botMicData.length) * 7;
  }
  speechLevel += (sl - speechLevel) * (sl > speechLevel ? 0.55 : 0.10);
  const ml = (state === "listening" && running) ? currentLevel() : 0;
  micLevel += (ml - micLevel) * (ml > micLevel ? 0.45 : 0.08);

  // --- expression: ease toward the connection phase's face ---
  const FT = FACE_EXPR[connPhase] || FACE_EXPR.offline;
  const fs = Math.min(1, dt * 1.7);
  FP.curve += (FT.curve - FP.curve) * fs;
  FP.amp += (FT.amp - FP.amp) * fs;
  FP.eye += (FT.eye - FP.eye) * fs;
  FP.squint += (FT.squint - FP.squint) * fs;
  // an occasional blink keeps the face alive; it rides on the squint blend
  if (t >= nextBlink) { blinkAt = t; nextBlink = t + 3.5 + Math.random() * 3.5; }
  const bph = t - blinkAt;
  const blink = (bph >= 0 && bph < 0.26) ? Math.sin(Math.PI * bph / 0.26) : 0;
  const fSquint = Math.min(1, FP.squint + blink);
  const fCurve = FP.curve + 0.01 * Math.sin(t * 0.45);   // gentle mood drift
  const fAmp = FP.amp * (0.86 + 0.14 * Math.sin(t * 0.9))   // seam shimmer
    + (state === "speaking" ? 0.2 * speechLevel : 0); // talk flare
  const fEye = FP.eye + (state === "listening" ? 0.05 * Math.min(1, micLevel) : 0);

  // --- blob positions in the planet frame ---
  const eOrb = Math.min(1, P.energy + (state === "listening" ? 0.35 * micLevel : 0));
  const spin = rotAngle;
  const cosS = Math.cos(spin), sinS = Math.sin(spin);
  const orb = [];
  let nB = 0;
  for (let i = 0; i < 8; i++) {
    const wgt = P.blobs - i;
    if (wgt <= 0.01) break;
    const b = BLOBS[i];
    const th = b.th0 + b.w1 * t * (0.6 + 0.8 * eOrb);
    const ph = b.ph0 + b.w2 * t;
    const r = b.lift + 0.06 * Math.sin(t * 0.8 + i);
    // spherical -> cartesian, then spin the planet about its (tilted) axis
    const cp = Math.cos(ph);
    let x = r * cp * Math.cos(th), y = r * Math.sin(ph), z = r * cp * Math.sin(th);
    const xs = x * cosS + z * sinS, zs = -x * sinS + z * cosS;
    const sg = b.sig * (0.8 + 0.35 * eOrb);
    orb.push(xs, y, zs, 1 / (2 * sg * sg), b.amp * Math.min(1, wgt));
    nB++;
  }
  // barge-in bulge: a soft blob pushed toward the camera when you speak
  const bulgeAmp = (state === "listening" ? 0.9 * micLevel : 0);
  const bBx = 0.22, bBy = 0.16, bBz = 0.96;

  // --- fill the low-res disc ---
  const swirlMax = P.swirl * 1.6;
  const baseF = 0.30 + 0.05 * Math.sin(t * 0.6);
  const bright = P.bright;
  let idx = 0;
  for (let i = 0; i < N_PX; i++, idx += 4) {
    const a = A[i];
    if (a === 0) { px[idx + 3] = 0; continue; }
    const vx = VX[i], vy = VY[i], tz = TZP[i];

    // ---- differential spin about Y (core turns faster than the rim) ----
    const sw = spin + swirlMax * SWK[i];
    const csw = Math.cos(sw), ssw = Math.sin(sw);
    const ux = vx * csw + tz * ssw;
    const uz = -vx * ssw + tz * csw;
    const uy = TY[i];

    // ---- metaball field ----
    let F = baseF
      + 0.07 * Math.sin(2.2 * ux + 1.3 * t) * Math.sin(1.8 * uy - 1.1 * t)
      + 0.05 * Math.sin(2.8 * uz - 0.9 * t) * Math.sin(2.4 * ux + 0.7 * t);
    for (let b = 0; b < nB; b++) {
      const o = b * 5;
      const dx = ux - orb[o], dy = uy - orb[o + 1], dz = uz - orb[o + 2];
      F += orb[o + 4] * Math.exp(-(dx * dx + dy * dy + dz * dz) * orb[o + 3]);
    }
    if (bulgeAmp > 0.01) {
      const dx = ux - bBx, dy = uy - bBy, dz = uz - bBz;
      F += bulgeAmp * Math.exp(-(dx * dx + dy * dy + dz * dz) * 3.2);
    }
    // ---- expression: eyes + seam mouth, weighted by the field already
    // under them so the face ignites with the surface instead of sitting
    // on top of it ----
    if (FMASK[i]) {
      const heat = 0.7 + 0.45 * (F < 1 ? F : 1);
      const m = MSH[i];
      const d = vy - FACE_MY - fCurve * m;
      F += fAmp * m * heat * Math.exp(-d * d * FACE_MINV);
      const ea = EYEA[i], eb = EYEB[i];
      F += fEye * heat * (ea + (eb - ea) * fSquint);
    }
    F *= 0.86;
    let ci = (F * 255) | 0;
    if (ci > 255) ci = 255; else if (ci < 0) ci = 0;

    // ---- precomputed spherical shading (key light, specular, limb) ----
    const shade = LIT[i] * LIMB[i] * bright;
    const sp = SPEC[i] * LIMB[i] * bright;
    px[idx] = LUT[ci * 3] * shade + 255 * sp;
    px[idx + 1] = LUT[ci * 3 + 1] * shade + 236 * sp;
    px[idx + 2] = LUT[ci * 3 + 2] * shade + 210 * sp;
    px[idx + 3] = a;
  }
  bctx.putImageData(img, 0, 0);

  // --- composite: atmosphere glow, then the planet ---
  const swell = 1 + 0.07 * speechLevel * (state === "speaking" ? 1 : 0.4);
  const drawR = ORB_R * swell;
  ctx2d.clearRect(0, 0, SIZE, SIZE);
  const glowR = drawR * 1.5;
  const g = ctx2d.createRadialGradient(SIZE / 2, SIZE / 2, drawR * 0.55, SIZE / 2, SIZE / 2, glowR);
  const ga = Math.min(0.5, 0.14 + 0.30 * P.energy);
  g.addColorStop(0, `rgba(255,110,26,${(ga * 0.55).toFixed(3)})`);
  g.addColorStop(0.55, `rgba(210,70,14,${(ga * 0.22).toFixed(3)})`);
  g.addColorStop(1, "rgba(120,35,8,0)");
  ctx2d.fillStyle = g;
  ctx2d.fillRect(0, 0, SIZE, SIZE);
  ctx2d.imageSmoothingEnabled = true;
  ctx2d.imageSmoothingQuality = "high";
  ctx2d.drawImage(buf, SIZE / 2 - drawR, SIZE / 2 - drawR, drawR * 2, drawR * 2);
}

// Frame-rate cap: the orb's motion is slow, and ~30 fps is visually
// identical for this animation while freeing real main-thread time for the
// audio scheduling and WebSocket work that share this thread (important on
// a Raspberry Pi, where Chromium, Piper and the orb share four cores).
const FRAME_MIN_MS = 33;
// prefers-reduced-motion: drop to ~4 fps, advance the ambient clock at 0.15x
// and damp spin/swirl (see renderPlanet) - the planet becomes a slow drift
// while states still ease and audio swells still register.
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");
const FRAME_MIN_MS_REDUCED = 250;
let ambientT = 0;  // ambient clock; runs slower than wall time when reduced
function frame(now) {
  const reduced = REDUCED_MOTION.matches;
  const minMs = reduced ? FRAME_MIN_MS_REDUCED : FRAME_MIN_MS;
  const dt = (now - lastT) / 1000;
  if (dt < minMs / 1000) {
    requestAnimationFrame(frame);
    return;
  }
  const step = Math.min(reduced ? 0.30 : 0.05, dt);
  lastT = now;
  ambientT += step * (reduced ? 0.15 : 1);
  renderPlanet(ambientT, step);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function currentLevel() {
  if (!analyser || !micData) return 0;
  analyser.getFloatTimeDomainData(micData);
  let sum = 0;
  for (let i = 0; i < micData.length; i++) sum += micData[i] * micData[i];
  return Math.sqrt(sum / micData.length) * 3.2; // normalized-ish
}

// ---------------- state helpers ----------------
function setState(s) {
  state = s;
  orbWrap.className = "orb-wrap" + (s === "idle" ? "" : " " + s);
  document.body.classList.remove("listening", "thinking", "speaking", "idle");
  document.body.classList.add(s);
  // the orb is a real button (role="button"); keep its name on the state
  orbWrap.setAttribute("aria-label",
    s === "idle" ? "Start talking to Lava" : "Stop talking to Lava");
  if (s === "idle") {
    stateLabel.textContent = "tap to talk";
    stateLabel.classList.remove("hide");
  } else {
    stateLabel.classList.remove("hide");
    stateLabel.textContent = s;
  }
}
setState("idle");

// ---------------- chat ----------------
// Autoscroll follows the stream only while the reader is already at the
// bottom (~80px of grace); scrolled up to read, the view stays put until
// they return. force=true after the user's own turn (they just acted).
function chatFollow(force) {
  if (!force && chat.scrollHeight - chat.scrollTop - chat.clientHeight > 80) return;
  chat.scrollTop = chat.scrollHeight;
}

function addUser(text) {
  const row = document.createElement("div");
  row.className = "row user";
  // Screen-reader attribution (visually hidden, see .sr-only): rows are only
  // visually distinct, so a screen reader otherwise can't tell who spoke.
  // Kept OUTSIDE .bubble — the repair path rewrites bubble text via textContent.
  const who = document.createElement("span");
  who.className = "sr-only";
  who.textContent = "You said: ";
  row.appendChild(who);
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  row.appendChild(b);
  chat.appendChild(row);
  chatFollow(true);
}

let botRow = null, botBubble = null;
function ensureBotRow() {
  if (botRow) return;
  botRow = document.createElement("div");
  botRow.className = "row bot";
  // Hidden "Lava: " attribution, outside the bubble — streaming rewrites
  // botBubble.textContent per chunk, which would destroy an inline label.
  const who = document.createElement("span");
  who.className = "sr-only";
  who.textContent = "Lava: ";
  botRow.appendChild(who);
  botBubble = document.createElement("div");
  botBubble.className = "bubble";
  botRow.appendChild(botBubble);
  chat.appendChild(botRow);
  chatFollow();
}
function finalizeBotRow() {
  botRow = null; botBubble = null;
}

// ---------------- bot audio playback ----------------
function ensureBotCtx() {
  if (!botAudioCtx) {
    // "playback" asks the browser for a larger output buffer: slightly more
    // latency, far fewer underruns on a busy Raspberry Pi.
    botAudioCtx = new (window.AudioContext || window.webkitAudioContext)({
      latencyHint: "playback",
    });
    botGain = botAudioCtx.createGain();
    botGain.connect(botAudioCtx.destination);
    botAnalyser = botAudioCtx.createAnalyser();
    botAnalyser.fftSize = 1024;
    botMicData = new Float32Array(botAnalyser.fftSize);
    botGain.connect(botAnalyser);
    // Keep the output device awake with an inaudible signal, so the first
    // word of a reply doesn't hit a cold/suspended sink (gappy start).
    if (botAudioCtx.createConstantSource) {
      try {
        const keepAlive = botAudioCtx.createConstantSource();
        const kaGain = botAudioCtx.createGain();
        kaGain.gain.value = 0.00001;
        keepAlive.connect(kaGain);
        kaGain.connect(botAudioCtx.destination);
        keepAlive.start();
      } catch (e) { }
    }
    botAudioCtx.addEventListener("statechange", updateSoundBanner);
  }
  unlockAudio();
}


function unlockAudio() {
  if (!botAudioCtx) return;
  if (botAudioCtx.state === "suspended") {
    const p = botAudioCtx.resume();
    if (p && p.then) p.then(() => { drainBotQueue(); }).catch(() => { });
  } else {
    drainBotQueue();
  }
  updateSoundBanner();
}

function updateSoundBanner() {
  const blocked = !!(botAudioCtx && botAudioCtx.state !== "running");
  soundBanner.classList.toggle("show", blocked && running);
}

// Short "audio is working" chirp played once when the session starts.
function playConfirmBeep() {
  if (!botAudioCtx || botAudioCtx.state !== "running") return;
  const t = botAudioCtx.currentTime;
  const o = botAudioCtx.createOscillator();
  const g = botAudioCtx.createGain();
  o.type = "sine";
  o.frequency.value = 880;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.25, t + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.18);
  o.connect(g); g.connect(botAudioCtx.destination);
  o.start(t); o.stop(t + 0.2);
}

// Start-of-reply gather: the first packet of a reply is held for up to
// ~80 ms (or until ~120 ms of audio has arrived) so it is never scheduled
// alone with a gap before the second packet.
const START_GATHER_MS = 120;
const START_WAIT_MS = 80;
// A silence longer than this between chunks is a new reply, not an underrun.
const NEW_REPLY_GAP_S = 1.5;
let pending = [], pendingMs = 0, pendingTimer = null;

function clearPending() {
  if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
  pending = [];
  pendingMs = 0;
}

function flushPending() {
  if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
  const items = pending;
  pending = [];
  pendingMs = 0;
  if (!botAudioCtx || botAudioCtx.state !== "running") {
    // context got suspended while we were gathering: keep the audio queued
    for (const it of items) botQueue.push(it);
    return;
  }
  for (const it of items) scheduleChunk(it.data, it.sr);
}

function replyStarting() {
  if (scheduledSrcs.size > 0) return false;
  if (nextStartTime === 0) return true;
  return botAudioCtx.currentTime - nextStartTime > NEW_REPLY_GAP_S;
}

function enqueueAudio(i16, sr) {
  ensureBotCtx();
  // int16 PCM -> [-1, 1] for AudioBuffer.copyToChannel().
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  if (!botAudioCtx || botAudioCtx.state !== "running") {
    // Safari: suspended context — stay queued; the statechange listener
    // (and every tap, via unlockAudio) drains us when audio unlocks.
    botQueue.push({ data: f32, sr });
    unlockAudio();
    return;
  }
  if (pending.length || replyStarting()) {
    pending.push({ data: f32, sr });
    pendingMs += (f32.length / sr) * 1000;
    if (pendingMs >= START_GATHER_MS) flushPending();
    else if (!pendingTimer) pendingTimer = setTimeout(flushPending, START_WAIT_MS);
    return;
  }
  drainBotQueue();
  scheduleChunk(f32, sr);
}


// Jitter buffer. The server paces reply audio in real time (one 40 ms packet
// every 40 ms), so whatever is scheduled ahead of the playhead is the only
// cushion against late packets. On a Raspberry Pi (Chromium + the orb
// renderer + Piper share four cores) a small cushion gets missed and the
// start of the reply comes out gappy and "stretched". So the cushion starts
// generous and grows each time a gap actually happens mid-reply; the learned
// value is remembered per device. Barge-in is unaffected: stopBotAudio()
// kills scheduled sources instantly regardless of how far ahead they sit.
// (The storage key is versioned so values learned under the old, smaller
// limits don't pin the cushion too low.)
const LEAD_MIN = 0.15, LEAD_MAX = 0.60, LEAD_STEP = 0.10;
const LEAD_KEY = "lava.jitterLead2";
let jitterLead = LEAD_MIN;
try {
  const saved = parseFloat(localStorage.getItem(LEAD_KEY));
  if (saved) jitterLead = Math.min(LEAD_MAX, Math.max(LEAD_MIN, saved));
} catch (e) { }

// Continuous resampler. Every packet used to be handed to the browser as its
// own AudioBuffer at the TTS rate (e.g. 22050 Hz) while the context runs at
// 48000/44100 Hz, so each 40 ms packet was resampled IN ISOLATION: the
// interpolator has no history at the buffer edges, which leaves a tiny dip or
// tick at every packet boundary even when the timing is perfect (hence "gaps"
// with no underrun logged). Here the resampling happens ourselves, with the
// last samples carried across packets (Catmull-Rom), so the stream is seamless
// and the buffers handed to Web Audio already match the context rate.
let rsTail = new Float32Array(3);   // last 3 input samples of the previous packet
let rsPos = 3;                      // read position in [tail(3) + packet] coordinates
function resetResampler() { rsTail = new Float32Array(3); rsPos = 3; }

function resampleStream(cur, srIn, srOut) {
  if (srIn === srOut) return cur;
  const ratio = srIn / srOut;
  const n = cur.length, L = n + 3;
  const e = new Float32Array(L);
  e.set(rsTail, 0);
  e.set(cur, 3);
  const count = Math.max(0, Math.ceil((L - 2 - rsPos) / ratio));
  const out = new Float32Array(count);
  for (let k = 0; k < count; k++) {
    const pos = rsPos + k * ratio;
    const i = Math.floor(pos);
    const t = pos - i;
    const p0 = e[i - 1], p1 = e[i], p2 = e[i + 1], p3 = e[i + 2];
    out[k] = p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
  }
  rsPos = rsPos + count * ratio - n;
  rsTail = e.slice(L - 3);
  return out;
}

// Packet start times are computed from an integer frame count instead of
// summing float durations, so consecutive packets abut on exact sample frames.
let baseFrame = 0, framesSched = 0;

function scheduleChunk(f32, sr) {
  if (chunkSr !== sr) { chunkSr = sr; nextStartTime = 0; resetResampler(); }
  const now = botAudioCtx.currentTime;
  if (nextStartTime > 0 && nextStartTime < now) {
    if (now - nextStartTime > NEW_REPLY_GAP_S) {
      // Long silence: this is a new reply, not a gap inside one.
      nextStartTime = 0;
      resetResampler();
    } else {
      // Underrun: the previous chunk finished before this one arrived — an
      // audible gap. Grow the cushion for the rest of this reply and the next.
      jitterLead = Math.min(LEAD_MAX, jitterLead + LEAD_STEP);
      try { localStorage.setItem(LEAD_KEY, jitterLead.toFixed(2)); } catch (e) { }
      console.info(`[lava] playback underrun — jitter buffer now ${Math.round(jitterLead * 1000)} ms`);
      // Rebuild the cushion NOW. Without this, nextStartTime stays > 0, the
      // check below adds no lead, and the rest of the reply plays with zero
      // buffer (so every later late packet is another gap).
      nextStartTime = 0;
    }
  }
  const rate = botAudioCtx.sampleRate;
  const data = resampleStream(f32, sr, rate);
  if (!data.length) return;
  if (nextStartTime < now + (nextStartTime > 0 ? 0 : jitterLead)) {
    baseFrame = Math.ceil((now + jitterLead) * rate);
    framesSched = 0;
  }
  try {
    const audioBuf = botAudioCtx.createBuffer(1, data.length, rate);
    audioBuf.copyToChannel(data, 0);
    const src = botAudioCtx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(botGain);
    src.start((baseFrame + framesSched) / rate);
    framesSched += data.length;
    nextStartTime = (baseFrame + framesSched) / rate;
    scheduledSrcs.add(src);
    botPlaying = true;
    // Only touch the DOM on a real transition: this runs for every 40 ms
    // packet, and setState rewrites classes/aria/label each time.
    if (state !== "speaking") setState("speaking");
    src.onended = () => {
      scheduledSrcs.delete(src);
      if (scheduledSrcs.size === 0) {
        botPlaying = false;
        if (running) setState("listening");
      }
    };
  } catch (e) {
    // Never let one bad chunk wedge the whole schedule.
    console.error("[lava] audio playback failed:", e);
  }
}

function drainBotQueue() {
  while (botQueue.length && botAudioCtx && botAudioCtx.state === "running") {
    const item = botQueue.shift();
    scheduleChunk(item.data, item.sr);
  }
}

function stopBotAudio() {
  botQueue = [];
  clearPending();
  nextStartTime = 0;
  chunkSr = null;
  resetResampler();
  const srcs = [...scheduledSrcs];
  scheduledSrcs.clear();
  for (const src of srcs) {
    src.onended = null;
    try { src.stop(); } catch (e) { }
  }
  botPlaying = false;
  duckBot(false);
}

// Barge-in ducking: when speech starts over the bot, the server ducks (not
// cuts) the playback immediately - the interruption is heard instantly - and
// either restores the gain (the overlap turned out to be "mm-hmm") or sends
// "barge" to hard-cut. The watchdog restores after 2.5 s no matter what, so
// a lost message can never leave the bot whispering.
let duckTimer = null;
function duckBot(on) {
  if (!botGain || !botAudioCtx) return;
  const t = botAudioCtx.currentTime;
  botGain.gain.cancelScheduledValues(t);
  botGain.gain.setTargetAtTime(on ? 0.15 : 1.0, t, on ? 0.04 : 0.06);
  if (duckTimer) { clearTimeout(duckTimer); duckTimer = null; }
  if (on) duckTimer = setTimeout(() => { duckTimer = null; duckBot(false); }, 2500);
}

// ---------------- command notifications (toasts) ----------------
// The server sends tool_start (optimistic, while the tool runs) and
// tool_done (final) frames carrying pre-computed display strings:
//   {type, key, cat, title, sub, err}
// We render one toast per unique key (dedupe: two identical "pause"s are
// one toast), flip it in place on done, and auto-dismiss. States:
//   spin  -> running (spinner, no timer)
//   ok    -> success (check, dismisses in ~3.2 s)
//   err   -> failure (x + error line, dismisses in ~6 s)
const toastHost = document.getElementById("toastHost");
const TOAST_MAX = 4;          // visible at once; oldest dismisses first
const TOAST_OK_MS = 3200;
const TOAST_ERR_MS = 6000;
const ICON_CHECK = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5 6.2 11.7 13 4.5"/></svg>';
const ICON_X = '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>';
const ICON_DISMISS = '<svg viewBox="0 0 12 12" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M3 3l6 6M9 3l-6 6"/></svg>';
const toasts = new Map();     // key -> {el, state, timer, hover}

function toastEl(cat, title, sub) {
  const el = document.createElement("div");
  el.className = `toast cat-${cat || "unknown"}`;
  const icon = document.createElement("span");
  icon.className = "toast-icon";
  const body = document.createElement("div");
  body.className = "toast-body";
  const t = document.createElement("div");
  t.className = "toast-title";
  t.textContent = title || "Command";
  body.appendChild(t);
  if (sub && sub !== "done") {  // "done" says what the check glyph already says
    const s = document.createElement("div");
    s.className = "toast-sub";
    s.textContent = sub;
    body.appendChild(s);
  }
  const x = document.createElement("span");
  x.className = "toast-x";
  x.innerHTML = ICON_DISMISS;
  el.append(icon, body, x);
  return el;
}

function toastSetState(rec, state, sub, err) {
  const el = rec.el;
  el.classList.remove("is-spin", "is-ok", "is-err");
  const icon = el.querySelector(".toast-icon");
  if (state === "spin") {
    el.classList.add("is-spin");
    icon.innerHTML = "";
  } else if (state === "err") {
    el.classList.add("is-err");
    icon.innerHTML = ICON_X;
    const s = el.querySelector(".toast-sub");
    if (s) s.textContent = err || "Failed";
  } else {
    el.classList.add("is-ok");
    icon.innerHTML = ICON_CHECK;
    let s = el.querySelector(".toast-sub");
    if (sub && sub !== "done") {  // a bare "done" is noise — the check speaks
      if (!s) {
        s = document.createElement("div");
        s.className = "toast-sub";
        el.querySelector(".toast-body").appendChild(s);
      }
      s.textContent = sub;
    } else if (s) {
      s.remove();
    }
  }
  rec.state = state;
  // (re)arm the auto-dismiss timer; hover pauses it
  if (rec.timer) { clearTimeout(rec.timer); rec.timer = null; }
  if (state === "ok" && !rec.hover) rec.timer = setTimeout(() => toastDismiss(rec.key), TOAST_OK_MS);
  if (state === "err" && !rec.hover) rec.timer = setTimeout(() => toastDismiss(rec.key), TOAST_ERR_MS);
}

function toastDismiss(key) {
  const rec = toasts.get(key);
  if (!rec || rec.leaving) return;
  rec.leaving = true;
  if (rec.timer) { clearTimeout(rec.timer); rec.timer = null; }
  const el = rec.el;
  // collapse the box while it fades, so the stack reflows instead of jumping
  el.style.height = el.offsetHeight + "px";
  requestAnimationFrame(() => {
    el.classList.remove("in");
    el.classList.add("out");
    el.style.height = "0px";
    el.style.marginTop = "0px";
    el.style.paddingTop = "0px";
    el.style.paddingBottom = "0px";
    el.style.borderTopWidth = "0px";
    el.style.borderBottomWidth = "0px";
  });
  setTimeout(() => { el.remove(); toasts.delete(key); }, 240);
}

function toastShow(m) {
  // Dedupe: same key updates the existing toast in place (start -> done).
  let rec = toasts.get(m.key);
  if (!rec) {
    // Evict the oldest non-leaving toast past the cap.
    const live = [...toasts.values()].filter((r) => !r.leaving);
    if (live.length >= TOAST_MAX) toastDismiss(live[0].key);
    const el = toastEl(m.cat, m.title, m.sub);
    el.addEventListener("mouseenter", () => {
      rec.hover = true;
      if (rec.timer) { clearTimeout(rec.timer); rec.timer = null; }
    });
    el.addEventListener("mouseleave", () => {
      rec.hover = false;
      if (rec.state === "ok") rec.timer = setTimeout(() => toastDismiss(m.key), TOAST_OK_MS);
      if (rec.state === "err") rec.timer = setTimeout(() => toastDismiss(m.key), TOAST_ERR_MS);
    });
    el.addEventListener("click", () => toastDismiss(m.key));
    toastHost.appendChild(el);
    rec = { key: m.key, el, state: "spin", timer: null, hover: false, leaving: false };
    toasts.set(m.key, rec);
    // force a frame so the entrance transition runs
    requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add("in")));
  }
  // Refresh copy (title/sub can resolve on done: track, device, volume %).
  const t = rec.el.querySelector(".toast-title");
  if (t && m.title) t.textContent = m.title;
  if (m.error) toastSetState(rec, "err", null, m.error);
  else if (m.type === "tool_done") toastSetState(rec, "ok", m.sub, null);
  else toastSetState(rec, "spin", m.sub, null);
}

// ---------------- websocket ----------------
function connect() {
  // Never hold two sockets: drop any pending retry and any live socket.
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  if (ws) { ws.onclose = null; try { ws.close(); } catch (e) { } ws = null; }
  const mySession = ++sessionId;
  connPhase = "retrying";  // concerned face while attempting
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const sock = new WebSocket(`${proto}://${location.host}/ws?cid=${encodeURIComponent(client_id)}`);
  ws = sock;
  sock.binaryType = "arraybuffer";

  sock.onopen = () => {
    if (sock !== ws) return;
    reconnectTries = 0;
    connPhase = "online";  // the smile arrives with the connection
    startMic(mySession);
  };

  sock.onmessage = (ev) => {
    if (sock !== ws) return;
    // Binary frames are bot audio: uint16 LE sample rate + 16-bit LE mono PCM.
    if (ev.data instanceof ArrayBuffer) {
      if (!heardVoice) { heardVoice = true; playConfirmBeep(); }
      const sr = new DataView(ev.data).getUint16(0, true);
      enqueueAudio(new Int16Array(ev.data, 2), sr);
      return;
    }
    const m = JSON.parse(ev.data);
    if (m.type === "busy") {
      // Another device owns the session: stop retrying, say so plainly.
      running = false;
      reconnectTries = 0;
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      connPhase = "offline";
      stopMic();
      if (ws) { ws.onclose = null; ws.close(); ws = null; }
      orbWrap.setAttribute("aria-label", "Start talking to Lava");
      stateLabel.textContent = "in use on another device";
      stateLabel.classList.remove("hide");
    } else if (m.type === "duck") {
      duckBot(true);
    } else if (m.type === "unduck") {
      duckBot(false);
    } else if (m.type === "barge") {
      // Server detected user speech while the bot was talking - cut the
      // remaining audio right now (the Pi pipeline behaves the same way).
      stopBotAudio();
      finalizeBotRow();
      setState("listening");
    } else if (m.type === "processing") {
      // The bridge has the utterance and is transcribing: switch to
      // "thinking" now instead of sitting on "listening" while a slow ASR
      // round trip runs (the orb looked deaf). Guarded by botPlaying: a
      // backchannel over the bot must not flicker the state away from
      // "speaking" (the bot keeps talking in that case).
      if (running && !botPlaying) setState("thinking");
    } else if (m.type === "timer") {
      // A timer/reminder came due: the announcement arrives as normal
      // speech (text/done events), this toast just names what fired.
      toastShow({
        type: "tool_done", key: "timer:" + (m.id || m.label || m.kind),
        cat: "timer", title: m.kind === "reminder" ? "Reminder" : "Timer",
        sub: m.label || "done"
      });
    } else if (m.type === "toast") {
      // Server-initiated notice (rate limits, …) in the tool tray; the
      // server throttles repeats, the tray dedupes by key.
      toastShow({
        type: "tool_done", key: m.key, cat: m.cat,
        title: m.title, sub: m.sub
      });
      setTimeout(() => toastDismiss(m.key), 6000);
    } else if (m.type === "voice_enrolled") {
      toastShow({
        type: "tool_done", key: "voice", cat: "system",
        title: "Voice enrolled", sub: "only your voice starts turns now"
      });
    } else if (m.type === "voice_rejected") {
      // Someone (or something) else spoke: the utterance was dropped
      // before it could become a turn. One shared toast key, so repeats
      // refresh it instead of stacking.
      toastShow({
        type: "tool_done", key: "voice", cat: "system",
        title: "Unknown voice", sub: "ignored — not the enrolled voiceprint"
      });
    } else if (m.type === "mcp") {
      // Live tool policy: a TOML allowlist changed on disk (or the connect
      // push arrived). Re-render the tray; toast only on later updates.
      renderMcpTray(m.servers);
    } else if (m.type === "tool_start" || m.type === "tool_done") {
      // Command notification: a tool is starting (spin) or finished
      // (ok/err). The server already computed cat/title/sub/err.
      toastShow(m);
    } else if (m.type === "transcript") {
      stopBotAudio();   // barge-in: user spoke over the agent
      addUser(m.text);
      setState("thinking");
    } else if (m.type === "transcript_fix") {
      // The proofread pass corrected a mis-hearing: update the user's last
      // bubble in place (it was shown immediately, before the fix landed).
      const rows = chat.querySelectorAll(".row.user");
      const b = rows.length && rows[rows.length - 1].querySelector(".bubble");
      if (b) b.textContent = m.text;
    } else if (m.type === "text") {
      ensureBotRow();
      // Frames are sentence-sized; keep one space between sentences even
      // when the upstream split dropped the boundary whitespace.
      const prev = botBubble.textContent, t = m.t;
      if (prev && /\S$/.test(prev) && /^\S/.test(t)) botBubble.textContent += " ";
      botBubble.textContent += t;
      chatFollow();
    } else if (m.type === "done") {
      // A turn with no reply AND no tool calls = the LLM SKIPped a TV/room
      // fragment: leave no trace. But a run that CALLED a tool did real work
      // (pipecat ends tool-call runs with an empty reply mid-turn) - the
      // user's transcript and the command toast stay.
      if (!m.reply && !m.tools) {
        if (botRow) { botRow.remove(); botRow = null; botBubble = null; }
        const rows = chat.querySelectorAll(".row.user");
        if (rows.length) rows[rows.length - 1].remove();
      }
      finalizeBotRow();
      // The turn is over, but queued audio may still be playing (the server
      // pipelines TTS) - onended flips the state when playback truly ends.
      if (running && !botPlaying) setState("listening");
    } else if (m.type === "error") {
      finalizeBotRow();
      if (running && !botPlaying) setState("listening");
    } else if (m.type === "silence") {
      // Same guard as done/error: with a deferred overlap the bot may still
      // be talking (the span was noise) - don't flip the orb out of speaking.
      if (running && !botPlaying) setState("listening");
    } else if (m.type === "bye") {
      // Idle timeout: the server heard nothing and closed the session.
      // Stop the mic quietly - no banner. The orb drops back down and the
      // next tap resumes the same conversation (the server keeps it by cid).
      running = false;
      reconnectTries = 0;
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      connPhase = "offline";
      stopMic();
      stopBotAudio();
      if (ws) { ws.onclose = null; ws.close(); ws = null; }
      setState("idle");
    }
  };

  sock.onclose = () => {
    if (sock !== ws) return;  // a superseded socket closing: ignore
    ws = null;
    const wasRunning = running;
    if (wasRunning) {
      running = false;
      stopMic();
      stopBotAudio();
      setState("idle");
    }
    // The server may just be restarting: keep retrying for ~30 s instead of
    // the single silent attempt that used to leave the orb dead until the
    // page was reloaded. A retry that itself fails (reconnectTries > 0 but
    // never running) must keep the loop going too.
    if (wasRunning || reconnectTries > 0) {
      if (reconnectTries < 20) {
        reconnectTries += 1;
        connPhase = "retrying";
        stateLabel.textContent = "reconnecting";
        stateLabel.classList.remove("hide");
        reconnectTimer = setTimeout(connect, 1500);
      } else {
        // Gave up: back to rest so the next tap starts fresh.
        reconnectTries = 0;
        connPhase = "offline";
        setState("idle");
      }
    } else {
      connPhase = "offline";  // first attempt never opened; back to rest
    }
  };
  sock.onerror = () => { try { sock.close(); } catch (e) { } };
}

// Cancel whatever connection work is in flight (pending retry, a socket that
// is still connecting, or one waiting on the mic prompt) and go back to idle.
function cancelConnect() {
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  reconnectTries = 0;
  sessionId++;  // any startMic still awaiting will see it is stale and bail
  connPhase = "offline";
  if (ws) { ws.onclose = null; try { ws.close(); } catch (e) { } ws = null; }
  setState("idle");
}

// ---------------- mic permission recovery ----------------
function showMicBanner(err) {
  const name = err && err.name;
  let msg, title = "Microphone unavailable";
  if (typeof navigator.mediaDevices === "undefined") {
    // No mediaDevices at all: plain-HTTP origin or an ancient browser.
    msg = location.protocol === "http:"
      ? "The mic only works on a secure page — open this site with https:// instead."
      : "This browser doesn't expose a mic here — try a current Chrome or Safari.";
    title = "Mic not reachable here";
  } else if (name === "NotFoundError") {
    msg = "No microphone found — plug one in, then retry.";
    title = "No microphone detected";
  } else if (name === "NotReadableError") {
    msg = "Mic is in use by another app — close it, then retry.";
    title = "Microphone in use";
  } else if (name === "NotAllowedError" || name === "SecurityError") {
    msg = "Mic blocked — allow it in site settings, then retry.";
    title = "Microphone blocked";
  } else {
    msg = `Mic access failed (${name || "unknown"}) — retry.`;
  }
  document.getElementById("micBannerTitle").textContent = title;
  micBannerMsg.textContent = msg;
  micBanner.classList.add("show");
}
function hideMicBanner() { micBanner.classList.remove("show"); }

micRetryBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  hideMicBanner();
  ensureBotCtx();
  connect();
});

// ---------------- mic ----------------
// Mic frames go out as binary WebSocket frames: raw 16-bit LE mono PCM at
// 16 kHz. No base64, no JSON - a third of the bandwidth and none of the
// per-message parse cost of the old scheme.
function sendMicFrame(buf) {
  if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(buf);
}

function micFailed(e) {
  running = false;
  connPhase = "offline";
  setState("idle");
  showMicBanner(e);
  if (ws) { ws.onclose = null; ws.close(); ws = null; }
}

async function startMic(mySession) {
  // startMic awaits several times (permission prompt, context resume, worklet
  // load). The session can end in that window - orb tapped off, socket
  // closed - so everything is built in locals and only committed to the
  // globals if the session is still the live one.
  const alive = () =>
    mySession === sessionId && ws && ws.readyState === WebSocket.OPEN;
  let stream = null, ctx = null;
  const abort = () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (ctx) { try { ctx.close().catch(() => { }); } catch (e) { } }
  };

  const want = { audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false } };
  try {
    stream = await navigator.mediaDevices.getUserMedia(want);
  } catch (e) {
    // Safari can reject the constraint dict outright (channelCount et al.);
    // fall back to a bare audio request, the browser's defaults are fine.
    if (e.name !== "OverconstrainedError" && e.name !== "TypeError") {
      if (alive()) micFailed(e);
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e2) {
      if (alive()) micFailed(e2);
      return;
    }
  }
  if (!alive()) { abort(); return; }

  ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR_IN });
  if (ctx.state === "suspended") { try { await ctx.resume(); } catch (e) { } }
  const srcNode = ctx.createMediaStreamSource(stream);
  const an = ctx.createAnalyser();
  an.fftSize = 1024;
  const data = new Float32Array(an.fftSize);
  srcNode.connect(an);

  // Raw PCM via an AudioWorklet, batched to 512-sample (32 ms) binary frames.
  // The worklet also does the float->int16 conversion, so the main thread only
  // forwards an ArrayBuffer. The server's Silero VAD consumes 512-sample
  // frames anyway, so batching here costs zero end-to-end latency while
  // cutting websocket message rate 4x vs per-quantum (128-sample) sends.
  let node = null;
  let useWorklet = false;
  if (ctx.audioWorklet) {
    try {
      const workletCode = `
class PcmTap extends AudioWorkletProcessor {
  constructor() { super(); this._buf = new Int16Array(512); this._n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) {
      for (let i = 0; i < ch.length; i++) {
        const s = Math.max(-1, Math.min(1, ch[i]));
        this._buf[this._n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        if (this._n === this._buf.length) {
          const out = this._buf;
          this._buf = new Int16Array(out.length);
          this._n = 0;
          this.port.postMessage(out.buffer, [out.buffer]);
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm-tap", PcmTap);`;
      const blobUrl = URL.createObjectURL(new Blob([workletCode], { type: "application/javascript" }));
      try {
        await ctx.audioWorklet.addModule(blobUrl);
      } finally {
        URL.revokeObjectURL(blobUrl);
      }
      node = new AudioWorkletNode(ctx, "pcm-tap", { numberOfOutputs: 1, outputChannelCount: [1] });
      node.port.onmessage = (ev) => sendMicFrame(ev.data);
      srcNode.connect(node);
      useWorklet = true;
    } catch (e) {
      console.warn("[lava] AudioWorklet unavailable, using ScriptProcessor:", e);
      node = null;
    }
  }
  if (!useWorklet) {
    node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (ev) => {
      const f = ev.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f.length);
      for (let i = 0; i < f.length; i++) {
        const s = Math.max(-1, Math.min(1, f[i]));
        i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      sendMicFrame(i16.buffer);
    };
    srcNode.connect(node);
  }
  // Keep the capture node pulled (browsers GC a silent-tail processor).
  const sink = ctx.createGain();
  sink.gain.value = 0;
  node.connect(sink);
  sink.connect(ctx.destination);

  // The awaits are done. If the session ended meanwhile, release everything
  // we built instead of committing it (and never touch a null socket).
  if (!alive()) { abort(); return; }

  mediaStream = stream;
  audioCtx = ctx;
  analyser = an;
  micData = data;
  processor = node;

  running = true;
  // One-way latch: once the orb has risen for a call, it stays up even
  // after the call ends (body.idle alone no longer re-centers it).
  document.body.classList.add("risen");
  hideMicBanner();
  setState("listening");
  // Tell the server the mic is live, twice over:
  //   POST /api/greet  fires the warm greeting instantly - the greeting
  //                    answers the "Allow" click instead of preempting it,
  //                    and the audio streams from the server's cache with
  //                    no pipeline boot or LLM round trip on the path.
  //   {"type":"ready"} gates the pipeline's VAD/turns (talking over the
  //                    greeting barges it) and the fallback greeting when
  //                    the warm cache is empty.
  fetch("/api/greet?cid=" + encodeURIComponent(client_id), { method: "POST" }).catch(() => { });
  ws.send(JSON.stringify({ type: "ready" }));
}

function stopMic() {
  if (processor) {
    try {
      if (processor.port) processor.port.onmessage = null;
      processor.onaudioprocess = null;
      processor.disconnect();
    } catch (e) { }
    processor = null;
  }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (audioCtx) {
    const c = audioCtx;
    audioCtx = null;
    try { c.close().catch(() => { }); } catch (e) { }  // close() returns a promise
  }
  analyser = null; micData = null;
}

// ---------------- toggle ----------------
orbWrap.addEventListener("click", () => {
  if (!running) {
    // A retry timer, a still-connecting socket, or a socket waiting on the
    // mic prompt means connection work is already in flight: a tap cancels
    // it (back to idle) instead of opening a second socket on top of it.
    if (reconnectTimer || ws) {
      cancelConnect();
      return;
    }
    ensureBotCtx();
    connect();
  } else {
    running = false;
    sessionId++;
    reconnectTries = 0;
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    connPhase = "offline";
    stopMic();
    stopBotAudio();
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
    setState("idle");
  }
});

// Keyboard activation of the orb (it is role="button", tabindex="0"):
// Enter or Space while focused. The document-level Space shortcut below
// stays as a convenience when nothing is focused; the two never double-fire
// (that one requires e.target === document.body).
orbWrap.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    orbWrap.click();
  }
});

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target === document.body) {
    e.preventDefault();
    orbWrap.click();
  }
});

// Safari: resume() only works inside a user gesture, and the mic-permission
// dialog eats the orb click's gesture. So ANY tap or keypress re-arms audio.
document.addEventListener("pointerdown", unlockAudio, true);
document.addEventListener("keydown", unlockAudio, true);

// ---------------- mcp tray ----------------
// Collapsible roster of the MCP servers: the toggle unfolds the panel, and
// any click outside (or Escape) folds it back. Inside the panel, each server
// row is itself a collapsible that reveals the tools it exposes (accordion:
// one unfolded at a time). Tapping a tool chip sends it to Lava as a regular
// user turn, so the reply lands in the chat and is spoken like voice.
const mcpTray = document.getElementById("mcpTray");
const mcpToggle = document.getElementById("mcpToggle");
const mcpTrayPanel = document.getElementById("mcpTrayPanel");

function setMcpTrayOpen(open) {
  mcpTray.classList.toggle("open", open);
  mcpToggle.setAttribute("aria-expanded", open ? "true" : "false");
  if (!open) {
    // keyboard users folding the tray land back on the toggle, not <body>
    if (mcpTrayPanel.contains(document.activeElement)) mcpToggle.focus();
    collapseMcpGroups(); // folding the tray folds the tools too
  }
}

function collapseMcpGroups() {
  mcpTrayPanel.querySelectorAll(".mcp-group.open").forEach((g) => g.classList.remove("open"));
  mcpTrayPanel.querySelectorAll(".mcp-item").forEach((item) => item.setAttribute("aria-expanded", "false"));
}

mcpToggle.addEventListener("click", (e) => {
  e.stopPropagation();
  setMcpTrayOpen(!mcpTray.classList.contains("open"));
});

// ---- live tool policy (agent/tools/*.toml) --------------------------------
// The tray is rendered from the server's policy snapshot, not static HTML:
// flipping `enabled` in a TOML re-renders it live (server watch → "mcp"
// message over the websocket; /api/allowlist is the initial fallback).
// All tray clicks are DELEGATED, so re-renders never lose their handlers.

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

const STROKE = 'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';
const FILL = 'fill="currentColor" stroke="none" aria-hidden="true"';

const MCP_ICONS = {
  spotify: `<svg viewBox="0 0 24 24" ${FILL}><path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/></svg>`,
  github: `<svg viewBox="0 0 24 24" ${FILL}><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>`,
  tavily: `<svg viewBox="0 0 24 24" ${STROKE}><circle cx="11" cy="11" r="7"/><path d="m21 21-4.35-4.35"/></svg>`,
  time: `<svg viewBox="0 0 24 24" ${STROKE}><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3.2 1.9"/></svg>`,
  asr: `<svg viewBox="0 0 24 24" ${STROKE}><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0"/><path d="M12 17.5V21"/><path d="M9 21h6"/></svg>`,
  tts: `<svg viewBox="0 0 24 24" ${STROKE}><path d="M11 5 6 9H3v6h3l5 4V5Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 5.5a9 9 0 0 1 0 13"/></svg>`,
  generic: `<svg viewBox="0 0 24 24" ${STROKE}><rect x="7" y="7" width="10" height="10" rx="2"/><path d="M12 2v5M12 17v5M2 12h5M17 12h5"/></svg>`,
};

function mcpIcon(name) {
  return MCP_ICONS[name] || MCP_ICONS.generic;
}

function renderMcpTray(servers) {
  const host = document.getElementById("mcpGroups");
  if (!host || !Array.isArray(servers)) return;
  if (!servers.length) {  // nothing connected: say so, not an empty box
    host.innerHTML = `<p class="mcp-empty">no servers connected</p>`;
    return;
  }
  const groups = servers.map((s) => {
    // callable tools first (buttons), the rest dimmed under a hairline
    const live = s.tools.filter((t) => t.enabled && s.role !== "bridge");
    const off = s.tools.filter((t) => !t.enabled || s.role === "bridge");
    const chip = (t) => {
      const tip = (t.confirm ? "requires spoken confirmation — " : "") + t.description;
      const ask = "Please use " + t.name + ".";
      const gate = t.confirm ? `<span class="gate">★</span>` : "";
      return `<button class="mcp-tool on" type="button" title="${esc(tip)}" data-ask="${esc(ask)}">${esc(t.name)}${gate}</button>`;
    };
    const dead = (t) =>
      `<span class="mcp-tool off" title="${esc(s.role === "bridge"
        ? "used by the voice loop directly — not exposed to the LLM"
        : "offered by the server, not in Lava's allowlist")}">${esc(t.name)}</span>`;
    const rows = live.map(chip).join("")
      + (live.length && off.length ? `<hr class="off-rule">` : "")
      + off.map(dead).join("");
    const role = s.role === "bridge" ? "bridge" : s.kind;
    // Commas separate the facts (a "·" gets read aloud as "dot"), and the
    // non-breaking spaces keep each count glued to its unit, so a meta too
    // long for the row (github/spotify) wraps at a comma, not mid-phrase.
    const meta = `${role}, ${s.tools_total}\u00A0${s.tools_total === 1 ? "tool" : "tools"}, ${s.enabled.length}\u00A0on`
      + (s.confirm.length ? `, ${s.confirm.length}\u00A0gated` : "");
    // live health (mcp_client.HEALTH via the snapshot): green since last
    // contact, red unreachable, dim unknown; the dot is decorative, the
    // row title carries the words for screen readers
    const health = s.connected === true ? "up" : s.connected === false ? "down" : "unknown";
    const healthTxt = s.connected === false ? " — not connected" : "";
    return `
    <div class="mcp-group">
    <button class="mcp-item" type="button" title="${esc(s.name)} (${role})${esc(healthTxt)}" aria-expanded="false">
      ${mcpIcon(s.name)}
      <span class="mcp-body">
        <span class="mcp-name"><span class="mcp-health ${health}" aria-hidden="true"></span>${esc(s.name)}</span>
        <span class="mcp-meta">${meta}</span>
      </span>
      <svg class="mcp-item-chev" viewBox="0 0 24 24" ${STROKE}><path d="m9 6 6 6-6 6"/></svg>
    </button>
    <div class="mcp-tools">${rows}</div>
    </div>`;
  });
  host.innerHTML = groups.join("");
}

fetch("/api/allowlist")
  .then((r) => r.json())
  .then((d) => renderMcpTray(d.servers))
  .catch(() => { });  // offline until the websocket pushes the snapshot

// Delegated tray clicks: one accordion per group (one unfolded at a time),
// tool chips send their prompt as a normal user turn through the pipeline.
mcpTrayPanel.addEventListener("click", (e) => {
  const chip = e.target.closest(".mcp-tool");
  if (chip && chip.tagName === "BUTTON") {
    askLava(chip.dataset.ask || "");
    return;
  }
  const item = e.target.closest(".mcp-item");
  if (item) {
    const group = item.closest(".mcp-group");
    const wasOpen = group.classList.contains("open");
    collapseMcpGroups();
    if (!wasOpen) {
      group.classList.add("open");
      item.setAttribute("aria-expanded", "true");
    }
  }
});

function askLava(text) {
  if (!text) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    toastShow({
      key: "tray-offline", cat: "tray", title: "Lava is offline",
      sub: "Tap the orb to connect first"
    });
    setTimeout(() => toastDismiss("tray-offline"), 4000);
    return;
  }
  addUser(text);          // the chip's prompt shows as this turn's user bubble
  setState("thinking");   // no ASR runs, so raise the orb state ourselves
  setMcpTrayOpen(false);  // fold the tray so the answer lands in view
  ws.send(JSON.stringify({ type: "ask", text }));
}

document.addEventListener("click", (e) => {
  if (mcpTray.classList.contains("open") && !mcpTray.contains(e.target)) {
    setMcpTrayOpen(false);
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && mcpTray.classList.contains("open")) {
    setMcpTrayOpen(false);
  }
});
