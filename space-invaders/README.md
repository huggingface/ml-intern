# Space Invaders — Nouveau

A mobile-responsive Progressive Web App recreation of the 1978 arcade classic,
built with vanilla HTML5 Canvas, CSS, and JavaScript (ES modules) — no build
step, no dependencies.

Faithful to the original's mechanics: the hive-mind alien swarm with its
hardware-accurate speed-up curve, deterministic Rolling/Plunger/Squiggly
firing patterns, the hidden shot-count-based mystery ship scoring table,
pixel-level destructible bunkers, and the one-bullet-at-a-time cannon — all
reskinned in a gilded, jewel-toned Art Nouveau visual style with procedurally
synthesized (not sampled) sound effects.

## Running locally

Any static file server works, e.g.:

```bash
cd space-invaders
python3 -m http.server 8080
```

Then open `http://localhost:8080`. Installable as a PWA (offline-capable via
service worker) from a supporting browser.

## Controls

- **Move**: Arrow keys / A-D, or the on-screen D-pad on touch devices.
- **Fire**: Space / Up / W, or the on-screen fire button.
