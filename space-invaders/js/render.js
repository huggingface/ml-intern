import { CANVAS_WIDTH, CANVAS_HEIGHT, PALETTE } from './constants.js';

// --- Alien sprites -----------------------------------------------------
// Each alien type is a small stained-glass/whiplash-curve creature drawn as
// a mirrored vector path (draw the right half, reflect for the left) so the
// silhouette stays perfectly symmetric like Art Nouveau glasswork.

function halfPath(type, frame) {
  switch (type) {
    case 'squid':
      return (c) => {
        c.moveTo(0, 0);
        c.quadraticCurveTo(2, -1, 3, -3);
        c.quadraticCurveTo(4, -5, 2, -6);
        c.quadraticCurveTo(1, -6.5, 0, -6);
        c.lineTo(0, -1);
        c.quadraticCurveTo(3, 0, 4, 2);
        if (frame === 0) {
          c.quadraticCurveTo(4.5, 4, 3, 4);
          c.quadraticCurveTo(2.5, 3, 2, 2);
        } else {
          c.quadraticCurveTo(5, 3, 4, 4.5);
          c.quadraticCurveTo(3, 3, 2, 2);
        }
        c.lineTo(0, 1);
      };
    case 'crab':
      return (c) => {
        c.moveTo(0, -6);
        c.quadraticCurveTo(3, -6.5, 4, -4);
        c.quadraticCurveTo(4.5, -2, 3, -1);
        c.lineTo(5.5, -1);
        c.quadraticCurveTo(6, 0, 5, 1);
        c.lineTo(3, 1);
        if (frame === 0) {
          c.lineTo(4, 4);
          c.quadraticCurveTo(3, 4.5, 2, 3);
        } else {
          c.lineTo(2, 4.5);
          c.quadraticCurveTo(1.5, 4, 1, 3);
        }
        c.lineTo(1, 1);
        c.lineTo(0, 1);
      };
    default: // octopus
      return (c) => {
        c.moveTo(0, -6);
        c.quadraticCurveTo(4, -6.5, 5, -3);
        c.quadraticCurveTo(5.5, -1, 4, 0);
        c.lineTo(5.5, 0.5);
        if (frame === 0) {
          c.lineTo(5, 4);
          c.lineTo(3.5, 2);
        } else {
          c.lineTo(3.5, 4.5);
          c.lineTo(2.5, 2);
        }
        c.lineTo(2, 3);
        c.lineTo(0, 1);
      };
  }
}

const ALIEN_COLORS = {
  squid: [PALETTE.squid, PALETTE.squidDark],
  crab: [PALETTE.crab, PALETTE.crabDark],
  octopus: [PALETTE.octopus, PALETTE.octopusDark],
};

export function drawAlien(ctx, alien, x, y, animFrame) {
  const [fill, dark] = ALIEN_COLORS[alien.type];
  const scale = alien.type === 'squid' ? 1 : alien.type === 'crab' ? 1.3 : 1.4;
  ctx.save();
  ctx.translate(x + alien.width / 2, y + alien.height / 2 + 1);
  ctx.scale(scale, scale);
  ctx.fillStyle = fill;
  ctx.strokeStyle = dark;
  ctx.lineWidth = 0.4;
  ctx.shadowColor = fill;
  ctx.shadowBlur = 2.5;

  const build = halfPath(alien.type, animFrame);
  ctx.beginPath();
  build(ctx);
  ctx.scale(-1, 1);
  build(ctx);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Jewel eyes.
  ctx.shadowBlur = 0;
  ctx.fillStyle = PALETTE.goldBright;
  ctx.beginPath();
  ctx.ellipse(-1, -2.5, 0.6, 0.6, 0, 0, Math.PI * 2);
  ctx.ellipse(1, -2.5, 0.6, 0.6, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// --- Explosion particles ------------------------------------------------
// A blooming-flower burst: petals of light radiating and curling outward.
export function drawExplosion(ctx, explosion) {
  const { x, y, age, duration, color } = explosion;
  const progress = age / duration;
  const petals = 8;
  const radius = 2 + progress * 7;
  const alpha = 1 - progress;

  ctx.save();
  ctx.translate(x, y);
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 4;

  for (let i = 0; i < petals; i++) {
    const angle = (i / petals) * Math.PI * 2 + progress * 1.2;
    const px = Math.cos(angle) * radius;
    const py = Math.sin(angle) * radius;
    ctx.beginPath();
    ctx.ellipse(px, py, 1.6 * (1 - progress * 0.4), 0.8, angle, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.beginPath();
  ctx.arc(0, 0, 1.4 * (1 - progress), 0, Math.PI * 2);
  ctx.fillStyle = PALETTE.goldBright;
  ctx.fill();
  ctx.restore();
}

// --- Backdrop -------------------------------------------------------------
// Deep nebula gradient with slow-drifting whiplash curves (the defining
// Art Nouveau line — an elongated "S" swept across the field).
export function drawBackground(ctx, time) {
  const grad = ctx.createLinearGradient(0, 0, 0, CANVAS_HEIGHT);
  grad.addColorStop(0, PALETTE.bgTop);
  grad.addColorStop(1, PALETTE.bgBottom);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  ctx.save();
  ctx.globalAlpha = 0.12;
  ctx.strokeStyle = PALETTE.gold;
  ctx.lineWidth = 1.2;
  const drift = (time * 0.005) % (Math.PI * 2);
  for (let i = 0; i < 3; i++) {
    const yBase = 40 + i * 80 + Math.sin(drift + i) * 10;
    ctx.beginPath();
    ctx.moveTo(-20, yBase);
    ctx.bezierCurveTo(
      CANVAS_WIDTH * 0.25, yBase - 30 + Math.sin(drift + i) * 12,
      CANVAS_WIDTH * 0.75, yBase + 30 - Math.sin(drift + i) * 12,
      CANVAS_WIDTH + 20, yBase,
    );
    ctx.stroke();
  }
  ctx.restore();

  // Faint static starfield (deterministic pseudo-random, no per-frame alloc).
  ctx.save();
  ctx.fillStyle = PALETTE.gold;
  for (let i = 0; i < 40; i++) {
    const sx = (i * 53.7) % CANVAS_WIDTH;
    const sy = (i * 97.3) % CANVAS_HEIGHT;
    const tw = 0.3 + 0.3 * Math.sin(time * 0.002 + i);
    ctx.globalAlpha = Math.max(0.05, tw);
    ctx.fillRect(sx, sy, 1, 1);
  }
  ctx.restore();
}

// --- Ornamental frame -----------------------------------------------------
// A gilded proscenium border wrapping the playfield, evoking a Mucha poster
// frame: corner medallions joined by flowing vine linework.
export function drawFrame(ctx) {
  ctx.save();
  ctx.strokeStyle = PALETTE.gold;
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, CANVAS_WIDTH - 2, CANVAS_HEIGHT - 2);

  ctx.lineWidth = 0.7;
  ctx.globalAlpha = 0.8;
  const corners = [
    [0, 0, 1, 1],
    [CANVAS_WIDTH, 0, -1, 1],
    [0, CANVAS_HEIGHT, 1, -1],
    [CANVAS_WIDTH, CANVAS_HEIGHT, -1, -1],
  ];
  for (const [cx, cy, sx, sy] of corners) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(sx, sy);
    ctx.beginPath();
    ctx.moveTo(2, 10);
    ctx.quadraticCurveTo(2, 2, 10, 2);
    ctx.moveTo(4, 14);
    ctx.quadraticCurveTo(14, 14, 14, 4);
    ctx.stroke();
    ctx.restore();
  }
  ctx.restore();
}
