import {
  MYSTERY_SHIP_WIDTH, MYSTERY_SHIP_HEIGHT, MYSTERY_SHIP_SPEED, CANVAS_WIDTH,
  MYSTERY_SHIP_MIN_DELAY, MYSTERY_SHIP_MAX_DELAY, UFO_SCORE_TABLE, PALETTE,
} from '../constants.js';

export class MysteryShip {
  constructor(game) {
    this.game = game;
    this.width = MYSTERY_SHIP_WIDTH;
    this.height = MYSTERY_SHIP_HEIGHT;
    this.x = 0;
    this.y = 12;
    this.direction = 1;
    this.active = false;
    this.spawnTimer = this._randomDelay();
  }

  _randomDelay() {
    return MYSTERY_SHIP_MIN_DELAY + Math.random() * (MYSTERY_SHIP_MAX_DELAY - MYSTERY_SHIP_MIN_DELAY);
  }

  _spawn() {
    this.direction = Math.random() < 0.5 ? 1 : -1;
    this.x = this.direction === 1 ? -this.width : CANVAS_WIDTH + this.width;
    this.active = true;
    this.game.audio.startUfoLoop();
  }

  despawn() {
    this.active = false;
    this.game.audio.stopUfoLoop();
    this.spawnTimer = this._randomDelay();
  }

  // Score awarded follows the original's hidden deterministic sequence based
  // on total shots fired, not randomness.
  scoreValue() {
    return UFO_SCORE_TABLE[this.game.playerShotCount % UFO_SCORE_TABLE.length];
  }

  update(dt) {
    if (!this.active) {
      this.spawnTimer -= dt * 1000;
      if (this.spawnTimer <= 0) this._spawn();
      return;
    }
    this.x += this.direction * MYSTERY_SHIP_SPEED * dt;
    if (this.direction === 1 && this.x > CANVAS_WIDTH) this.despawn();
    if (this.direction === -1 && this.x + this.width < 0) this.despawn();
  }

  render(ctx) {
    if (!this.active) return;
    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.shadowColor = PALETTE.ufo;
    ctx.shadowBlur = 5;
    ctx.fillStyle = PALETTE.ufo;
    ctx.strokeStyle = PALETTE.gold;
    ctx.lineWidth = 0.6;

    ctx.beginPath();
    ctx.moveTo(1, this.height);
    ctx.bezierCurveTo(-2, this.height, -2, 2, this.width / 2, 1);
    ctx.bezierCurveTo(this.width + 2, 2, this.width + 2, this.height, this.width - 1, this.height);
    ctx.bezierCurveTo(this.width - 3, this.height + 2, 3, this.height + 2, 1, this.height);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Gem-like canopy jewel, the classic Nouveau "peacock eye" motif.
    ctx.fillStyle = PALETTE.goldBright;
    ctx.beginPath();
    ctx.ellipse(this.width / 2, this.height - 1, 2.2, 1.6, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}
