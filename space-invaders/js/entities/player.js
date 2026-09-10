import {
  PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_SPEED, CANVAS_WIDTH, CANVAS_HEIGHT, PALETTE,
} from '../constants.js';
import { PlayerBullet } from './bullet.js';

export class Player {
  constructor(game) {
    this.game = game;
    this.width = PLAYER_WIDTH;
    this.height = PLAYER_HEIGHT;
    this.x = (CANVAS_WIDTH - this.width) / 2;
    this.y = CANVAS_HEIGHT - 24;
    this.canFire = true;
    this.isAlive = true;
    this.hitFlashTimer = 0;
  }

  update(dt, input) {
    if (!this.isAlive) return;
    let dx = 0;
    if (input.left) dx -= 1;
    if (input.right) dx += 1;
    this.x += dx * PLAYER_SPEED * dt;
    this.x = Math.max(2, Math.min(CANVAS_WIDTH - this.width - 2, this.x));

    if (input.fire) this.fire();
    if (this.hitFlashTimer > 0) this.hitFlashTimer -= dt;
  }

  fire() {
    if (!this.canFire || !this.isAlive) return;
    const bullet = new PlayerBullet(this.x + this.width / 2 - 0.5, this.y - 4);
    this.game.playerBullets.push(bullet);
    this.canFire = false;
    this.game.playerShotCount += 1;
    this.game.audio.playSound('shoot');
  }

  onBulletResolved() {
    this.canFire = true;
  }

  render(ctx) {
    if (!this.isAlive) return;
    const cx = this.x + this.width / 2;
    const cy = this.y + this.height;
    ctx.save();
    ctx.translate(cx, cy);

    // Art Nouveau cannon: a stylised winged/finned dart with jewel core and
    // whiplash tendrils sweeping back from the hull, rendered as vector paths.
    const glow = this.hitFlashTimer > 0 ? PALETTE.goldBright : PALETTE.player;
    ctx.strokeStyle = PALETTE.gold;
    ctx.lineWidth = 0.6;
    ctx.fillStyle = glow;
    ctx.shadowColor = glow;
    ctx.shadowBlur = 4;

    ctx.beginPath();
    ctx.moveTo(0, -this.height);
    ctx.bezierCurveTo(-2, -this.height + 2, -3, -2, -this.width / 2, 0);
    ctx.lineTo(-this.width / 2 + 1, -1);
    ctx.bezierCurveTo(-3, -2.5, -1.5, -3.5, 0, -this.height + 1);
    ctx.bezierCurveTo(1.5, -3.5, 3, -2.5, this.width / 2 - 1, -1);
    ctx.lineTo(this.width / 2, 0);
    ctx.bezierCurveTo(3, -2, 2, -this.height + 2, 0, -this.height);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Tendril flourishes curling from either side of the hull.
    ctx.strokeStyle = PALETTE.goldDim;
    ctx.lineWidth = 0.5;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.moveTo(-this.width / 2, -1);
    ctx.quadraticCurveTo(-this.width / 2 - 3, -3, -this.width / 2 - 1, -6);
    ctx.moveTo(this.width / 2, -1);
    ctx.quadraticCurveTo(this.width / 2 + 3, -3, this.width / 2 + 1, -6);
    ctx.stroke();

    // Central jewel core.
    ctx.fillStyle = PALETTE.goldBright;
    ctx.beginPath();
    ctx.ellipse(0, -this.height + 3, 1.4, 1.8, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }
}
