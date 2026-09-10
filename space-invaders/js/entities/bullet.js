import { PLAYER_BULLET_SPEED, ALIEN_BULLET_SPEED, CANVAS_HEIGHT, PALETTE } from '../constants.js';

export class PlayerBullet {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.width = 1;
    this.height = 4;
    this.isAlive = true;
  }

  update(dt) {
    this.y -= PLAYER_BULLET_SPEED * dt;
    if (this.y + this.height < 0) this.isAlive = false;
  }

  render(ctx) {
    ctx.fillStyle = PALETTE.bulletPlayer;
    ctx.shadowColor = PALETTE.bulletPlayer;
    ctx.shadowBlur = 3;
    ctx.fillRect(this.x, this.y, this.width, this.height);
    ctx.shadowBlur = 0;
  }
}

const ALIEN_BULLET_WAVE = { amplitude: 1.5, frequency: 10 };

export class AlienBullet {
  constructor(x, y, style = 'rolling') {
    this.x = x;
    this.y = y;
    this.width = 3;
    this.height = 7;
    this.isAlive = true;
    this.style = style; // 'rolling' | 'plunger' | 'squiggly'
    this.age = 0;
  }

  update(dt) {
    this.age += dt;
    this.y += ALIEN_BULLET_SPEED * dt;
    if (this.style === 'squiggly') {
      this.x += Math.sin(this.age * ALIEN_BULLET_WAVE.frequency) * ALIEN_BULLET_WAVE.amplitude * dt * 10;
    }
    if (this.y > CANVAS_HEIGHT) this.isAlive = false;
  }

  render(ctx) {
    ctx.fillStyle = PALETTE.bulletAlien;
    ctx.shadowColor = PALETTE.bulletAlien;
    ctx.shadowBlur = 3;
    const wobble = this.style === 'squiggly' ? Math.sin(this.age * 14) * 1.2 : 0;
    ctx.fillRect(this.x + wobble, this.y, this.width, this.height);
    ctx.shadowBlur = 0;
  }
}
