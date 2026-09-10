import { drawExplosion } from '../render.js';

export class Explosion {
  constructor(x, y, color, duration = 0.35) {
    this.x = x;
    this.y = y;
    this.color = color;
    this.age = 0;
    this.duration = duration;
    this.isAlive = true;
  }

  update(dt) {
    this.age += dt;
    if (this.age >= this.duration) this.isAlive = false;
  }

  render(ctx) {
    drawExplosion(ctx, this);
  }
}
