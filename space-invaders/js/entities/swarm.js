import {
  ALIEN_ROWS, ALIEN_COLS, ALIEN_H_SPACING, ALIEN_V_SPACING, ALIEN_START_X, ALIEN_START_Y,
  ALIEN_DROP_DISTANCE, CANVAS_WIDTH, BASE_MOVE_INTERVAL, MIN_MOVE_INTERVAL,
} from '../constants.js';
import { Alien } from './alien.js';
import { AlienFireController } from './alienFireController.js';
import { drawAlien } from '../render.js';

const EDGE_MARGIN = 10;
const STEP_SIZE = 3;

// The "Hive Mind": a single controller owning the swarm's collective position,
// direction and pace. Individual Aliens never store world coordinates — their
// screen position is always swarm-origin + fixed grid offset.
export class Swarm {
  constructor(game) {
    this.game = game;
    this.x = 0;
    this.y = 0;
    this.direction = 1;
    this.timer = 0;
    this.animFrame = 0;
    this.marchStep = 0;
    this.aliens = [];
    this.fireController = new AlienFireController(this);
    this._buildGrid(ALIEN_START_Y);
  }

  _buildGrid(startY) {
    this.aliens = [];
    for (let row = 0; row < ALIEN_ROWS; row++) {
      for (let col = 0; col < ALIEN_COLS; col++) {
        const offsetX = ALIEN_START_X + col * ALIEN_H_SPACING;
        const offsetY = startY + row * ALIEN_V_SPACING;
        this.aliens.push(new Alien(row, col, offsetX, offsetY));
      }
    }
  }

  resetForWave(startY) {
    this.x = 0;
    this.y = 0;
    this.direction = 1;
    this.timer = 0;
    this.marchStep = 0;
    this._buildGrid(startY);
    this.fireController.reset();
  }

  get aliveCount() {
    return this.aliens.reduce((n, a) => n + (a.isAlive ? 1 : 0), 0);
  }

  get isDefeated() {
    return this.aliveCount === 0;
  }

  alienScreenPosition(alien) {
    return { x: this.x + alien.offsetX, y: this.y + alien.offsetY };
  }

  lowestAlienInColumn(col) {
    let best = null;
    for (const a of this.aliens) {
      if (!a.isAlive || a.col !== col) continue;
      if (!best || a.row > best.row) best = a;
    }
    return best;
  }

  bottomMostY() {
    let maxY = -Infinity;
    for (const a of this.aliens) {
      if (!a.isAlive) continue;
      const y = this.y + a.offsetY + a.height;
      if (y > maxY) maxY = y;
    }
    return maxY === -Infinity ? 0 : maxY;
  }

  update(dt, alienBullets) {
    const total = ALIEN_ROWS * ALIEN_COLS;
    const alive = this.aliveCount;
    if (alive === 0) return;

    const moveInterval = Math.max(MIN_MOVE_INTERVAL, (alive / total) * BASE_MOVE_INTERVAL);
    this.timer += dt * 1000;

    if (this.timer >= moveInterval) {
      this.timer = 0;
      this._step();
      this.game.audio.playMarchStep(this.marchStep);
      this.marchStep += 1;
      this.animFrame = 1 - this.animFrame;
    }

    this.fireController.update(dt, alienBullets);
  }

  _step() {
    this.x += this.direction * STEP_SIZE;

    let leftmost = Infinity;
    let rightmost = -Infinity;
    for (const a of this.aliens) {
      if (!a.isAlive) continue;
      const left = this.x + a.offsetX;
      const right = left + a.width;
      if (left < leftmost) leftmost = left;
      if (right > rightmost) rightmost = right;
    }
    if (leftmost === Infinity) return;

    if (rightmost >= CANVAS_WIDTH - EDGE_MARGIN || leftmost <= EDGE_MARGIN) {
      this.y += ALIEN_DROP_DISTANCE;
      this.direction *= -1;
    }
  }

  render(ctx) {
    for (const alien of this.aliens) {
      if (!alien.isAlive) continue;
      const pos = this.alienScreenPosition(alien);
      drawAlien(ctx, alien, pos.x, pos.y, this.animFrame);
    }
  }
}
