import { FIRE_PATTERNS, ALIEN_FIRE_MIN_INTERVAL, ALIEN_FIRE_MAX_INTERVAL, ALIEN_COLS } from '../constants.js';
import { AlienBullet } from './bullet.js';

const PATTERN_NAMES = ['ROLLING', 'PLUNGER', 'SQUIGGLY'];

// Models the original's deterministic, column-based alien firing rather than
// pure randomness: a shot "personality" is picked, then a sequence of target
// columns is walked, always hitting the lowest living alien in that column.
export class AlienFireController {
  constructor(swarm) {
    this.swarm = swarm;
    this.timer = 0;
    this.nextInterval = this._randomInterval();
    this.sequenceIndex = { ROLLING: 0, PLUNGER: 0, SQUIGGLY: 0 };
  }

  _randomInterval() {
    return ALIEN_FIRE_MIN_INTERVAL + Math.random() * (ALIEN_FIRE_MAX_INTERVAL - ALIEN_FIRE_MIN_INTERVAL);
  }

  reset() {
    this.timer = 0;
    this.nextInterval = this._randomInterval();
  }

  update(dt, alienBullets) {
    this.timer += dt * 1000;
    if (this.timer < this.nextInterval) return;
    this.timer = 0;
    this.nextInterval = this._randomInterval();

    const patternName = PATTERN_NAMES[Math.floor(Math.random() * PATTERN_NAMES.length)];
    const sequence = FIRE_PATTERNS[patternName];
    const idx = this.sequenceIndex[patternName];
    const col = sequence[idx % sequence.length] % ALIEN_COLS;
    this.sequenceIndex[patternName] = idx + 1;

    const shooter = this.swarm.lowestAlienInColumn(col);
    if (!shooter) return;

    const pos = this.swarm.alienScreenPosition(shooter);
    const bullet = new AlienBullet(
      pos.x + shooter.width / 2 - 1.5,
      pos.y + shooter.height,
      patternName.toLowerCase(),
    );
    alienBullets.push(bullet);
  }
}
