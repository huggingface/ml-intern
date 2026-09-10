import {
  CANVAS_WIDTH, CANVAS_HEIGHT, ALIEN_START_Y, ALIEN_DROP_DISTANCE, ALIEN_MAX_DEPTH_Y,
  BUNKER_COUNT, BUNKER_WIDTH, BUNKER_Y, EXTRA_LIFE_SCORE, PLAYER_START_LIVES,
} from './constants.js';
import { InputHandler } from './input.js';
import { AudioManager } from './audio.js';
import { Player } from './entities/player.js';
import { Swarm } from './entities/swarm.js';
import { MysteryShip } from './entities/mysteryShip.js';
import { Bunker } from './entities/bunker.js';
import { Explosion } from './entities/explosion.js';
import { StateMachine } from './state/stateMachine.js';
import { createAttractState } from './state/attractState.js';
import { createPlayState } from './state/playState.js';
import { createGameOverState } from './state/gameOverState.js';

const HIGH_SCORE_KEY = 'space-invaders-nouveau-high-score';

class Game {
  constructor(ctx) {
    this.ctx = ctx;
    this.input = new InputHandler();
    this.audio = new AudioManager();

    this.score = 0;
    this.highScore = Number(localStorage.getItem(HIGH_SCORE_KEY)) || 0;
    this.lives = PLAYER_START_LIVES;
    this.wave = 1;
    this.playerShotCount = 0;
    this.extraLifeAwarded = false;

    this.player = new Player(this);
    this.swarm = new Swarm(this);
    this.mysteryShip = new MysteryShip(this);
    this.bunkers = this._createBunkers();
    this.playerBullets = [];
    this.alienBullets = [];
    this.explosions = [];

    this.hud = {
      score: document.getElementById('score-value'),
      lives: document.getElementById('lives-value'),
      highScore: document.getElementById('highscore-value'),
    };

    this._updateHud();

    const unlockOnce = () => { this.audio.unlock(); window.removeEventListener('keydown', unlockOnce); window.removeEventListener('touchstart', unlockOnce); };
    window.addEventListener('keydown', unlockOnce);
    window.addEventListener('touchstart', unlockOnce);
  }

  _createBunkers() {
    const margin = (CANVAS_WIDTH - BUNKER_COUNT * BUNKER_WIDTH) / (BUNKER_COUNT + 1);
    const bunkers = [];
    for (let i = 0; i < BUNKER_COUNT; i++) {
      const x = margin + i * (BUNKER_WIDTH + margin);
      bunkers.push(new Bunker(Math.round(x), BUNKER_Y));
    }
    return bunkers;
  }

  _updateHud() {
    this.hud.score.textContent = String(this.score).padStart(4, '0');
    this.hud.lives.textContent = String(Math.max(0, this.lives));
    this.hud.highScore.textContent = String(this.highScore).padStart(4, '0');
  }

  onScoreChanged() {
    if (this.score > this.highScore) {
      this.highScore = this.score;
      localStorage.setItem(HIGH_SCORE_KEY, String(this.highScore));
    }
    if (!this.extraLifeAwarded && this.score >= EXTRA_LIFE_SCORE) {
      this.extraLifeAwarded = true;
      this.lives += 1;
      this.audio.playSound('extraLife');
    }
    this._updateHud();
  }

  onPlayerHit() {
    if (!this.player.isAlive) return;
    this.player.isAlive = false;
    this.lives -= 1;
    this.player.hitFlashTimer = 0.3;
    this.audio.playSound('playerExplosion');
    this._updateHud();
    this.explosions.push(new Explosion(
      this.player.x + this.player.width / 2,
      this.player.y + this.player.height / 2,
      '#4fb8c9',
      0.6,
    ));
  }

  startNewGame() {
    this.score = 0;
    this.lives = PLAYER_START_LIVES;
    this.wave = 1;
    this.playerShotCount = 0;
    this.extraLifeAwarded = false;
    this.playerBullets = [];
    this.alienBullets = [];
    this.explosions = [];
    this.player = new Player(this);
    this.mysteryShip = new MysteryShip(this);
    this.swarm.resetForWave(ALIEN_START_Y);
    for (const b of this.bunkers) b.repair();
    this._updateHud();
  }

  nextWave() {
    this.wave += 1;
    const depth = Math.min(ALIEN_START_Y + (this.wave - 1) * ALIEN_DROP_DISTANCE, ALIEN_MAX_DEPTH_Y);
    this.swarm.resetForWave(depth);
    this.playerBullets = [];
    this.alienBullets = [];
    this.player.isAlive = true;
    this.player.canFire = true;
    this.player.x = (CANVAS_WIDTH - this.player.width) / 2;
    for (const b of this.bunkers) b.repair();
  }
}

function init() {
  const canvas = document.getElementById('game-canvas');
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;

  const game = new Game(ctx);
  window.__game = game; // debugging hook only

  const machine = new StateMachine({});
  // Build states after machine exists so they can call machine.transition().
  machine.states.attract = createAttractState(game, machine);
  machine.states.play = createPlayState(game, machine);
  machine.states.gameOver = createGameOverState(game, machine);
  machine.start('attract');

  game.input.bindTouchControls(document.getElementById('touch-controls'));

  let lastTime = performance.now();
  function loop(now) {
    let deltaTime = (now - lastTime) / 1000;
    lastTime = now;
    deltaTime = Math.min(deltaTime, 1 / 20); // clamp to avoid spiral-of-death on tab switch back

    machine.update(deltaTime);
    machine.render(ctx);

    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('./sw.js').catch(() => { /* offline support is best-effort */ });
    });
  }
}

window.addEventListener('load', init);
