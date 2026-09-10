import { CANVAS_WIDTH, PALETTE } from '../constants.js';
import { drawBackground, drawFrame } from '../render.js';
import { checkCollisions } from '../collision.js';
import { Explosion } from '../entities/explosion.js';

const RESPAWN_DELAY = 1200;
const WAVE_CLEAR_DELAY = 1600;
const GAME_OVER_DELAY = 900;

export function createPlayState(game, machine) {
  let phase = 'running'; // running | respawning | waveClear | dying
  let timer = 0;
  let bgTime = 0;

  function beginRespawn() {
    phase = 'respawning';
    timer = RESPAWN_DELAY;
  }

  function beginGameOverTransition() {
    phase = 'dying';
    timer = GAME_OVER_DELAY;
  }

  function beginWaveClear() {
    phase = 'waveClear';
    timer = WAVE_CLEAR_DELAY;
    game.audio.playSound('waveClear');
  }

  return {
    enter() {
      phase = 'running';
      timer = 0;
    },
    exit() {
      game.audio.stopUfoLoop();
    },
    update(dt) {
      bgTime += dt;
      game.input.update();

      if (phase === 'respawning') {
        timer -= dt * 1000;
        if (timer <= 0) {
          game.player.isAlive = true;
          game.player.x = (CANVAS_WIDTH - game.player.width) / 2;
          game.player.canFire = true;
          phase = 'running';
        }
        return;
      }

      if (phase === 'waveClear') {
        timer -= dt * 1000;
        if (timer <= 0) {
          game.nextWave();
          phase = 'running';
        }
        return;
      }

      if (phase === 'dying') {
        timer -= dt * 1000;
        if (timer <= 0) machine.transition('gameOver');
        return;
      }

      // --- normal gameplay update ---
      game.player.update(dt, game.input);
      game.swarm.update(dt, game.alienBullets);
      game.mysteryShip.update(dt);

      for (const b of game.playerBullets) b.update(dt);
      for (const b of game.alienBullets) b.update(dt);
      for (const e of game.explosions) e.update(dt);
      game.explosions = game.explosions.filter((e) => e.isAlive);

      checkCollisions(game);

      // Aliens reaching the player's line is an instant loss condition.
      if (game.swarm.aliveCount > 0 && game.swarm.bottomMostY() >= game.player.y) {
        game.lives = 0;
        game.player.isAlive = false;
        game.explosions.push(new Explosion(
          game.player.x + game.player.width / 2,
          game.player.y + game.player.height / 2,
          PALETTE.player,
          0.6,
        ));
        game.audio.playSound('playerExplosion');
      }

      if (!game.player.isAlive) {
        if (game.lives <= 0) beginGameOverTransition();
        else beginRespawn();
        return;
      }

      if (game.swarm.isDefeated) {
        beginWaveClear();
      }
    },
    render(ctx) {
      drawBackground(ctx, bgTime * 1000);
      drawFrame(ctx);

      for (const bunker of game.bunkers) bunker.render(ctx);
      game.swarm.render(ctx);
      game.mysteryShip.render(ctx);
      game.player.render(ctx);
      for (const b of game.playerBullets) b.render(ctx);
      for (const b of game.alienBullets) b.render(ctx);
      for (const e of game.explosions) e.render(ctx);

      if (phase === 'waveClear') {
        ctx.save();
        ctx.textAlign = 'center';
        ctx.fillStyle = PALETTE.goldBright;
        ctx.font = '10px Georgia, serif';
        ctx.shadowColor = PALETTE.gold;
        ctx.shadowBlur = 5;
        ctx.fillText(`WAVE ${game.wave} CLEARED`, CANVAS_WIDTH / 2, 130);
        ctx.restore();
      }
    },
  };
}
