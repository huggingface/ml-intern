import { CANVAS_WIDTH, CANVAS_HEIGHT, PALETTE } from '../constants.js';
import { drawBackground, drawFrame } from '../render.js';

const MIN_DISPLAY_TIME = 1500;

export function createGameOverState(game, machine) {
  let time = 0;
  let blink = 0;

  return {
    enter() {
      time = 0;
      blink = 0;
      game.audio.playSound('gameOver');
      game.audio.stopUfoLoop();
    },
    exit() {},
    update(dt) {
      time += dt;
      blink += dt;
      game.input.update();
      if (time * 1000 > MIN_DISPLAY_TIME && game.input.firePressed) {
        machine.transition('attract');
      }
    },
    render(ctx) {
      drawBackground(ctx, time * 1000);
      drawFrame(ctx);

      ctx.save();
      ctx.textAlign = 'center';
      ctx.fillStyle = PALETTE.ufo;
      ctx.shadowColor = PALETTE.gold;
      ctx.shadowBlur = 6;
      ctx.font = '18px Georgia, serif';
      ctx.fillText('GAME OVER', CANVAS_WIDTH / 2, 110);

      ctx.font = '8px Georgia, serif';
      ctx.fillStyle = PALETTE.text;
      ctx.shadowBlur = 0;
      ctx.fillText(`SCORE  ${game.score}`, CANVAS_WIDTH / 2, 135);
      ctx.fillText(`HIGH SCORE  ${game.highScore}`, CANVAS_WIDTH / 2, 150);

      if (Math.floor(blink * 2) % 2 === 0) {
        ctx.fillStyle = PALETTE.goldBright;
        ctx.fillText('PRESS FIRE TO CONTINUE', CANVAS_WIDTH / 2, CANVAS_HEIGHT - 50);
      }
      ctx.restore();
    },
  };
}
