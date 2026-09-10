import { CANVAS_WIDTH, CANVAS_HEIGHT, PALETTE } from '../constants.js';
import { drawBackground, drawFrame, drawAlien } from '../render.js';
import { Alien } from '../entities/alien.js';

const LEGEND = [
  { type: 'squid', label: '= 30 PTS' },
  { type: 'crab', label: '= 20 PTS' },
  { type: 'octopus', label: '= 10 PTS' },
];

export function createAttractState(game, machine) {
  let time = 0;
  let blink = 0;
  const legendAliens = LEGEND.map((entry, i) => {
    const a = new Alien(i === 0 ? 0 : i === 1 ? 1 : 3, 0, 0, 0);
    return a;
  });

  return {
    enter() {
      time = 0;
      game.audio.stopUfoLoop();
    },
    exit() {},
    update(dt) {
      time += dt;
      blink += dt;
      game.input.update();
      if (game.input.firePressed) {
        game.audio.unlock();
        game.startNewGame();
        machine.transition('play');
      }
    },
    render(ctx) {
      drawBackground(ctx, time * 1000);
      drawFrame(ctx);

      ctx.save();
      ctx.textAlign = 'center';
      ctx.fillStyle = PALETTE.goldBright;
      ctx.shadowColor = PALETTE.gold;
      ctx.shadowBlur = 6;
      ctx.font = '16px Georgia, serif';
      ctx.fillText('SPACE', CANVAS_WIDTH / 2, 56);
      ctx.fillText('INVADERS', CANVAS_WIDTH / 2, 74);
      ctx.font = 'italic 6px Georgia, serif';
      ctx.fillStyle = PALETTE.gold;
      ctx.shadowBlur = 0;
      ctx.fillText('— nouveau —', CANVAS_WIDTH / 2, 84);
      ctx.restore();

      ctx.save();
      ctx.textAlign = 'left';
      ctx.font = '6px Georgia, serif';
      ctx.fillStyle = PALETTE.text;
      let ly = 118;
      ctx.fillText('* SCORE ADVANCE TABLE *', CANVAS_WIDTH / 2 - 52, ly - 12);
      for (let i = 0; i < legendAliens.length; i++) {
        const alien = legendAliens[i];
        const x = CANVAS_WIDTH / 2 - 40;
        drawAlien(ctx, alien, x, ly - 3, 0);
        ctx.fillText(LEGEND[i].label, x + 14, ly);
        ly += 16;
      }
      ctx.restore();

      if (Math.floor(blink * 2) % 2 === 0) {
        ctx.save();
        ctx.textAlign = 'center';
        ctx.fillStyle = PALETTE.goldBright;
        ctx.font = '8px Georgia, serif';
        ctx.fillText('PRESS FIRE TO START', CANVAS_WIDTH / 2, CANVAS_HEIGHT - 40);
        ctx.restore();
      }

      ctx.save();
      ctx.textAlign = 'center';
      ctx.fillStyle = PALETTE.text;
      ctx.font = '6px Georgia, serif';
      ctx.fillText('ARROWS / A-D MOVE   SPACE FIRE', CANVAS_WIDTH / 2, CANVAS_HEIGHT - 20);
      ctx.restore();
    },
  };
}
