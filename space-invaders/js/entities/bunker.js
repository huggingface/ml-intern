import { BUNKER_WIDTH, BUNKER_HEIGHT, PALETTE } from '../constants.js';

const DAMAGE_BLOCK = 3; // chunky erase granularity, matches the original's blocky look

export class Bunker {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.width = BUNKER_WIDTH;
    this.height = BUNKER_HEIGHT;
    this.canvas = document.createElement('canvas');
    this.canvas.width = BUNKER_WIDTH;
    this.canvas.height = BUNKER_HEIGHT;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.repair();
  }

  // Draws the pristine ornamental bunker — an Art Nouveau arch with carved
  // filigree — onto the off-screen bitmap that damage will later erase from.
  repair() {
    const c = this.ctx;
    c.clearRect(0, 0, this.width, this.height);
    c.fillStyle = PALETTE.bunker;
    c.strokeStyle = PALETTE.bunkerDark;
    c.lineWidth = 1;

    c.beginPath();
    c.moveTo(0, this.height);
    c.lineTo(0, 6);
    c.quadraticCurveTo(0, 0, 6, 0);
    c.lineTo(this.width - 6, 0);
    c.quadraticCurveTo(this.width, 0, this.width, 6);
    c.lineTo(this.width, this.height);
    // notch cut from underside like the classic bunker silhouette
    c.lineTo(this.width * 0.68, this.height);
    c.quadraticCurveTo(this.width * 0.55, this.height - 6, this.width * 0.5, this.height - 6);
    c.quadraticCurveTo(this.width * 0.45, this.height - 6, this.width * 0.32, this.height);
    c.closePath();
    c.fill();
    c.stroke();

    // Ornamental filigree veins.
    c.strokeStyle = PALETTE.goldDim;
    c.lineWidth = 0.5;
    c.beginPath();
    c.moveTo(3, this.height - 3);
    c.quadraticCurveTo(this.width / 2, 2, this.width - 3, this.height - 3);
    c.stroke();

    this.isDestroyed = false;
  }

  getBounds() {
    return { x: this.x, y: this.y, width: this.width, height: this.height };
  }

  // Narrow-phase impact: erase a chunky radius of pixels around the impact
  // point by zeroing their alpha, then write the modified bitmap back.
  applyDamage(worldX, worldY, radius = 4) {
    const localX = Math.round(worldX - this.x);
    const localY = Math.round(worldY - this.y);
    const imageData = this.ctx.getImageData(0, 0, this.width, this.height);
    const data = imageData.data;

    for (let by = -radius; by <= radius; by += DAMAGE_BLOCK) {
      for (let bx = -radius; bx <= radius; bx += DAMAGE_BLOCK) {
        if (bx * bx + by * by > radius * radius) continue;
        for (let dy = 0; dy < DAMAGE_BLOCK; dy++) {
          for (let dx = 0; dx < DAMAGE_BLOCK; dx++) {
            const px = localX + bx + dx;
            const py = localY + by + dy;
            if (px < 0 || py < 0 || px >= this.width || py >= this.height) continue;
            const idx = (py * this.width + px) * 4 + 3;
            data[idx] = 0;
          }
        }
      }
    }
    this.ctx.putImageData(imageData, 0, 0);
  }

  // Pixel-perfect hit test used as the narrow-phase check after an AABB hit.
  isSolidAt(worldX, worldY) {
    const localX = Math.round(worldX - this.x);
    const localY = Math.round(worldY - this.y);
    if (localX < 0 || localY < 0 || localX >= this.width || localY >= this.height) return false;
    const pixel = this.ctx.getImageData(localX, localY, 1, 1).data;
    return pixel[3] > 0;
  }

  render(ctx) {
    ctx.drawImage(this.canvas, this.x, this.y);
  }
}
