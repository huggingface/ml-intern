import { alienTypeForRow, pointsForType } from '../constants.js';

export class Alien {
  constructor(row, col, offsetX, offsetY) {
    this.row = row;
    this.col = col;
    this.offsetX = offsetX;
    this.offsetY = offsetY;
    this.type = alienTypeForRow(row);
    this.points = pointsForType(this.type);
    this.isAlive = true;
    this.frame = 0;
    this.width = this.type === 'squid' ? 8 : this.type === 'crab' ? 11 : 12;
    this.height = 8;
  }
}
