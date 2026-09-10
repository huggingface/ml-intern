// Core game constants — the "magic numbers" centralized for easy tuning.
export const CANVAS_WIDTH = 224;
export const CANVAS_HEIGHT = 256;

export const ALIEN_ROWS = 5;
export const ALIEN_COLS = 11;
export const ALIEN_H_SPACING = 16;
export const ALIEN_V_SPACING = 16;
export const ALIEN_START_X = 16;
export const ALIEN_START_Y = 34;
export const ALIEN_DROP_DISTANCE = 8;
export const ALIEN_MAX_DEPTH_Y = 170;

export const ALIEN_TYPES = {
  SQUID: { name: 'squid', row: 0, width: 8, height: 8, points: 30 },
  CRAB: { name: 'crab', rows: [1, 2], width: 11, height: 8, points: 20 },
  OCTOPUS: { name: 'octopus', rows: [3, 4], width: 12, height: 8, points: 10 },
};

export function alienTypeForRow(row) {
  if (row === 0) return 'squid';
  if (row === 1 || row === 2) return 'crab';
  return 'octopus';
}

export function pointsForType(type) {
  if (type === 'squid') return 30;
  if (type === 'crab') return 20;
  return 10;
}

export const BASE_MOVE_INTERVAL = 900; // ms, slowest possible step (full swarm)
export const MIN_MOVE_INTERVAL = 45; // ms, fastest possible step (last alien)

export const PLAYER_WIDTH = 13;
export const PLAYER_HEIGHT = 8;
export const PLAYER_SPEED = 90; // px/sec
export const PLAYER_START_LIVES = 3;
export const EXTRA_LIFE_SCORE = 1500;

export const PLAYER_BULLET_SPEED = 220; // px/sec upward
export const ALIEN_BULLET_SPEED = 110; // px/sec downward

export const MYSTERY_SHIP_WIDTH = 16;
export const MYSTERY_SHIP_HEIGHT = 7;
export const MYSTERY_SHIP_SPEED = 60; // px/sec
export const MYSTERY_SHIP_MIN_DELAY = 12000;
export const MYSTERY_SHIP_MAX_DELAY = 22000;

export const UFO_SCORE_TABLE = [100, 50, 50, 100, 150, 100, 100, 50, 300, 100, 100, 100, 50, 150, 100];

export const BUNKER_COUNT = 4;
export const BUNKER_WIDTH = 22;
export const BUNKER_HEIGHT = 16;
export const BUNKER_Y = 200;

export const ALIEN_FIRE_MIN_INTERVAL = 350;
export const ALIEN_FIRE_MAX_INTERVAL = 1000;

// Column target sequences for the three canonical shot "personalities".
export const FIRE_PATTERNS = {
  ROLLING: [4, 8, 2, 6, 10, 0, 9, 3, 7, 1, 5],
  PLUNGER: [1, 7, 1, 1, 1, 4, 8, 2, 6, 10, 0, 9, 3, 7, 1],
  SQUIGGLY: [5, 9, 3, 7, 1, 5, 10, 4, 8, 2, 6, 0, 9, 3, 7],
};

// Art Nouveau palette — jewel tones on deep indigo, gold linework throughout.
export const PALETTE = {
  bgTop: '#120a24',
  bgBottom: '#1d1030',
  gold: '#d4af37',
  goldBright: '#f3d47a',
  goldDim: '#8a6a2a',
  squid: '#3ea88a',
  squidDark: '#215e4c',
  crab: '#a56cc1',
  crabDark: '#5c3670',
  octopus: '#d4763f',
  octopusDark: '#7a4022',
  player: '#4fb8c9',
  playerDark: '#245a63',
  ufo: '#c0384f',
  ufoDark: '#6e1c2b',
  bunker: '#7a6a9a',
  bunkerDark: '#3a2f52',
  bulletPlayer: '#f3d47a',
  bulletAlien: '#e0668a',
  text: '#e8d9a8',
};
