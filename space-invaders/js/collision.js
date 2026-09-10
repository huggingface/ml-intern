import { Explosion } from './entities/explosion.js';
import { PALETTE } from './constants.js';

// Broad-phase: cheap Axis-Aligned Bounding Box overlap test used for every
// entity-to-entity pair before any expensive pixel work is considered.
export function isColliding(a, b) {
  return a.x < b.x + b.width
    && a.x + a.width > b.x
    && a.y < b.y + b.height
    && a.y + a.height > b.y;
}

function bunkerNarrowPhaseHit(bunker, bullet) {
  // Sample a few points along the bullet's leading edge against the bunker's
  // live pixel bitmap — the narrow-phase check, only reached after AABB hits.
  const points = [
    [bullet.x + bullet.width / 2, bullet.y],
    [bullet.x + bullet.width / 2, bullet.y + bullet.height],
    [bullet.x, bullet.y + bullet.height / 2],
    [bullet.x + bullet.width, bullet.y + bullet.height / 2],
  ];
  for (const [px, py] of points) {
    if (bunker.isSolidAt(px, py)) return { x: px, y: py };
  }
  return null;
}

export function checkCollisions(game) {
  const { player, swarm, mysteryShip, bunkers, playerBullets, alienBullets, explosions, audio } = game;

  // Player bullets vs aliens.
  for (const bullet of playerBullets) {
    if (!bullet.isAlive) continue;
    for (const alien of swarm.aliens) {
      if (!alien.isAlive) continue;
      const pos = swarm.alienScreenPosition(alien);
      const rect = { x: pos.x, y: pos.y, width: alien.width, height: alien.height };
      if (isColliding(bullet, rect)) {
        alien.isAlive = false;
        bullet.isAlive = false;
        game.score += alien.points;
        game.onScoreChanged();
        explosions.push(new Explosion(pos.x + alien.width / 2, pos.y + alien.height / 2, PALETTE.goldBright));
        audio.playSound('alienHit');
        break;
      }
    }
  }

  // Player bullets vs mystery ship.
  if (mysteryShip.active) {
    for (const bullet of playerBullets) {
      if (!bullet.isAlive) continue;
      if (isColliding(bullet, mysteryShip)) {
        bullet.isAlive = false;
        const points = mysteryShip.scoreValue();
        game.score += points;
        game.onScoreChanged();
        explosions.push(new Explosion(
          mysteryShip.x + mysteryShip.width / 2,
          mysteryShip.y + mysteryShip.height / 2,
          PALETTE.ufo,
          0.5,
        ));
        audio.playSound('ufoHit');
        mysteryShip.despawn();
      }
    }
  }

  // Alien bullets vs player.
  if (player.isAlive) {
    for (const bullet of alienBullets) {
      if (!bullet.isAlive) continue;
      if (isColliding(bullet, player)) {
        bullet.isAlive = false;
        game.onPlayerHit();
      }
    }
  }

  // Bullets vs bunkers (AABB broad-phase, then pixel narrow-phase + damage).
  const allBullets = [...playerBullets, ...alienBullets];
  for (const bullet of allBullets) {
    if (!bullet.isAlive) continue;
    for (const bunker of bunkers) {
      if (!isColliding(bullet, bunker.getBounds())) continue;
      const hit = bunkerNarrowPhaseHit(bunker, bullet);
      if (hit) {
        bunker.applyDamage(hit.x, hit.y);
        bullet.isAlive = false;
        break;
      }
    }
  }

  // Reset the player's single-shot lock whenever its bullet has resolved.
  for (const bullet of playerBullets) {
    if (!bullet.isAlive) player.onBulletResolved();
  }

  game.playerBullets = playerBullets.filter((b) => b.isAlive);
  game.alienBullets = alienBullets.filter((b) => b.isAlive);
}
