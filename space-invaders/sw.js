const CACHE_NAME = 'space-invaders-nouveau-v1';

const ASSET_LIST = [
  './',
  './index.html',
  './manifest.webmanifest',
  './css/style.css',
  './js/game.js',
  './js/constants.js',
  './js/input.js',
  './js/audio.js',
  './js/render.js',
  './js/collision.js',
  './js/entities/alien.js',
  './js/entities/alienFireController.js',
  './js/entities/bullet.js',
  './js/entities/bunker.js',
  './js/entities/explosion.js',
  './js/entities/mysteryShip.js',
  './js/entities/player.js',
  './js/entities/swarm.js',
  './js/state/stateMachine.js',
  './js/state/attractState.js',
  './js/state/playState.js',
  './js/state/gameOverState.js',
  './icons/icon.svg',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/icon-maskable-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSET_LIST)),
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) => Promise.all(
      names.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name)),
    )),
  );
  self.clients.claim();
});

// Cache-first, falling back to network — ideal for a self-contained game
// where every asset is required for the app to function at all.
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request)),
  );
});
