// Polls keyboard + touch input and exposes a single, unified state object.
export class InputHandler {
  constructor() {
    this.left = false;
    this.right = false;
    this.fire = false;
    // Edge-triggered convenience for menu / restart actions. Latched by the
    // raw press event itself (not just polled state) so a tap shorter than
    // one animation frame is never dropped.
    this.firePressed = false;
    this._firePressLatch = false;

    window.addEventListener('keydown', (e) => this._onKey(e, true));
    window.addEventListener('keyup', (e) => this._onKey(e, false));
  }

  _onKey(e, isDown) {
    switch (e.code) {
      case 'ArrowLeft':
      case 'KeyA':
        this.left = isDown;
        e.preventDefault();
        break;
      case 'ArrowRight':
      case 'KeyD':
        this.right = isDown;
        e.preventDefault();
        break;
      case 'Space':
      case 'ArrowUp':
      case 'KeyW':
        if (isDown && !this.fire) this._firePressLatch = true;
        this.fire = isDown;
        e.preventDefault();
        break;
    }
  }

  bindTouchControls(root) {
    const bind = (selector, prop) => {
      const el = root.querySelector(selector);
      if (!el) return;
      const start = (e) => {
        e.preventDefault();
        if (prop === 'fire' && !this.fire) this._firePressLatch = true;
        this[prop] = true;
      };
      const end = (e) => { e.preventDefault(); this[prop] = false; };
      el.addEventListener('touchstart', start, { passive: false });
      el.addEventListener('touchend', end, { passive: false });
      el.addEventListener('touchcancel', end, { passive: false });
      el.addEventListener('mousedown', start);
      el.addEventListener('mouseup', end);
      el.addEventListener('mouseleave', end);
    };
    bind('[data-control="left"]', 'left');
    bind('[data-control="right"]', 'right');
    bind('[data-control="fire"]', 'fire');
  }

  // Call once per frame: drains the latch into a one-frame-wide pulse.
  update() {
    this.firePressed = this._firePressLatch;
    this._firePressLatch = false;
  }
}
