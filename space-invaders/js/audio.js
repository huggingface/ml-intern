// AudioManager — all sound effects are synthesized procedurally at load time
// (no external .wav assets), then played back as low-latency AudioBufferSourceNodes.
export class AudioManager {
  constructor() {
    this.ctx = null;
    this.buffers = new Map();
    this.masterGain = null;
    this.ufoSource = null;
    this.ufoGain = null;
    this.enabled = true;
  }

  // AudioContext must be created/resumed from a user gesture on most browsers.
  unlock() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') this.ctx.resume();
      return;
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx();
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 0.55;
    this.masterGain.connect(this.ctx.destination);
    this._generateAll();
  }

  _makeBuffer(duration, fn) {
    const sr = this.ctx.sampleRate;
    const length = Math.max(1, Math.floor(sr * duration));
    const buffer = this.ctx.createBuffer(1, length, sr);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i++) {
      const t = i / sr;
      data[i] = fn(t, i, sr);
    }
    return buffer;
  }

  _generateAll() {
    const noise = (t, decay) => (Math.random() * 2 - 1) * Math.exp(-t * decay);

    // Player shot: quick descending square-ish sweep, bright and thin.
    this.buffers.set('shoot', this._makeBuffer(0.16, (t) => {
      const freq = 900 - t * 3200;
      const env = Math.exp(-t * 14);
      return Math.sign(Math.sin(2 * Math.PI * freq * t)) * 0.35 * env;
    }));

    // Alien hit: a short "pop" — filtered noise burst with a falling thud.
    this.buffers.set('alienHit', this._makeBuffer(0.14, (t) => {
      const thud = Math.sin(2 * Math.PI * (140 - t * 400) * t) * 0.5;
      return (noise(t, 24) * 0.6 + thud) * Math.exp(-t * 18);
    }));

    // Player explosion: bigger, longer, low rumbling noise burst.
    this.buffers.set('playerExplosion', this._makeBuffer(0.5, (t) => {
      const rumble = Math.sin(2 * Math.PI * (90 - t * 60) * t) * 0.4;
      return (noise(t, 5) * 0.8 + rumble) * Math.exp(-t * 4.2);
    }));

    // UFO hit: rewarding bright descending chime/arpeggio.
    this.buffers.set('ufoHit', this._makeBuffer(0.55, (t) => {
      const notes = [1568, 1244, 987, 1568];
      const idx = Math.min(notes.length - 1, Math.floor(t * 9));
      const freq = notes[idx];
      const env = Math.exp(-((t % 0.14)) * 10);
      return Math.sin(2 * Math.PI * freq * t) * 0.4 * env;
    }));

    // UFO flying loop: two alternating warbling tones (76477-chip style siren).
    this.buffers.set('ufoFlying', this._makeBuffer(0.6, (t) => {
      const warble = Math.sin(2 * Math.PI * 6 * t);
      const freq = 300 + warble * 90;
      return Math.sin(2 * Math.PI * freq * t) * 0.25;
    }));

    // Four descending "march" notes forming the alien heartbeat.
    const marchFreqs = [110, 98, 87, 82];
    marchFreqs.forEach((freq, idx) => {
      this.buffers.set(`march${idx + 1}`, this._makeBuffer(0.11, (t) => {
        const env = Math.exp(-t * 16);
        return (Math.sign(Math.sin(2 * Math.PI * freq * t)) * 0.5 + Math.sin(2 * Math.PI * freq * 2 * t) * 0.2) * env;
      }));
    });

    // Extra life awarded: bright ascending arpeggio.
    this.buffers.set('extraLife', this._makeBuffer(0.5, (t) => {
      const notes = [523, 659, 784, 1046];
      const idx = Math.min(notes.length - 1, Math.floor(t * 9));
      const env = Math.exp(-((t % 0.12)) * 9);
      return Math.sin(2 * Math.PI * notes[idx] * t) * 0.35 * env;
    }));

    // Wave clear fanfare.
    this.buffers.set('waveClear', this._makeBuffer(0.7, (t) => {
      const notes = [392, 523, 659, 784, 1046];
      const idx = Math.min(notes.length - 1, Math.floor(t * 7.5));
      const env = Math.exp(-((t % 0.13)) * 7);
      return Math.sin(2 * Math.PI * notes[idx] * t) * 0.32 * env;
    }));

    // Game over: descending, mournful.
    this.buffers.set('gameOver', this._makeBuffer(1.1, (t) => {
      const freq = 220 - t * 120;
      const env = Math.exp(-t * 1.6);
      return Math.sin(2 * Math.PI * Math.max(40, freq) * t) * 0.3 * env;
    }));
  }

  playSound(name, { loop = false, gain = 1 } = {}) {
    if (!this.enabled || !this.ctx) return null;
    const buffer = this.buffers.get(name);
    if (!buffer) return null;
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.loop = loop;
    const g = this.ctx.createGain();
    g.gain.value = gain;
    source.connect(g).connect(this.masterGain);
    source.start(0);
    return source;
  }

  startUfoLoop() {
    if (this.ufoSource || !this.ctx) return;
    this.ufoSource = this.playSound('ufoFlying', { loop: true, gain: 0.6 });
  }

  stopUfoLoop() {
    if (this.ufoSource) {
      try { this.ufoSource.stop(); } catch (e) { /* already stopped */ }
      this.ufoSource = null;
    }
  }

  playMarchStep(step) {
    this.playSound(`march${(step % 4) + 1}`);
  }
}
