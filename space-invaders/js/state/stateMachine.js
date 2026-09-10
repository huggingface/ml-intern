export class StateMachine {
  constructor(states) {
    this.states = states;
    this.current = null;
  }

  start(initial) {
    this.transition(initial);
  }

  transition(name) {
    if (this.current && this.states[this.current].exit) this.states[this.current].exit();
    this.current = name;
    if (this.states[this.current].enter) this.states[this.current].enter();
  }

  update(dt) {
    this.states[this.current].update(dt);
  }

  render(ctx) {
    this.states[this.current].render(ctx);
  }
}
