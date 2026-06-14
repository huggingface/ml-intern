# Agent

Async agent loop with LiteLLM.

## Architecture

**Queue-based async system:**
- Submissions in (user input) → Agent Loop → Events output for possible UI updates
- Session maintains state (context + tools) for possible future Context Engineering
- Handlers operations like (USER_INPUT, COMPACT, UNDO, SHUTDOWN) for possible UI control

## Components

| Component | Purpose | Long Term Goal |
|-----------|---------|----------------|
| **`agent_loop.py`** | Core agentic loop: processes user input, calls LLM via LiteLLM, executes tool calls iteratively until completion, emits events | Support parallel tool execution, streaming responses, and advanced reasoning patterns |
| **`session.py`** | Maintains session state and interaction with potential UI (context, config, event queue), handles interrupts, assigns unique session IDs for tracing | Enable plugging in different UIs (CLI, web, API, programmatic etc.) |
| **`tools.py`** | `ToolRouter` manages potential built-in tools (e.g. bash, read_file, write_file which are dummy implementations rn) + MCP tools, converts specs to OpenAI format | Be the place for tools that can be used by the agent. All crazy tool design happens here. |
| **`context_manager/`** | Manages conversation history, very rudimentary context engineering support | Implement intelligent context engineering to keep the agent on track |
| **`config.py`** | Loads JSON config for the agent | Support different configs etc. |
| **`main.py`** | Interactive CLI with async queue architecture (submission→agent, agent→events) (simple way to interact with the agent now)| Serve as reference implementation for other UIs (web, API, programmatic) |
