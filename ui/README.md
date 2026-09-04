# LiteMind web interface

A React + Material UI front end over a small Node backend that drives the
engine. See [`docs/14-web-ui.md`](../docs/14-web-ui.md) for how it works and why.

```powershell
powershell -ExecutionPolicy Bypass -File ..\scripts\ui.ps1
```

Then open <http://localhost:5174>.

| | |
|---|---|
| `server/server.mjs` | HTTP, static files, server-sent events. No packages. |
| `server/litemind.mjs` | Spawning the engine and parsing its JSON events. |
| `web/` | The Vite + React + Material UI bundle. |

The chat remembers. The engine does not: each prompt is a fresh process, so the
browser keeps the transcript and sends the finished exchanges back with every new
question. Whatever no longer fits the context window is dropped, oldest first,
and the interface says how much. **Remember** turns it off, which makes every
question independent again.

The backend uses `node:http` only. The browser bundle has the dependencies
Material UI implies; the C++ engine has none and is untouched by any of this.
