# 00 · Setup

Everything needed to go from a clean machine to a running model. Read the
requirements, then follow the section for your platform.

The engine itself has **no third-party libraries** — the tools below are a
compiler and a build system, not dependencies of the code. Only the optional web
interface pulls in packages, and only for the browser.

---

## Requirements at a glance

### Hardware

| | Minimum | Comfortable | Why |
|---|---|---|---|
| **Disk** | 35 GB free | 50 GB | The checkpoint is 29.3 GiB; the build and a spare copy need the rest |
| **RAM** | 6 GB | 16 GB+ | 2.44 GiB stays resident always. Spare RAM becomes page cache, which is what makes expert streaming fast |
| **CPU** | Any x86-64 | AVX2 + FMA | Without AVX2 the portable scalar kernels are used — correct, but several times slower |
| **GPU** | none | none | There is no GPU path. This is a CPU engine by design |

An SSD matters more than clock speed. Experts are read on demand, and on a
spinning disk the seeks dominate everything else.

### Software

| Tool | Version | Needed for | Optional? |
|---|---|---|---|
| **CMake** | 3.20 or newer | Configuring the build | Required |
| **C++ compiler** | GCC 10+ (64-bit), or MSVC 2019 16.11+ | C++20 | Required |
| **Ninja** | any | Faster builds | Optional — falls back to Makefiles |
| **curl** | any | Downloading the checkpoint | Ships with Windows 10 1803+ |
| **Python** | 3.8+ | `make_test_model.py`, `reference_logits.py` | Optional |
| **Node.js** | 18, 20, or 22+ | The web interface | Optional |

Vite declares `^18 || ^20 || >=22`, so Node **19 and 21 will not work** — they are
odd-numbered development releases that never became LTS. Install an even-numbered
version.

**Python needs no packages** for the synthetic test model — `make_test_model.py`
uses only the standard library. Only `reference_logits.py`, which compares
LiteMind's logits against Hugging Face, needs `torch` and `transformers`, and
that is a verification tool, not part of building or running.

---

## Windows

The path this project is developed on.

### 1. Check what you already have

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_environment.ps1
```

This inspects compilers, CMake, Python, MSYS2, disk, RAM and CPU in one pass and
changes nothing. It tells you what is missing rather than failing one item at a
time. If everything reports green, skip to step 3.

### 2. Install the toolchain

**MSYS2** provides GCC, CMake and Ninja together and is the recommended route:

```powershell
winget install MSYS2.MSYS2
```

Then open **MSYS2 MINGW64** from the Start menu (not the plain "MSYS2" shell —
the environment matters) and install the toolchain:

```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
```

If `winget` exits with code 1, MSYS2 is already installed somewhere. Find it and
install only the packages:

```powershell
& "C:\msys64\usr\bin\bash.exe" -lc "pacman -Sy --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-ninja"
```

**Visual Studio 2019 16.11 or newer** also works. `build.ps1` finds `cl.exe`
when no suitable GCC is present.

> **Note:** `build.ps1` deliberately does not take the first `g++` on `PATH`. A
> 32-bit or ancient compiler on `PATH` is a common cause of confusing failures,
> so each candidate is tested before it is chosen. Override with
> `-CompilerPath D:\path\to\g++.exe` if you need a specific one.

**Node.js**, only if you want the web interface:

```powershell
winget install OpenJS.NodeJS.LTS
```

Close and reopen the terminal afterwards so `PATH` updates.

### 3. Build and test

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -RunTests
```

All suites should pass. They need no model files.

### 4. Prove the build works without downloading 31 GB

```powershell
python tools\make_test_model.py models-test
.\build\bin\LiteMind.exe models-test -p "hello" -n 8
```

A few megabytes of random weights. The text is meaningless — random weights
produce random tokens — but a run that completes proves the loader, the tensor
shapes and the forward pass are all correct.

### 5. Download the real checkpoint

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_model.ps1
```

29.3 GiB across four shards. It resumes if interrupted, so a dropped connection
is not a restart.

For the **instruction-tuned** checkpoint — the one that answers questions rather
than continuing text — pass the repository explicitly:

```powershell
powershell -File scripts\download_model.ps1 -Repository deepseek-ai/DeepSeek-V2-Lite-Chat -Destination models
```

See [the chat template](13-chat-template.md) for why that distinction changes
the replies so completely.

### 6. Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

Builds anything stale, then prompts for input. Settings live in
[`litemind.json`](../litemind.json); see [command line and settings](10-cli-and-config.md).

### 7. The web interface (optional)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ui.ps1
```

First run installs the browser packages, which takes a minute and writes about
40,000 files into `ui/web/node_modules`. That directory is ignored by git and
regenerated from `package-lock.json` on any machine — it is never committed.

Then open <http://localhost:5174>.

---

## Linux and macOS

There are no PowerShell wrappers, but nothing in the engine is Windows-specific.

```bash
# Debian / Ubuntu
sudo apt install build-essential cmake ninja-build

# macOS
xcode-select --install
brew install cmake ninja
```

Build and test:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Run:

```bash
./build/bin/LiteMind models -p "The capital of France is" -n 32
```

Download the checkpoint with `huggingface-cli download deepseek-ai/DeepSeek-V2-Lite-Chat --local-dir models`,
or any tool that fetches the four `.safetensors` shards plus `config.json`,
`tokenizer.json` and `tokenizer_config.json`.

The web interface is platform-independent:

```bash
cd ui/web && npm install && npm run build
cd ../.. && node ui/server/server.mjs
```

---

## Script options

Every script prints its own help with `Get-Help scripts\<name>.ps1 -Full`.

| Script | Option | Effect |
|---|---|---|
| `build.ps1` | `-RunTests` | Run the suites after building |
| | `-Clean` | Delete the build directory first |
| | `-CompilerPath <path>` | Use a specific `g++.exe` |
| | `-Toolchain MinGW` or `-Toolchain MSVC` | Force one, instead of auto-detecting |
| `download_model.ps1` | `-Repository <id>` | Which Hugging Face repository |
| | `-Destination <dir>` | Where to put it (default `models`) |
| | `-Revision <ref>` | A branch or commit (default `main`) |
| `run.ps1` | `-Config <file>` | Settings file other than `litemind.json` |
| | `-SkipBuild` | Do not rebuild first |
| `ui.ps1` | `-Port <n>` | Listen somewhere other than 5174 |
| | `-Dev` | Vite dev server with hot reload |
| | `-SkipBuild` | Do not rebuild the engine |

---

## Troubleshooting

**`cmake` is not recognised**
It is installed inside MSYS2 but not on the Windows `PATH`. Either run from the
MSYS2 MINGW64 shell, or add `C:\msys64\mingw64\bin` to `PATH`, or install CMake
for Windows separately.

**The build picks the wrong compiler**
Pass it explicitly: `scripts\build.ps1 -CompilerPath C:\msys64\mingw64\bin\g++.exe`.

**`AVX2 available but unused`, or the portable scalar kernels are reported**
The build did not enable native tuning. Confirm you are building Release; the
startup line reports which kernels are active.

**The download stops partway**
Re-run `download_model.ps1`. It resumes with `curl -C -`. A `curl exit 33` on an
already-complete file is expected and handled — the server returns 416 for a
range request past the end.

**Replies wander, or answer a question you did not ask**
Almost always the checkpoint, not the engine. A base model continues text
instead of answering; see [the chat template](13-chat-template.md).

**Long replies repeat the same phrase forever**
Greedy decoding with no repetition penalty. Set `temperature` to `0.7` and
`repeat_penalty` to `1.15` in `litemind.json`.

**0% expert cache hit rate**
A budget below about 2.6 GB evicts every expert before it can be reused — one
token touches that much. Use `0` (page cache) or at least `4`. See
[the expert cache](07-expert-cache.md).

**The interface says the engine is not built**
The server looks for `build/bin/LiteMind[.exe]`. Run `scripts\build.ps1` first,
or start the server from the repository root so the relative paths resolve.

---

## What gets installed where

| Path | What | In git? |
|---|---|---|
| `build/` | Compiled objects and `LiteMind.exe` | No |
| `models/` | The checkpoint, 29.3 GiB | No |
| `models-test/` | Synthetic test model, a few MB | No |
| `ui/web/node_modules/` | Browser packages, ~40,000 files | No |
| `ui/web/dist/` | Built interface bundle | No |
| `litemind.json` | Your settings | Yes |

Nothing in the first five is committed. A fresh clone plus this guide reproduces
all of it.
