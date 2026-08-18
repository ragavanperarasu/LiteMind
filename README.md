# LiteMind

An educational, from-first-principles C++20 AI inference engine for the DeepSeek-V2 language model. LiteMind is designed to demonstrate advanced AI concepts including Mixture-of-Experts (MoE) routing, multi-head latent attention with RoPE encoding, and autoregressive text generation.

**Version:** 0.1.0  
**Language:** C++20  
**Build System:** CMake 3.20+

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Building the Project](#building-the-project)
- [Running the CLI](#running-the-cli)
- [Project Structure](#project-structure)
- [Component Reference](#component-reference)
- [Model Configuration](#model-configuration)
- [Testing](#testing)
- [Usage Examples](#usage-examples)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 Overview

LiteMind is an educational AI inference engine built from first principles in modern C++. It demonstrates how to build an efficient inference framework for large language models, specifically targeting the DeepSeek-V2 architecture. The project is structured to progressively add layers of functionality, starting with model loading and metadata inspection.

### What's Included

- ✅ Model configuration parsing and metadata inspection
- ✅ SafeTensors file format parsing
- ✅ DeepSeek-V2 architecture implementation with MoE routing
- ✅ Multi-head latent attention with RoPE positional encoding
- ✅ Tokenization and text generation pipeline
- ✅ KV-cache management for efficient generation
- ✅ Comprehensive test suite
- ✅ CLI tool for model inspection and generation

### What's Coming

- 🔄 Optimized CUDA kernels for GPU acceleration
- 🔄 Quantization support (INT8, FP8)
- 🔄 Multi-device inference
- 🔄 Performance benchmarking tools
- 🔄 Web API server

---

## ✨ Features

### Core ML Capabilities

- **DeepSeek-V2 Support**: Full implementation of the DeepSeek-V2 architecture
- **Mixture-of-Experts (MoE)**: Expert routing with 64 routed + 2 shared experts
- **Attention Mechanisms**: 
  - Multi-head latent attention (16 heads, 128 + 64 dims)
  - Rotary position embeddings (RoPE) with YaRN scaling
  - Context length: up to 163,840 tokens
- **Model Parameters**: ~27 billion parameters across 27 transformer layers
- **Vocabulary**: 102,400 token vocabulary with bfloat16 precision

### Technical Features

- **C++20 Standard**: Modern C++ with zero-cost abstractions
- **Header-Only Interfaces**: Clean public API design
- **Efficient Memory Management**: Stack allocation and smart pointers
- **Comprehensive Logging**: Debug, info, and error level logging
- **Unit Tests**: Full test coverage with CTest

---

## 🏗️ Architecture

LiteMind follows a modular, layered architecture:

```
┌─────────────────────────────────────┐
│  Application Layer (CLI)            │
├─────────────────────────────────────┤
│  DeepSeekRunner (Generation)        │
│  - Orchestrates inference pipeline  │
│  - Manages weight loading           │
│  - Handles token-by-token output    │
├─────────────────────────────────────┤
│  Core Components                    │
│  ┌────────────────────────────────┐ │
│  │ Tokenizer    │ Model Inference │ │
│  │              │ & Routing       │ │
│  ├────────────────────────────────┤ │
│  │ Attention    │ MoeRouter       │ │
│  │ KvCache      │ Sampler         │ │
│  └────────────────────────────────┘ │
├─────────────────────────────────────┤
│  Storage & Data Structures          │
│  ┌────────────────────────────────┐ │
│  │ SafeTensor   │ CpuTensor       │ │
│  │ WeightReader │ Tensor          │ │
│  └────────────────────────────────┘ │
├─────────────────────────────────────┤
│  Configuration & Logging            │
│  - Config, Logger                   │
└─────────────────────────────────────┘
```

### Component Interaction

```
main.cpp
  └─→ Cli (entry point)
      └─→ Model (config + tensors)
          ├─→ Config (model hyperparameters)
          ├─→ WeightReader (loads SafeTensors)
          ├─→ DeepSeekRunner (inference engine)
          │   ├─→ Tokenizer (text ↔ tokens)
          │   ├─→ Attention (transformer layer)
          │   ├─→ MoeRouter (expert selection)
          │   ├─→ KvCache (KV store)
          │   ├─→ Sampler (token sampling)
          │   └─→ CpuTensor (tensor operations)
          └─→ Logger (diagnostics)
```

---

## 📋 Prerequisites

### System Requirements

- **OS**: Windows, Linux, or macOS
- **Compiler**: MSVC 19.10+, GCC 11+, or Clang 12+
- **CMake**: 3.20 or later
- **RAM**: Minimum 8GB (16GB recommended for model loading)

### Dependencies

- **OpenBLAS**: For linear algebra operations
  - Windows: `libopenblas.dll.a`
  - Linux: `libblas3`, `libopenblas-dev`
  - macOS: Available via Homebrew

### Optional Dependencies

- **Testing**: CTest (included with CMake)
- **Documentation**: Doxygen (for generating API docs)

---

## 🔧 Installation

### Windows (with MSYS2/MinGW-w64)

1. **Install OpenBLAS** (if not already installed):
   ```powershell
   pacman -S mingw-w64-x86_64-openblas
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/LiteMind.git
   cd LiteMind
   ```

3. **Download the DeepSeek-V2 model** (4 files):
   ```bash
   # Place model files in models/ directory
   # - model-00001-of-000004.safetensors
   # - model-00002-of-000004.safetensors
   # - model-00003-of-000004.safetensors
   # - model-00004-of-000004.safetensors
   # - config.json
   # - tokenizer.json
   ```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get install cmake build-essential libblas3 libopenblas-dev

# Clone and navigate
git clone https://github.com/yourusername/LiteMind.git
cd LiteMind

# Download model files to models/ directory
```

### macOS

```bash
# Install dependencies
brew install cmake openblas

# Clone and navigate
git clone https://github.com/yourusername/LiteMind.git
cd LiteMind
```

---

## 🏗️ Building the Project

### Configure and Build

**Debug Build**:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

**Release Build** (recommended for inference):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**Visual Studio (Multi-config)**:
```bash
cmake -S . -B build
cmake --build build --config Release
```

### Build Output

- **Executable**: `build/bin/LiteMind` (or `build/bin/Release/LiteMind` on Windows)
- **Tests**: Various test executables in `build/bin/`
- **Build artifacts**: `build/CMakeFiles/`, `build/CMakeCache.txt`

### Build Options

```bash
# Enable or disable testing
cmake -S . -B build -DBUILD_TESTING=ON  # Default: ON

# Custom OpenBLAS path (if needed)
cmake -S . -B build -DOPENBLAS_PATH=/custom/path
```

---

## 🚀 Running the CLI

### Basic Usage

```bash
# Run with default model path (models/)
./build/bin/LiteMind

# Or specify a custom model directory
./build/bin/LiteMind /path/to/models
```

### What the CLI Does

The CLI performs the following operations:

1. **Model Inspection**:
   - Loads model configuration from `config.json`
   - Reads SafeTensors metadata from model shards
   - Reports total model size and parameter count

2. **MoE Analysis**:
   - Extracts Mixture-of-Experts configuration
   - Reports number of routed vs. shared experts
   - Calculates expert evaluations per token

3. **Interactive Prompt** (if model is loaded):
   - Accepts user input for text generation
   - Runs DeepSeek-V2 inference pipeline
   - Streams generated tokens to stdout

### Example Session

```bash
$ ./build/bin/LiteMind models

LiteMind
Version 0.1.0

[INFO] Loading model from models/
[INFO] Model size: 45.2 GB (4 SafeTensors shards)
[INFO] Parameters: 27B
[INFO] MoE Configuration:
        - Routed experts: 64
        - Shared experts: 2
        - Experts per token: 6
        - Expert evaluations: 72 per token

Enter prompt: Write a poem about AI.
> Artificial minds awakening, patterns flowing...
> Circuits singing ancient songs of logic...
```

---

## 📁 Project Structure

```
LiteMind/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── LICENSE                     # License information
│
├── include/                    # Public component interfaces
│   ├── Attention.hpp          # Multi-head attention layer
│   ├── Cli.hpp                # Command-line interface
│   ├── Config.hpp             # Model configuration
│   ├── CpuTensor.hpp          # CPU tensor operations
│   ├── DeepSeekRunner.hpp     # Full inference pipeline
│   ├── KvCache.hpp            # Key-value cache management
│   ├── Logger.hpp             # Logging utilities
│   ├── Model.hpp              # Model metadata holder
│   ├── MoeRouter.hpp          # Expert routing logic
│   ├── SafeTensor.hpp         # SafeTensors parser
│   ├── Sampler.hpp            # Token sampling strategies
│   ├── Tensor.hpp             # Tensor metadata
│   ├── Tokenizer.hpp          # Tokenization pipeline
│   └── WeightReader.hpp       # Weight loading from disk
│
├── src/                        # Implementation files
│   ├── main.cpp               # Application entry point
│   ├── Attention.cpp
│   ├── Cli.cpp
│   ├── Config.cpp
│   ├── CpuTensor.cpp
│   ├── DeepSeekRunner.cpp
│   ├── KvCache.cpp
│   ├── Logger.cpp
│   ├── Model.cpp
│   ├── MoeRouter.cpp
│   ├── SafeTensor.cpp
│   ├── Sampler.cpp
│   ├── Tensor.cpp
│   ├── Tokenizer.cpp
│   └── WeightReader.cpp
│
├── tests/                      # Unit test suite
│   ├── CMakeLists.txt
│   ├── AttentionTest.cpp      # Test attention layer
│   ├── CpuTensorTest.cpp      # Test tensor operations
│   ├── MoeRouterSamplerTest.cpp # Test routing & sampling
│   ├── TokenizerRoundTrip.cpp # Test tokenizer
│   └── WeightReaderTest.cpp   # Test weight loading
│
├── models/                     # Model directory (for runtime)
│   ├── config.json            # Model hyperparameters
│   ├── tokenizer.json         # Tokenizer vocabulary
│   ├── model-*.safetensors    # Model weights (4 shards)
│   └── generation_config.json # Generation parameters
│
├── docs/                       # Documentation (reserved)
├── examples/                   # Usage examples (reserved)
├── scripts/                    # Utility scripts (reserved)
├── third_party/               # External dependencies
├── tools/                      # Development tools (reserved)
└── build/                      # Build output (generated)
    ├── bin/                    # Compiled executables
    ├── CMakeFiles/
    └── ...
```

---

## 🔧 Component Reference

### Model Components

#### `Config` (include/Config.hpp)
Manages model hyperparameters and configuration metadata.

**Key Methods**:
- `attention_head_dim()`: Get attention head dimension
- `num_hidden_layers()`: Get number of transformer layers
- `num_routed_experts()`: Get number of routed experts
- `hidden_size()`: Get hidden layer dimension

#### `Model` (include/Model.hpp)
Owns model metadata including configuration, tensors, and file path.

**Key Methods**:
- `config()`: Access model configuration
- `tensors()`: Access loaded tensor metadata
- `path()`: Get model directory path
- `set_path()`: Set model directory

#### `Tokenizer` (include/Tokenizer.hpp)
Handles text-to-token and token-to-text conversion.

**Key Methods**:
- `encode()`: Convert text to token IDs
- `decode()`: Convert token IDs to text
- `vocab_size()`: Get vocabulary size

### Inference Components

#### `DeepSeekRunner` (include/DeepSeekRunner.hpp)
Full end-to-end inference pipeline for DeepSeek-V2.

**Key Methods**:
- `DeepSeekRunner(model_path, error)`: Load model weights
- `ready()`: Check if model loaded successfully
- `generate(tokenizer, tokens, max_new_tokens)`: Generate text

#### `Attention` (include/Attention.hpp)
Multi-head latent attention with RoPE positional encoding.

**Features**:
- 16 attention heads
- 128-dim + 64-dim latent space
- Grouped query attention
- YaRN RoPE scaling

#### `MoeRouter` (include/MoeRouter.hpp)
Mixture-of-Experts routing logic.

**Configuration**:
- 64 routed experts
- 2 shared experts
- Top-6 routing per token
- Softmax scoring

#### `Sampler` (include/Sampler.hpp)
Token sampling and selection strategies.

**Methods**:
- `sample_argmax()`: Greedy selection (highest probability)
- `sample_multinomial()`: Probabilistic sampling
- `sample_top_k()`: Top-k filtering
- `sample_top_p()`: Nucleus (top-p) filtering

### Data Structures

#### `Tensor` (include/Tensor.hpp)
Metadata-only tensor representation (no data ownership).

**Properties**:
- Shape (dimensions)
- Data type (float32, bfloat16, int32, etc.)
- Stride information

#### `CpuTensor` (include/CpuTensor.hpp)
CPU-resident tensor with owned data.

**Operations**:
- Element-wise operations
- Matrix multiplication (via OpenBLAS)
- Reshaping and transposition

#### `SafeTensor` (include/SafeTensor.hpp)
Parser for HuggingFace SafeTensors format.

**Features**:
- Multi-file shard support
- Lazy tensor loading
- Metadata validation

#### `KvCache` (include/KvCache.hpp)
Key-value cache for efficient generation.

**Optimizations**:
- Pre-allocated buffers
- Sliding window cache
- Memory-efficient updates

### Utilities

#### `Logger` (include/Logger.hpp)
Lightweight diagnostic logging.

**Levels**:
- DEBUG: Detailed internal state
- INFO: General information
- WARN: Warnings and recoverable errors
- ERROR: Critical failures

#### `Cli` (include/Cli.hpp)
Command-line interface and user interaction.

---

## 🔍 Model Configuration

The model configuration is stored in `models/config.json`:

```json
{
  "model_type": "deepseek_v2",
  "hidden_size": 2048,
  "num_hidden_layers": 27,
  "num_attention_heads": 16,
  "num_key_value_heads": 16,
  "num_experts_per_tok": 6,
  "n_routed_experts": 64,
  "n_shared_experts": 2,
  "vocab_size": 102400,
  "max_position_embeddings": 163840,
  "hidden_act": "silu",
  "intermediate_size": 10944,
  "moe_intermediate_size": 1408,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "torch_dtype": "bfloat16",
  "rope_theta": 10000,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "type": "yarn",
    "factor": 40,
    "original_max_position_embeddings": 4096
  },
  "attention_bias": false,
  "tie_word_embeddings": false
}
```

### Key Parameters Explained

- **hidden_size**: Dimension of each transformer layer (2048)
- **num_hidden_layers**: Number of stacked transformer layers (27)
- **num_attention_heads**: Number of attention heads (16)
- **num_experts_per_tok**: How many experts process each token (6)
- **n_routed_experts**: Total routed experts available (64)
- **n_shared_experts**: Shared experts used by all tokens (2)
- **vocab_size**: Token vocabulary size (102,400)
- **max_position_embeddings**: Maximum context length (163,840 tokens)
- **torch_dtype**: Precision (bfloat16 for efficiency)
- **rope_scaling**: Positional encoding with YaRN scaling method

---

## 🧪 Testing

### Run All Tests

```bash
cd build
ctest
```

### Run Specific Test

```bash
ctest -R TokenizerRoundTrip -V
```

### Test Details

| Test | Purpose |
|------|---------|
| `AttentionTest` | Validates attention computation correctness |
| `CpuTensorTest` | Tests tensor operations and memory handling |
| `MoeRouterSamplerTest` | Verifies expert routing and token sampling |
| `TokenizerRoundTrip` | Ensures text ↔ token conversion is reversible |
| `WeightReaderTest` | Validates SafeTensors file parsing |

### Adding New Tests

1. Create `tests/MyTest.cpp`
2. Add to `tests/CMakeLists.txt`:
   ```cmake
   add_test(MyTest tests/MyTest.cpp)
   ```
3. Rebuild and run

---

## 💡 Usage Examples

### Example 1: Basic Model Inspection

```cpp
#include "Model.hpp"
#include "WeightReader.hpp"

int main() {
    litemind::Model model;
    model.set_path("models");
    
    std::cout << "Model size: " << model.config().hidden_size() << "\n";
    std::cout << "Layers: " << model.config().num_hidden_layers() << "\n";
    std::cout << "Experts: " << model.config().n_routed_experts() << "\n";
}
```

### Example 2: Text Generation

```cpp
#include "DeepSeekRunner.hpp"
#include "Tokenizer.hpp"

int main() {
    std::string error;
    litemind::DeepSeekRunner runner("models", error);
    
    if (!runner.ready()) {
        std::cerr << "Failed to load: " << error << "\n";
        return 1;
    }
    
    litemind::Tokenizer tokenizer("models/tokenizer.json");
    auto tokens = tokenizer.encode("Write a poem:");
    
    std::string generated = runner.generate(tokenizer, tokens, 200);
    std::cout << "Generated: " << generated << "\n";
}
```

### Example 3: Custom Inference Loop

```cpp
#include "Attention.hpp"
#include "MoeRouter.hpp"

int main() {
    litemind::Attention attention;
    litemind::MoeRouter router;
    
    // Your custom inference logic here
    
    return 0;
}
```

---

## 🗺️ Future Roadmap

### Phase 2: Performance & Optimization
- [ ] CUDA kernel implementations for GPU acceleration
- [ ] Quantization support (INT8, FP8, Dynamic)
- [ ] Graph optimization and kernel fusion
- [ ] Batch inference support

### Phase 3: Features
- [ ] Multi-device inference (CPU + GPU)
- [ ] Fine-tuning pipeline
- [ ] LoRA (Low-Rank Adaptation) support
- [ ] Additional model support (LLaMA, Qwen, etc.)

### Phase 4: Integration & Tools
- [ ] HTTP REST API server
- [ ] WebSocket streaming support
- [ ] Python bindings (via pybind11)
- [ ] Performance benchmarking suite
- [ ] Model quantization tools

### Phase 5: Production Ready
- [ ] Comprehensive error handling and validation
- [ ] Distributed inference across multiple nodes
- [ ] Advanced caching strategies
- [ ] Monitoring and telemetry

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and add tests
4. **Ensure all tests pass**:
   ```bash
   cd build && ctest
   ```
5. **Commit with clear messages**
6. **Push and create a Pull Request**

### Guidelines

- Follow C++20 best practices
- Write unit tests for new features
- Update documentation for API changes
- Keep commits atomic and focused
- Run clang-format for consistent style

---

## 📄 License

This project is licensed under the [LICENSE](LICENSE) file. See the LICENSE file for details.

---

## 🔗 Resources

- [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434)
- [SafeTensors Format](https://huggingface.co/docs/safetensors/)
- [RoPE Positional Encoding](https://arxiv.org/abs/2104.09864)
- [YaRN Scaling](https://arxiv.org/abs/2309.00071)
- [C++20 Standard](https://en.cppreference.com/w/cpp/20)

---

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an [Issue](https://github.com/yourusername/LiteMind/issues)
- Start a [Discussion](https://github.com/yourusername/LiteMind/discussions)
- Email: your.email@example.com

---

**Last Updated**: 2026-08-18  
**Maintained by**: LiteMind Team
