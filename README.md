# LiteMind

**DeepSeek-V2 mixture-of-experts inference on the CPU, in C++20, with no
third-party libraries.**

A 15.7-billion-parameter model normally needs 32–80 GB of GPU memory. LiteMind
runs one on a laptop with no GPU at all.

It works because DeepSeek-V2-Lite is a *mixture of experts*. Each of its 26 MoE
layers holds 64 independent feed-forward experts, and a small router picks 6 of
them per token — so roughly a tenth of the expert weights decide any single
token. LiteMind memory-maps the 29.3 GiB checkpoint on the SSD and lets that
routing decision drive what is paged into RAM. A token pays for six experts,
not for the model.

| | |
|---|---|
| Parameters | 15.7 B in the checkpoint, 2.45 B active per token (15.6%) |
| Checkpoint | 29.3 GiB, memory-mapped — load time 0.2 s, because nothing is copied |
| Always resident | 2.44 GiB; the other 26.81 GiB streams as the router asks |
| Requires | A 64-bit CPU and an SSD. No GPU, no BLAS, no vcpkg, no Python |

## Documentation

### **https://ragavanperarasu.github.io/LiteMind/**

Setup and prerequisites, the architecture one page at a time, the measured
numbers, the command line, and troubleshooting. The site is built from
[`docs/`](docs/README.md), which reads just as well here on GitHub.

## License

MIT. See [`LICENSE`](LICENSE). The model weights are
[deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
and carry their own licence.
