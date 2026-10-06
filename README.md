# Dungeon Generator

A roguelike dungeon generator that's also a hands-on project for learning how neural networks work under the hood.

There are two goals:

1. **Generate roguelike dungeons.** Classic ASCII dungeons (`#` walls, `.` floors) on a 64×64 grid, eventually produced by a learned model rather than hand-written rules.
2. **Learn AI programming from first principles.** There's no PyTorch and no TensorFlow. Every layer, forward pass, gradient and optimizer step is written by hand as a Vulkan compute shader and wired together in C++. The point is to understand what a framework would otherwise hide.

## How it works

### Procedural baseline

`dungeon.cpp` contains a hand-written generator that serves as the baseline and as a source of training data. It uses a random walker:

- It places a starting room near the center of the map.
- For 300 iterations, the walker either teleports to a random open cell, tries to carve a room next to itself (each room keeps a one-cell wall border), or digs a corridor forward, sometimes turning first.

The map is drawn as text in an Irrlicht GUI panel. Click the map to generate a new dungeon, or drag it to pan.

### Neural model (in progress)

The learned model has two stages. This is a common recipe for generating images and levels:

```
64×64 map ──► VQ-VAE encoder ──► 8×8 grid of codebook tokens ──► Transformer ──► next-token distribution
                (conv + GroupNorm + GELU,      (512-entry codebook,          (attention + FFN,
                 three 2× downsamples)          128-dim)                      vocab = 512)
```

- **VQ-VAE** (`make_default_workflow` in `vkWorkflow.cpp`) compresses a map into a small grid of discrete tokens and decodes it back. The decoder is a ResBlock followed by nearest-neighbor upscale blocks and a sigmoid output, trained against the original map with binary cross-entropy loss. The codebook is updated with EMA and uses a commitment loss.
- **Transformer** (`make_transformer_workflow`) takes the quantized 8×8 grid as a sequence of 64 tokens. It transposes the grid from channel-first to token-first layout, adds positional embeddings, then runs LayerNorm, multi-head self-attention and a feed-forward block. A vocab projection and a softmax predict codebook tokens. Sampling those tokens and decoding them with the VQ-VAE should eventually produce new dungeons.

## Architecture

### The Vulkan compute framework

The core of the project is a small deep-learning framework made of three layers:

| Layer | What it is | Examples |
|---|---|---|
| **Shaders** (`*.comp`) | GLSL compute kernels, one for each forward and backward op | `conv64.comp`, `layernorm.comp`, `attn_scores.comp`, `softmax.comp`, `adam.comp`, `*_grad*.comp` |
| **Modules** (`vk*Module.*`) | C++ wrappers that own a pipeline, declare inputs/outputs/parameters, and dispatch the forward and backward shaders | `Convolution_Module`, `Quantize_Module`, `LayerNorm_Module`, `Attn_Scores_Module`, `Vocab_Softmax_Module` |
| **Workblocks** (`vkWorkblock.*`, `vkAttentionBlock.*`, `vkFFNBlock.*`) | Composite modules that build and wire their own sub-module graph | `Convolution_Gelu_Block`, `ResBlock_Module`, `UpscaleBlock_Module`, `Attention_Block`, `FFN_Block` |

A `Vulkan_Workflow` puts modules together into a network:

1. **Build.** Modules and workblocks are created and their dimensions are set.
2. **Connect.** `reflect::connect(output, input)` wires tensors together. The backward pass is wired separately, from gradient outputs to gradient inputs.
3. **Plan memory.** Every buffer is tagged as a *param*, *feature*, *grad* or *other*. `plan_memory()` lays them out in shared GPU allocations, and all learnable parameters go into one contiguous buffer so a single Adam dispatch can update them.
4. **Run.** Modules run `forward()` in order, then `backward()` in reverse. The optimizer is registered first so that it runs last in the backward pass, after every gradient has accumulated.

### Reflection

`Reflection.h` provides a small runtime reflection system. Each module declares its `reflect::input<>`, `reflect::output<>` and `reflect::parameter<>` members. The framework uses those declarations to:

- connect modules
- discover buffers during memory planning
- warn about parameters that were never initialized

### GUI

The UI is built on the [Irrlicht](https://irrlicht.sourceforge.io/) engine. Text is rendered with [AGG](https://agg.sourceforge.net/antigrain.com/) TrueType fonts (`fonts.*`). Panels, scrollbars, a custom skin and buttons come from `CGUI*`, `GUI_tools.*`, `CameraPanel*` and `edit_*`.

## Status

This is a learning project, so it's very much a work in progress.

- ✅ Procedural random-walker dungeon generator with an interactive viewer
- ✅ VQ-VAE forward pass, with backward passes and gradient shaders for its modules (conv, GroupNorm, GELU/SiLU/sigmoid, NN upscale, quantize, BCE)
- ✅ Adam optimizer on a single unified parameter buffer
- ✅ Transformer forward-pass modules: positional embedding, LayerNorm, QKV/O projections, attention scores, softmax, weighted sum of V, FFN (up/down projections), vocab projection and softmax
- 🚧 Transformer loss module and end-to-end backward pass. The backward loop is currently disabled in `initialize_and_run`.
- 🚧 Training on generated dungeons and sampling new ones from the model

## Building

The project is Windows-only and builds with Visual Studio 2022 (`dungeon_generator.sln`, x64).

**Dependencies**

- [Vulkan SDK](https://vulkan.lunarg.com/) 1.3.268.0
- [Irrlicht](https://irrlicht.sourceforge.io/). `Irrlicht.dll` is included in the repo root.
- [AGG 2.6](https://agg.sourceforge.net/antigrain.com/), including `font_win32_tt`

> **Note:** `dungeon_generator.vcxproj` currently hard-codes absolute paths to these libraries. Change the include and library directories, and the AGG source file paths, to match your machine.

**Shaders**

Compile the GLSL compute shaders to SPIR-V before running:

```bat
shader_compile.bat
```

This calls `glslc` from the Vulkan SDK and writes the `.spv` files to `shaders/`.

**Run**

Build and run from Visual Studio. The app opens the dungeon viewer and runs one pass of the current network workflow, printing per-pass timing to the console.
