#pragma once

class Vulkan_App;

// Standalone finite-difference gradient check for Attention_Block.
//
// Builds its own workflow:
//   Create_Tensor (x) -> Attention_Block
//   Create_Tensor (R) -> Attention_Block gradient input   (dL/dout = R)
// with the fake loss L = sum(out * R), and no optimizer.
//
// Checks:
//   1. Identity path: with W_O = 0, dx must equal R exactly.
//   2. Central differences on x and on every parameter of the block
//      (PosEmbed, LayerNorm, W_qkv split into Q/K/V columns, W_O).
//
// Calls vulkan->cleanup() at the end, like Vulkan_Workflow::initialize_and_run.
// Returns true if every check passed.
bool run_attention_gradient_check(Vulkan_App* vulkan);
