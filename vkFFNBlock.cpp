#include <irrlicht.h>
#include "Reflection.h"
#include "vkModules.h"
#include "vkFFNBlock.h"

REFLECT_VKMOD_BEGIN(FFN_Block)
	ALIAS("FFN Block")
	INHERIT_FROM(Workblock_Module)
	// Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	// Backward Pass
	REFLECT_VKMOD_GRAD(input_grad)
	REFLECT_VKMOD_GRAD(output_grad)
REFLECT_VKMOD_END()

void FFN_Block::build_workflow(std::vector<Vulkan_Module*>& append_list)
{
	u32 B    = input_dimension.B;
	u32 T    = input_dimension.C;  // seq_len (64)
	u32 D    = input_dimension.W;  // embed_dim (128)
	u32 D4   = D * 4;              // expanded dim (512)

	// --- Proj Up: [B x T x D] * [D x 4D] = [B x T x 4D] ---
	proj_up = new Projection_Up_Module();
	proj_up->input_dimensions = { B, T, 1, D };
	proj_up->output_dim       = D4;
	proj_up->pushconstants    = { T, D4, D };

	// --- GELU: [B x T x 1 x 4D] → [B x T x 1 x 4D] ---
	gelu = new Gelu_Module();
	gelu->input_dimensions = { B, T, 1, D4 };
	gelu->pushconstants    = { B, T, 1, D4 };

	// --- Proj Down: [B x T x 4D] * [4D x D] = [B x T x D] ---
	proj_down = new Projection_Down_Module();
	proj_down->input_dimensions = { B, T, 1, D4 };
	proj_down->output_dim       = D;
	proj_down->pushconstants    = { T, D, D4 };

	modules.push_back(proj_up);
	modules.push_back(gelu);
	modules.push_back(proj_down);

	// Forward chain: proj_up → gelu → proj_down
	reflect::connect(&proj_up->output_tensor,   &gelu->input_tensor);
	reflect::connect(&gelu->output_tensor,      &proj_down->input_tensor);

	// Backward chain: proj_down → gelu → proj_up
	reflect::connect(&proj_down->grad_output,   &gelu->grad_input);
	reflect::connect(&gelu->grad_output,        &proj_up->grad_input);

	for (Vulkan_Module* mod : modules)
	{
		mod->depth        = this->depth + 1;
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>&  FFN_Block::head_input()      { return proj_up->input_tensor; }
reflect::output<vkBufferResource>& FFN_Block::tail_output()     { return proj_down->output_tensor; }
reflect::input<vkBufferResource>&  FFN_Block::gradient_input()  { return proj_down->grad_input; }
reflect::output<vkBufferResource>& FFN_Block::gradient_output() { return proj_up->grad_output; }
