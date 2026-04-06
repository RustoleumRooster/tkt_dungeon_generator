#include <irrlicht.h>
#include "Reflection.h"
#include "vkModules.h"
#include "vkAttentionBlock.h"

REFLECT_VKMOD_BEGIN(Attention_Block)
	ALIAS("Attention Block")
	INHERIT_FROM(Workblock_Module)
	// Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	// Backward Pass
	REFLECT_VKMOD_GRAD(input_grad)
	REFLECT_VKMOD_GRAD(output_grad)
REFLECT_VKMOD_END()

void Attention_Block::build_workflow(std::vector<Vulkan_Module*>& append_list)
{
	u32 B = input_dimension.B;
	u32 R = input_dimension.C;  // rows  (8)
	u32 C = input_dimension.H;  // cols  (8)
	u32 D = input_dimension.W;  // embed (128)
	u32 T = R * C;              // seq_len (64)

	// --- PosEmbed: [B x R x C x D] → [B x R x C x D] ---
	pos_embed = new PosEmbed_Module();
	pos_embed->input_dimensions = { B, R, C, D };

	// --- LayerNorm: [B x T x 1 x D] → [B x T x 1 x D] ---
	layer_norm = new LayerNorm_Module();
	layer_norm->input_dimensions = { B, T, 1, D };
	layer_norm->pushconstants    = { B, T, D };

	// --- QKV Projection: [B x T x 1 x D] → [B x T x 1 x 3D] ---
	qkv = new Projection_QKV_Module();
	qkv->input_dimensions = { B, T, 1, D };
	qkv->pushconstants    = { T, D * 3, D };

	// --- Attention Scores: [B x T x 1 x 3D] → [B x num_heads x T x T] ---
	u32 num_heads = 4;
	u32 head_dim  = D / num_heads;
	attn_scores = new Attn_Scores_Module();
	attn_scores->input_dimensions = { B, T, 1, D * 3 };
	attn_scores->num_heads        = num_heads;
	attn_scores->pushconstants    = { T, T, head_dim, D, num_heads, 0 };

	// --- Softmax: [B x num_heads x T x T] → [B x num_heads x T x T] ---
	softmax = new Softmax_Module();
	softmax->input_dimensions = { B, num_heads, T, T };
	softmax->pushconstants    = { B, num_heads, T, T };

	modules.push_back(pos_embed);
	modules.push_back(layer_norm);
	modules.push_back(qkv);
	modules.push_back(attn_scores);
	modules.push_back(softmax);

	// Forward chain: input → pos_embed → layer_norm → qkv → attn_scores → softmax
	// (pos_embed output [B,R,C,D] and layer_norm input [B,T,1,D] are the
	//  same byte size — T=R*C — so the buffer is safely reinterpreted)
	reflect::connect(&pos_embed->output_tensor,   &layer_norm->input_tensor);
	reflect::connect(&layer_norm->output_tensor,  &qkv->input_tensor);
	reflect::connect(&qkv->output_tensor,         &attn_scores->input_tensor);
	reflect::connect(&attn_scores->output_tensor, &softmax->input_tensor);

	// Backward chain (incomplete — QKV backward is a stub)
	reflect::connect(&softmax->grad_output,        &attn_scores->grad_input);
	reflect::connect(&attn_scores->grad_output,    &qkv->grad_input);
	reflect::connect(&qkv->grad_output,            &layer_norm->grad_input);
	reflect::connect(&layer_norm->grad_output,     &pos_embed->grad_input);

	for (Vulkan_Module* mod : modules)
	{
		mod->depth        = this->depth + 1;
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>&  Attention_Block::head_input()      { return pos_embed->input_tensor; }
reflect::output<vkBufferResource>& Attention_Block::tail_output()     { return softmax->output_tensor; }
reflect::input<vkBufferResource>&  Attention_Block::gradient_input()  { return softmax->grad_input; }
reflect::output<vkBufferResource>& Attention_Block::gradient_output() { return pos_embed->grad_output; }
