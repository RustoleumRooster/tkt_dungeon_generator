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

	// --- Weighted Sum V: (QKV [B x T x 1 x 3D], scores [B x num_heads x T x T]) → [B x num_heads x T x head_dim] ---
	weighted_sum_v = new WeightedSumV_Module();
	weighted_sum_v->input_dimensions = { B, T, 1, D * 3 };
	weighted_sum_v->num_heads        = num_heads;
	weighted_sum_v->pushconstants    = { T, T, head_dim, D, num_heads, 0 };

	// --- Output Projection: [B x T x D] * [D x D] → [B x T x D] ---
	// Input is the weighted_sum_v output reinterpreted as [B x T x D]
	// (num_heads * head_dim == D, same byte count)
	projection_o = new Projection_O_Module();
	projection_o->input_dimensions = { B, T, 1, D };
	projection_o->pushconstants    = { T, D, D };

	// --- Skip: x + projection_o(attention(norm(pos_embed(x)))) ---
	// skip->input_a = raw x (connected externally via connect_input)
	// skip->input_b = projection_o output
	skip = new Skip_Module();
	skip->input_dimensions = { B, T, 1, D };

	// add_grad sums the two backward paths: identity (skip->grad_output_a) and
	// attention (pos_embed->grad_output). Inserted at index 0 so it runs last in backward.
	add_grad = new Add_Grad_Module();
	add_grad->input_dimensions = { B, T, 1, D };

	modules.push_back(add_grad);
	modules.push_back(pos_embed);
	modules.push_back(layer_norm);
	modules.push_back(qkv);
	modules.push_back(attn_scores);
	modules.push_back(softmax);
	modules.push_back(weighted_sum_v);
	modules.push_back(projection_o);
	modules.push_back(skip);

	// Forward chain: input → pos_embed → layer_norm → qkv → attn_scores → softmax → weighted_sum_v → projection_o → skip
	// raw input also fans out to skip->input_a via connect_input
	// (pos_embed output [B,R,C,D] and layer_norm input [B,T,1,D] are the
	//  same byte size — T=R*C — so the buffer is safely reinterpreted)
	reflect::connect(&pos_embed->output_tensor,        &layer_norm->input_tensor);
	reflect::connect(&layer_norm->output_tensor,       &qkv->input_tensor);
	reflect::connect(&qkv->output_tensor,              &attn_scores->input_tensor);
	reflect::connect(&attn_scores->output_tensor,      &softmax->input_tensor);
	reflect::connect(&qkv->output_tensor,              &weighted_sum_v->qkv_tensor);
	reflect::connect(&softmax->output_tensor,          &weighted_sum_v->scores_tensor);
	reflect::connect(&weighted_sum_v->output_tensor,   &projection_o->input_tensor);
	reflect::connect(&projection_o->output_tensor,     &skip->input_b);

	// Backward chain (stubs — most backward passes not yet implemented)
	// skip->grad_output_b: gradient through the attention path
	// skip->grad_output_a: gradient through the identity path
	// add_grad merges both into the upstream gradient
	reflect::connect(&skip->grad_output_b,          &projection_o->grad_input);
	reflect::connect(&projection_o->grad_output,    &weighted_sum_v->grad_input);
	reflect::connect(&weighted_sum_v->grad_scores,  &softmax->grad_input);
	reflect::connect(&softmax->grad_output,         &attn_scores->grad_input);
	reflect::connect(&attn_scores->grad_output,     &qkv->grad_input);
	reflect::connect(&qkv->grad_output,             &layer_norm->grad_input);
	reflect::connect(&layer_norm->grad_output,      &pos_embed->grad_input);

	reflect::connect(&skip->grad_output_a,          &add_grad->grad_input_a);
	reflect::connect(&pos_embed->grad_output,       &add_grad->grad_input_b);

	for (Vulkan_Module* mod : modules)
	{
		mod->depth        = this->depth + 1;
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>&  Attention_Block::head_input()      { return pos_embed->input_tensor; }
reflect::output<vkBufferResource>& Attention_Block::tail_output()     { return skip->output; }
reflect::input<vkBufferResource>&  Attention_Block::gradient_input()  { return skip->grad_input; }
reflect::output<vkBufferResource>& Attention_Block::gradient_output() { return add_grad->grad_output; }

void Attention_Block::connect_input(reflect::output<vkBufferResource>* out)
{
	reflect::connect(out, &pos_embed->input_tensor);
	reflect::connect(out, &skip->input_a);
}
