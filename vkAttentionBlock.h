#pragma once

#include "vkWorkblock.h"
#include "vkLayerNormModule.h"
#include "vkPosEmbedModule.h"
#include "vkProjectionQKVModule.h"
#include "vkAttnScoresModule.h"
#include "vkSoftmaxModule.h"

struct Attention_Block : public Workblock_Module
{
	virtual void build_workflow(std::vector<Vulkan_Module*>&) override;
	virtual reflect::input<vkBufferResource>&  head_input()       override;
	virtual reflect::output<vkBufferResource>& tail_output()      override;
	virtual reflect::input<vkBufferResource>&  gradient_input()   override;
	virtual reflect::output<vkBufferResource>& gradient_output()  override;

	PosEmbed_Module*        pos_embed   = NULL;
	LayerNorm_Module*       layer_norm  = NULL;
	Projection_QKV_Module*  qkv         = NULL;
	Attn_Scores_Module*     attn_scores = NULL;
	Softmax_Module*         softmax     = NULL;

	// TODO: value aggregation, output projection

	REFLECT_VKMOD()
};
