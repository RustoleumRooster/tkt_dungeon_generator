#pragma once
#ifndef _VK_ATTN_SCORES_MODULE_H_
#define _VK_ATTN_SCORES_MODULE_H_

#include <irrlicht.h>
#include <vector>
#include "vkDevice.h"
#include "vkModules.h"
#include <vulkan/vulkan.h>
#include "vkDescriptors.h"
#include "vkBufferObject.h"
#include "vkComputePipeline.h"
#include "reflect_custom_types.h"

class MyDescriptorPool;

struct Attn_Scores_Module : public Vulkan_Module
{
	// input_dimensions: QKV buffer [B, T, 1, 3D]
	TensorDimension input_dimensions{ 16, 64, 1, 384 };

	u32 num_heads = 4;

	Attn_Scores_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 M;         // seq_len (64)
		u32 N;         // seq_len (64)
		u32 K;         // head_dim = D / num_heads (32)
		u32 D;         // full embed dim (128)
		u32 num_heads; // 4
		u32 head_n;    // set per-dispatch
	};

	pushconstant_struct pushconstants
	{
		64,   // M
		64,   // N
		32,   // K (head_dim)
		128,  // D
		4,    // num_heads
		0,    // head_n (set in forward loop)
	};

	struct Pass
	{
		VkPipelineLayout                          pipelineLayout;
		MyDescriptorSetLayout*                    descriptorSetLayout = NULL;
		ComputePipeline*                          pipeline            = NULL;
		std::vector<VkDescriptorSet>              descriptorSets;
		std::vector<VkDescriptorSetLayoutBinding> bindings;

		void createPipeline(MyDevice*, const char* spv, VkPushConstantRange);
		void cleanup(VkDevice);
	};

	Pass fwd_pass;  // attn_scores.spv — scores[b, h, i, j] = Q_h @ K_h^T / sqrt(head_dim)

	virtual void setDimensions() override;
	virtual void startup() override;
	virtual void forward() override;
	virtual void backward() override;
	virtual void cleanup_passes() override;

	void dispatch_forward(VkCommandBuffer, u32 head_n);

	reflect::input<vkBufferResource>  input_tensor;   // QKV [B x M x 3D]
	reflect::output<vkBufferResource> output_tensor;  // scores [B x num_heads x M x N]
	reflect::input<vkBufferResource>  grad_input;
	reflect::output<vkBufferResource> grad_output;

	REFLECT_VKMOD()
};

#endif
