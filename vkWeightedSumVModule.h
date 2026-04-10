#pragma once
#ifndef _VK_WEIGHTED_SUM_V_MODULE_H_
#define _VK_WEIGHTED_SUM_V_MODULE_H_

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

struct WeightedSumV_Module : public Vulkan_Module
{
	// input_dimensions: QKV buffer [B, T, 1, 3D]
	TensorDimension input_dimensions{ 16, 64, 1, 384 };

	u32 num_heads = 4;

	WeightedSumV_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 M;          // seq_len (64)
		u32 N;          // seq_len (64)
		u32 K;          // head_dim = D / num_heads
		u32 D;          // full embed dim
		u32 num_heads;
		u32 head_n;     // set per-dispatch
	};

	pushconstant_struct pushconstants
	{
		64,   // M
		64,   // N
		32,   // K
		128,  // D
		4,    // num_heads
		0,    // head_n
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

	Pass fwd_pass;  // weighted_sum_V.spv — out[b, h, i, k] = sum_j scores[b, h, i, j] * V_h[j, k]

	virtual void setDimensions() override;
	virtual void startup() override;
	virtual void forward() override;
	virtual void backward() override;
	virtual void cleanup_passes() override;

	void dispatch_forward(VkCommandBuffer, u32 head_n);

	reflect::input<vkBufferResource>  qkv_tensor;     // QKV [B x M x 3D],              binding 0
	reflect::input<vkBufferResource>  scores_tensor;  // scores [B x num_heads x M x N], binding 1
	reflect::output<vkBufferResource> output_tensor;  // result [B x num_heads x M x K], binding 2

	reflect::input<vkBufferResource>  grad_input;     // incoming gradient [B x num_heads x M x K]
	reflect::output<vkBufferResource> grad_scores;    // gradient w.r.t. scores
	reflect::output<vkBufferResource> grad_qkv;       // gradient w.r.t. V slice of QKV

	REFLECT_VKMOD()
};

#endif
