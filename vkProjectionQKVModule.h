#pragma once
#ifndef _VK_PROJECTION_QKV_MODULE_H_
#define _VK_PROJECTION_QKV_MODULE_H_

#include <irrlicht.h>
#include <vector>
#include <iterator>
#include "vkDevice.h"
#include "vkModules.h"
#include <vulkan/vulkan.h>
#include "vkDescriptors.h"
#include "vkBufferObject.h"
#include "vkComputePipeline.h"
#include "reflect_custom_types.h"

class MyDescriptorPool;

struct Projection_QKV_Module : public Vulkan_Module
{
	// input_dimensions: B=batch, C=seq_len(64), H=1, W=embedding_dim(128)
	TensorDimension input_dimensions{ 16, 64, 1, 128 };

	Projection_QKV_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 M;  // seq_len  (64)
		u32 N;  // 3*D      (384)
		u32 K;  // embed_dim(128)
	};

	pushconstant_struct pushconstants
	{
		64,   // M
		384,  // N
		128,  // K
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

	Pass fwd_pass;  // matmul.spv — [B x M x K] * [K x N] = [B x M x N]

	virtual void setDimensions() override;
	virtual void startup() override;
	virtual void forward() override;
	virtual void backward() override;
	virtual void cleanup_passes() override;
	virtual void initialize_parameters() override;

	void dispatch_forward(VkCommandBuffer);

	reflect::input<vkBufferResource>     input_tensor;   // [B x M x K]
	reflect::output<vkBufferResource>    output_tensor;  // [B x M x N]
	reflect::parameter<vkBufferResource> parameters;     // [K x N]  (weight matrix)
	reflect::input<vkBufferResource>     grad_input;     // [B x M x N]  (upstream dL/dY)
	reflect::output<vkBufferResource>    grad_output;    // [B x M x K]  (dL/dX to propagate back)

	REFLECT_VKMOD()
};

#endif
