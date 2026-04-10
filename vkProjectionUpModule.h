#pragma once
#ifndef _VK_PROJECTION_UP_MODULE_H_
#define _VK_PROJECTION_UP_MODULE_H_

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

struct Projection_Up_Module : public Vulkan_Module
{
	// input_dimensions:  [B x seq_len x 1 x D]      e.g. [16 x 64 x 1 x 128]
	// output_dimensions: [B x seq_len x 1 x output_dim]
	TensorDimension input_dimensions{ 16, 64, 1, 128 };
	u32 output_dim = 512;  // 4 * D by default

	Projection_Up_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 M;  // seq_len       (64)
		u32 N;  // output_dim    (512)
		u32 K;  // input_dim     (128)
	};

	pushconstant_struct pushconstants
	{
		64,   // M
		512,  // N
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
	reflect::parameter<vkBufferResource> parameters;     // [K x N]
	reflect::input<vkBufferResource>     grad_input;     // [B x M x N]  (upstream dL/dY)
	reflect::output<vkBufferResource>    grad_output;    // [B x M x K]  (dL/dX)

	REFLECT_VKMOD()
};

#endif
