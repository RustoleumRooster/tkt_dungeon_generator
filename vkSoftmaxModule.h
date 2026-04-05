#pragma once
#ifndef _VK_SOFTMAX_MODULE_H_
#define _VK_SOFTMAX_MODULE_H_

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

struct Softmax_Module : public Vulkan_Module
{
	// input_dimensions: [B x num_heads x M x N] = [16 x 4 x 64 x 64]
	TensorDimension input_dimensions{ 16, 4, 64, 64 };

	Softmax_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 B;
		u32 num_heads;
		u32 M;   // query positions (64)
		u32 N;   // key positions   (64)
	};

	pushconstant_struct pushconstants
	{
		16,  // B
		4,   // num_heads
		64,  // M
		64,  // N
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

	Pass fwd_pass;  // softmax.spv

	virtual void setDimensions() override;
	virtual void startup() override;
	virtual void forward() override;
	virtual void backward() override;
	virtual void cleanup_passes() override;

	void dispatch_forward(VkCommandBuffer);

	reflect::input<vkBufferResource>  input_tensor;   // [B x num_heads x M x N]
	reflect::output<vkBufferResource> output_tensor;  // [B x num_heads x M x N]
	reflect::input<vkBufferResource>  grad_input;
	reflect::output<vkBufferResource> grad_output;

	REFLECT_VKMOD()
};

#endif
