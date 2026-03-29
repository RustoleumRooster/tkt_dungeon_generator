#pragma once
#ifndef _VK_LAYERNORM_MODULE_H_
#define _VK_LAYERNORM_MODULE_H_

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

struct LayerNorm_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16, 64, 1, 128 };  // B, T, 1, D

	LayerNorm_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 B;
		u32 T;
		u32 D;
	};

	pushconstant_struct pushconstants
	{
		16,   // B
		64,   // T
		128,  // D
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

	Pass fwd_pass;    // layernorm.spv         — mean/invstd/normalize/scale
	Pass bwd_pass_x;  // layernorm_grad_x.spv  — dL/dx
	Pass bwd_pass_p;  // layernorm_grad_p.spv  — dγ/dβ

	virtual void setDimensions() override;
	virtual void startup() override;
	virtual void forward() override;
	virtual void backward() override;
	virtual void cleanup_passes() override;
	virtual void initialize_parameters() override;

	void dispatch_forward(VkCommandBuffer);
	void dispatch_backward_x(VkCommandBuffer);
	void dispatch_backward_p(VkCommandBuffer);

	reflect::input<vkBufferResource>     input_tensor;
	reflect::output<vkBufferResource>    invstd_buffer;
	reflect::output<vkBufferResource>    xnorm_buffer;
	reflect::output<vkBufferResource>    output_tensor;
	reflect::parameter<vkBufferResource> parameters;
	reflect::input<vkBufferResource>     grad_input;
	reflect::output<vkBufferResource>    grad_output;

	REFLECT_VKMOD()
};

#endif
