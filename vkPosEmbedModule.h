#pragma once
#ifndef _VK_POSEMBED_MODULE_H_
#define _VK_POSEMBED_MODULE_H_

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

struct PosEmbed_Module : public Vulkan_Module
{
	// input_dimensions: B=batch, C=rows, H=cols, W=embedding_dim
	TensorDimension input_dimensions{ 16, 8, 8, 128 };

	PosEmbed_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 B;  // batch
		u32 R;  // rows
		u32 C;  // cols
		u32 D;  // embedding dim
	};

	pushconstant_struct pushconstants
	{
		16,   // B
		8,    // R
		8,    // C
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

	Pass fwd_pass;    // pos_embed.spv         — output = input + row_param[r] + col_param[c]
	Pass bwd_pass_x;  // pos_embed_grad_x.spv  — pass-through input gradient
	Pass bwd_pass_p;  // pos_embed_grad_p.spv  — row/col param gradients

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
	reflect::output<vkBufferResource>    output_tensor;
	reflect::parameter<vkBufferResource> parameters;   // [(R+C)*D] row then col embeddings
	reflect::input<vkBufferResource>     grad_input;
	reflect::output<vkBufferResource>    grad_output;

	REFLECT_VKMOD()
};

#endif
