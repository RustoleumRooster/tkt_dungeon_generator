#pragma once
#ifndef _VK_PROJECTION_O_MODULE_H_
#define _VK_PROJECTION_O_MODULE_H_

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

struct Projection_O_Module : public Vulkan_Module
{
	// input_dimensions: B=batch, C=seq_len(64), H=1, W=embedding_dim(128)
	// Expects concatenated head outputs laid out as [B x T x D]
	TensorDimension input_dimensions{ 16, 64, 1, 128 };

	Projection_O_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 M;  // seq_len   (64)
		u32 N;  // embed_dim (128)  — square projection, W_O is [D x D]
		u32 K;  // embed_dim (128)
	};

	struct pushconstant_backward_struct
	{
		u32 M;  // embed_dim
		u32 N;  // embed_dim 
		u32 K;  // seq_len
		u32 B;	// batch_size - needed so we can sum dL/dW over batches 
	};

	pushconstant_struct pushconstants
	{
		64,   // M
		128,  // N
		128,  // K
	};

	pushconstant_backward_struct pushconstants_b
	{
		128,   // M
		128,  // N
		64,  // K
		16	  //B
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

	Pass fwd_pass;   // matmul.spv — [B x M x D] * [D x D] = [B x M x D]
	Pass bwd_pass_1; // dL/dX
	Pass bwd_pass_2; // dL/dW

	virtual void setDimensions() override;
	virtual void startup() override;
	virtual void forward() override;
	virtual void backward() override;
	virtual void cleanup_passes() override;
	virtual void initialize_parameters() override;

	void dispatch_forward(VkCommandBuffer);
	void dispatch_backward_grad(VkCommandBuffer);
	void dispatch_backward_grad_w(VkCommandBuffer);

	reflect::input<vkBufferResource>     input_tensor;   // [B x M x D]  (concatenated heads)
	reflect::output<vkBufferResource>    output_tensor;  // [B x M x D]
	reflect::parameter<vkBufferResource> parameters;     // [D x D]       (W_O weight matrix)
	reflect::input<vkBufferResource>     grad_input;     // [B x M x D]   (upstream dL/dY)
	reflect::output<vkBufferResource>    grad_output;    // [B x M x D]   (dL/dX to propagate back)

	REFLECT_VKMOD()
};

#endif
