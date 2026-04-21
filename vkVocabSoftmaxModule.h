#pragma once
#ifndef _VK_VOCAB_SOFTMAX_MODULE_H_
#define _VK_VOCAB_SOFTMAX_MODULE_H_

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

struct Vocab_Softmax_Module : public Vulkan_Module
{
	// input_dimensions: [B x 1 x seq_len x vocab_size]  e.g. [16 x 1 x 64 x 512]
	TensorDimension input_dimensions{ 16, 1, 64, 512 };

	Vocab_Softmax_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 B;
		u32 num_heads;  // 1 for vocab softmax
		u32 M;          // seq_len    (64)
		u32 N;          // vocab_size (512)
	};

	pushconstant_struct pushconstants{ 16, 1, 64, 512 };

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

	Pass fwd_pass;  // softmax_vocab.spv

	virtual void setDimensions() override;
	virtual void startup() override;
	virtual void forward() override;
	virtual void backward() override;
	virtual void cleanup_passes() override;

	reflect::input<vkBufferResource>  input_tensor;   // [B x 1 x M x N]
	reflect::output<vkBufferResource> output_tensor;  // [B x 1 x M x N]
	reflect::input<vkBufferResource>  grad_input;
	reflect::output<vkBufferResource> grad_output;

	REFLECT_VKMOD()
};

#endif
