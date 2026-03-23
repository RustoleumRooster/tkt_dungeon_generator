#pragma once
#ifndef _VK_QUANTIZE_MODULE_H_
#define _VK_QUANTIZE_MODULE_H_

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

struct Quantize_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,8,8 };
	TensorDimension codebook_size{ 1,1,512,128 };

	Quantize_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 n_vectors;
	};

	pushconstant_struct pushconstants
	{
		1 //n_vectors
	};

	struct commit_pushconstant_struct
	{
		u32 n_elements;
	};

	commit_pushconstant_struct commit_pushconstants
	{
		1 //n_elements
	};

	struct bwd_pushconstant_struct
	{
		u32   n_elements;  // B*H*W spatial positions (normalization factor for commitment loss)
		u32   n_channels;  // C
		float beta;        // commitment loss weight
	};

	bwd_pushconstant_struct bwd_pushconstants
	{
		1024, //n_elements
		128,  //n_channels
		0.25f //beta
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

	Pass fwd_pass;
	Pass commit_loss_pass;
	Pass bwd_pass;

	virtual void setDimensions() override;
	virtual void forward() override;
	virtual void backward() override;
	virtual void initialize_parameters() override;

	std::vector<f32> mapped_codebook;

	reflect::input<vkBufferResource>     input_tensor;
	reflect::output<vkBufferResource>    output_tensor;
	reflect::output<vkBufferResource>    indices_buffer;  // nearest codebook index per position, saved for backward
	reflect::output<vkBufferResource>    loss_buffer;     // per-position quantization distance (commitment loss)
	reflect::output<vkBufferResource>    commit_loss;     // scalar mean commitment loss
	reflect::parameter<vkBufferResource> codebook;

	reflect::input<vkBufferResource>     grad_input;      // dL/d_quantized_output from downstream
	reflect::output<vkBufferResource>    grad_output;     // dL/d_encoder_output (straight-through)

	REFLECT_VKMOD()
};

#endif
