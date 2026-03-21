#pragma once
#ifndef _VK_NN_UP_MODULE_H_
#define _VK_NN_UP_MODULE_H_

#include <irrlicht.h>
#include <vector>
#include "vkDevice.h"
#include "vkModules.h"
#include <vulkan/vulkan.h>
#include "vkDescriptors.h"
#include "vkBufferObject.h"
#include "reflect_custom_types.h"

class MyDescriptorPool;

struct NNUp_Module : public Vulkan_Module
{
	// Input dimensions (before upscale)
	TensorDimension input_dimensions{ 16,128,8,8 };

	NNUp_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 C;
		u32 H;          // output height
		u32 W;          // output width
		u32 n_elements;
	};

	pushconstant_struct pushconstants{ 128, 16, 16, 16*128*16*16 };

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
	Pass bwd_pass;

	void setDimensions();
	void forward();
	void backward();

	reflect::input<vkBufferResource>  input;
	reflect::output<vkBufferResource> output;
	reflect::input<vkBufferResource>  grad_input;
	reflect::output<vkBufferResource> grad_output;

	REFLECT_VKMOD()
};

#endif
