#pragma once
#ifndef _VK_ADD_MODULE_H_
#define _VK_ADD_MODULE_H_

#include <irrlicht.h>
#include <vector>
#include "vkDevice.h"
#include "vkModules.h"
#include <vulkan/vulkan.h>
#include "vkDescriptors.h"
#include "vkBufferObject.h"
#include "reflect_custom_types.h"

class MyDescriptorPool;

struct Skip_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,8,8 };

	Skip_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 n_elements;
	};

	pushconstant_struct pushconstants{ 16*128*8*8 };

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

	reflect::input<vkBufferResource>  input_a;
	reflect::input<vkBufferResource>  input_b;
	reflect::output<vkBufferResource> output;
	reflect::input<vkBufferResource>  grad_input;
	reflect::output<vkBufferResource> grad_output_a;
	reflect::output<vkBufferResource> grad_output_b;

	REFLECT_VKMOD()
};

#endif
