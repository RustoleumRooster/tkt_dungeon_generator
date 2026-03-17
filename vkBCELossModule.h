#pragma once
#ifndef _VK_BCE_LOSS_MODULE_H_
#define _VK_BCE_LOSS_MODULE_H_

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

struct BCE_Loss_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,1,64,64 };

	BCE_Loss_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 n_elements;
	};

	pushconstant_struct pushconstants{ 16 * 64 * 64 };

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
	virtual void run();

	reflect::input<vkBufferResource>  predictions;
	reflect::input<vkBufferResource>  ground_truth;
	reflect::output<vkBufferResource> loss;
	reflect::output<vkBufferResource> gradient_out;

	REFLECT_VKMOD()
};

#endif
