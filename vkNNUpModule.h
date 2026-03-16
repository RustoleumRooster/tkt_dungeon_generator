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

	void setDimensions();
	void createDescriptorSets();
	void createDescriptorSetLayout();

	virtual void run();
	void execute();
	void cleanup();

	reflect::input<vkBufferResource>  input;
	reflect::output<vkBufferResource> output;

	std::vector<VkDescriptorSetLayoutBinding> bindings;

	REFLECT_VKMOD()
};

#endif
