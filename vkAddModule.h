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

struct Add_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,8,8 };

	Add_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 n_elements;
	};

	pushconstant_struct pushconstants{ 16*128*8*8 };

	void setDimensions();
	void createDescriptorSets();
	void createDescriptorSetLayout();

	virtual void run();
	void execute();
	void cleanup();

	reflect::input<vkBufferResource>  input_a;
	reflect::input<vkBufferResource>  input_b;
	reflect::output<vkBufferResource> output;

	std::vector<VkDescriptorSetLayoutBinding> bindings;

	REFLECT_VKMOD()
};

#endif
