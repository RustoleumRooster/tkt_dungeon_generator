#pragma once
#ifndef _VK_SILU_MODULE_H_
#define _VK_SILU_MODULE_H_

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

struct Silu_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,8,8 };

	Silu_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 n;
		u32 c;
		u32 h;
		u32 w;
	};

	pushconstant_struct pushconstants
	{
		16,  //n
		128, //c
		8,   //h
		8,   //w
	};

	void setDimensions();
	void createDescriptorSets();
	void createDescriptorSetLayout();

	virtual void run();
	void execute();
	void cleanup();

	reflect::input<vkBufferResource> input_tensor;
	reflect::output<vkBufferResource> output_tensor;

	std::vector<VkDescriptorSetLayoutBinding> bindings;

	REFLECT_VKMOD()
};

#endif
