#pragma once
#ifndef _VK_GROUPNORM2_MODULE_H_
#define _VK_GROUPNORM2_MODULE_H_

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

struct GroupNorm2_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,8,8 };

	GroupNorm2_Module() : parameters(mapped_parameters)
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 n;
		u32 c;
		u32 h;
		u32 w;
		u32 num_groups;
	};

	pushconstant_struct pushconstants
	{
		16,  //n
		128, //c
		8,   //h
		8,   //w
		32,  //num_groups
	};

	void setDimensions();
	void createDescriptorSets();
	void createDescriptorSetLayout();

	virtual void run();
	void execute();
	void cleanup();

	std::vector<f32> mapped_parameters;

	reflect::input<vkBufferResource> input_tensor;
	reflect::parameter<vkBufferResource> parameters;
	reflect::input<vkBufferResource> mean_buffer;
	reflect::input<vkBufferResource> var_buffer;
	reflect::output<vkBufferResource> pass_output;
	reflect::input<vkBufferResource> scratchpad;

	std::vector<VkDescriptorSetLayoutBinding> bindings;

	REFLECT_VKMOD()
};

#endif
