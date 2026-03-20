#pragma once
#ifndef _VK_SKIP_MODULE_H_
#define _VK_SKIP_MODULE_H_

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

	void setDimensions();

	virtual void run();

	reflect::input<vkBufferResource>  input_tensor;
	reflect::output<vkBufferResource> output_tensor;
	reflect::output<vkBufferResource> skip_output;

	REFLECT_VKMOD()
};

#endif
