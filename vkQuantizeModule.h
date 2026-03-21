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

	virtual void setDimensions() override;
	void createDescriptorSets();
	void createDescriptorSetLayout();

	virtual void run();
	void execute();
	void cleanup();

	std::vector<f32> mapped_codebook;

	reflect::input<vkBufferResource> input_tensor;
	reflect::output<vkBufferResource> output_tensor;
	reflect::parameter<vkBufferResource> codebook;

	REFLECT_VKMOD()
};

#endif
