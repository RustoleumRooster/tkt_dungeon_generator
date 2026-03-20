#pragma once
#ifndef _VK_SIGMOID_MODULE_H_
#define _VK_SIGMOID_MODULE_H_

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

struct Sigmoid_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,8,8 };

	Sigmoid_Module()
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

	reflect::input<vkBufferResource>  input_tensor;
	reflect::output<vkBufferResource> output_tensor;
	reflect::input<vkBufferResource>  grad_input;
	reflect::output<vkBufferResource> grad_output;

	REFLECT_VKMOD()
};


#endif
