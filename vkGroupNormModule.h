#pragma once
#ifndef _VK_GROUPNORM_MODULE_H_
#define _VK_GROUPNORM_MODULE_H_

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

struct GroupNorm_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,8,8 };

	GroupNorm_Module() 
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

	Pass fwd_pass_A;  // groupnorm.spv       — computes mean/invstd
	Pass fwd_pass_B;  // groupnorm2.spv      — applies gamma/beta, writes xnorm
	Pass bwd_pass_1;  // groupnorm_grad_1.spv — dβ gradient
	Pass bwd_pass_2;  // groupnorm_grad_2.spv — intermediate sums (sum_dy, sum_dy_xnorm)
	Pass bwd_pass_4;  // groupnorm_grad_4.spv — dL/dx

	void setDimensions();
	void forward_A();
	void forward_B();
	void backward_1();
	void backward_2();
	void backward_4();
	void backward();
	virtual void run();

	std::vector<f32> mapped_parameters;

	reflect::input<vkBufferResource>     input_tensor;
	reflect::output<vkBufferResource>    mean_buffer;
	reflect::output<vkBufferResource>    invstd_buffer;
	reflect::output<vkBufferResource>    output_tensor;
	reflect::output<vkBufferResource>    xnorm_buffer;
	reflect::output<vkBufferResource>    int_sums_buffer;
	reflect::parameter<vkBufferResource> parameters;
	reflect::input<vkBufferResource>     grad_input;
	reflect::output<vkBufferResource>    grad_output;
	reflect::output<vkBufferResource>    param_grad;

	REFLECT_VKMOD()
};

#endif
