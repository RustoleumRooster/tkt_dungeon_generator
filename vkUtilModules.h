#pragma once
#ifndef _VK_UTIL_MODULES_H_
#define _VK_UTIL_MODULES_H_

#include <irrlicht.h>
#include <vector>
//#include "BufferManager.h"
#include <iterator>
#include "vkDevice.h"
//#include "vkModel.h"
#include "vkModules.h"
//#include "vk_BVH.h"
#include <vulkan/vulkan.h>
#include "vkDescriptors.h"
#include "vkBufferObject.h"
#include "vkComputePipeline.h"
#include "reflect_custom_types.h"

class MyDescriptorPool;

class Create_Tensor_Module : public Vulkan_Module
{
public:
	TensorDimension dimensions{ 1,1,1,1 };

	Create_Tensor_Module()
	{
		set_ptrs();
	}

	virtual void initialize(Vulkan_App* vulkan);

	void setDimensions();
	virtual void forward();
	void createImages(bool random_data = false);

	reflect::output<vkBufferResource> output_tensor;

	REFLECT_VKMOD()
};

struct Convolution_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,64,64 };
	TensorDimension output_dimensions{ 16,128,32,32 };

	Convolution_Module() 
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 k;
		u32 s;
		u32 p;
		u32 c_in;
		u32 c_out;
		u32 img_size_in;
		u32 img_size_out;
		u32 n;
	};

	pushconstant_struct pushconstants
	{
		4,  //k
		2,  //s
		1,  //p
		128,//c_in
		128,//c_out
		64, //img_size_in
		32, //img_size_out
		16, //n
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
	Pass bwd_pass_B;

	void setDimensions();
	void forward();
	void backward();
	void backward_A();
	void backward_B();

	reflect::input<vkBufferResource>     input_tensor;
	reflect::output<vkBufferResource>    output_tensor;
	reflect::parameter<vkBufferResource> weights;
	reflect::input<vkBufferResource>     grad_input;
	reflect::output<vkBufferResource>    grad_output;

	REFLECT_VKMOD()
};

struct Normalization_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,32,32 };

	Normalization_Module()
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

	pushconstant_struct pushconstants{ 16, 128, 32, 32 };

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

	void setDimensions();
	virtual void forward();

	reflect::input<vkBufferResource>  input_tensor;
	reflect::output<vkBufferResource> mean_buffer;
	reflect::output<vkBufferResource> var_buffer;

	REFLECT_VKMOD()
};


struct Activation_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,32,32 };

	Activation_Module()
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

	pushconstant_struct pushconstants{ 16, 128, 32, 32 };

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

	void setDimensions();
	void forward();

	reflect::input<vkBufferResource>     input_tensor;
	reflect::output<vkBufferResource>    output_tensor;
	reflect::input<vkBufferResource>     mean_buffer;
	reflect::input<vkBufferResource>     var_buffer;
	reflect::output<vkBufferResource> parameters;

	REFLECT_VKMOD()
};


#endif