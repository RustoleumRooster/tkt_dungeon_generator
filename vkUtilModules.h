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
	TensorDimension dimensions{ 16,16,32,32 };

	Create_Tensor_Module()
	{
		set_ptrs();
	}

	virtual void initialize(Vulkan_App* vulkan);

	void setDimensions();
	virtual void run();
	void createImages(bool random_data = false);

	reflect::output<vkBufferResource> output_tensor;
	reflect::output<vkBufferResource> scratchpad;

	REFLECT_VKMOD()
};

struct Convolution_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,64,64 };
	TensorDimension output_dimensions{ 16,128,32,32 };
	TensorDimension weight_dimensions{ 128,128,4,4 };

	Convolution_Module() : weights(mapped_weights)
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 k;
		u32 s;
		u32 n;
		u32 c_in;
		u32 c_out;
		u32 img_size_in;
		u32 img_size_out;
	};

	pushconstant_struct pushconstants
	{
		4,//k
		2,//s
		16,//n
		16,//c_in
		16,//c_out
		32,//h
		32,//uw
	};

	bool use_one_to_one_shader = false;

	void setDimensions();
	void createImages();
	void createDescriptorSets();
	void createDescriptorSetLayout();

	virtual void run();
	void execute();
	void read_results();
	void cleanup();

	std::vector<f32> mapped_weights;

	reflect::parameter<vkBufferResource> weights;
	reflect::input<vkBufferResource> input_tensor;
	reflect::output<vkBufferResource> pass_output;
	reflect::input<vkBufferResource> scratchpad;

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
		u32 k;
		u32 s;
		u32 n;
		u32 c_in;
		u32 c_out;
		u32 img_size_in;
		u32 img_size_out;
	};

	pushconstant_struct pushconstants
	{
		4,//k
		2,//s
		16,//n
		16,//c_in
		16,//c_out
		32,//h
		32,//uw
	};

	void setDimensions();
	//void createImages();
	void createBuffer();
	void createDescriptorSets();
	void createDescriptorSetLayout();

	virtual void run();
	void execute();
	void read_results();
	void cleanup();

	reflect::input<vkBufferResource> input_tensor;
	reflect::output<vkBufferResource> mean_buffer;
	reflect::output<vkBufferResource> var_buffer;
	reflect::output<vkBufferResource> pass_output;
	reflect::input<vkBufferResource> scratchpad;

	std::vector<VkDescriptorSetLayoutBinding> bindings;

	REFLECT_VKMOD()
};


struct Activation_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,128,32,32 };

	Activation_Module() : parameters(mapped_parameters)
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 k;
		u32 s;
		u32 n;
		u32 c_in;
		u32 c_out;
		u32 img_size_in;
		u32 img_size_out;
	};

	pushconstant_struct pushconstants
	{
		4,//k
		2,//s
		16,//n
		128,//c_in
		128,//c_out
		32,//h
		32,//uw
	};

	void setDimensions();
	void createDescriptorSets();
	void createDescriptorSetLayout();

	virtual void run();
	void execute();
	void cleanup();

	std::vector<f32> mapped_parameters;

	reflect::input<vkBufferResource> input_tensor;
	reflect::output<vkBufferResource> pass_output;
	reflect::input<vkBufferResource> mean_buffer;
	reflect::input<vkBufferResource> var_buffer;
	reflect::parameter<vkBufferResource> parameters;
	reflect::input<vkBufferResource> results_buffer;

	std::vector<VkDescriptorSetLayoutBinding> bindings;

	REFLECT_VKMOD()
};


/*
class Load_Textures_Module : public Vulkan_Module
{
public:
	Load_Textures_Module()
	{
		set_ptrs();
	}

	virtual void initialize(Vulkan_App* vulkan);

	virtual void run() override;
	void createImages();
	void destroyImages();

	std::vector<VkImage> lightmapImages;
	std::vector<VkDeviceMemory> lightmapsMemory;
	std::vector<VkImageView> lightmapImageViews;

	std::vector<video::ITexture*> textures;
	video::IVideoDriver* driver;
};

class Download_Textures_Module : public Vulkan_Module
{
public:
	Download_Textures_Module()
	{
		set_ptrs();
	}

	virtual void initialize(Vulkan_App* vulkan);

	virtual void run() override;

	reflect::input<vkMultiImageResource> images_in;

	std::vector<video::ITexture*> textures;

	video::IVideoDriver* driver;

	bool bFlip = false;

	REFLECT_VKMOD()
};

class Download_TextureArray_Module : public Vulkan_Module
{
public:

	Download_TextureArray_Module()
	{
		set_ptrs();
	}

	virtual void initialize(Vulkan_App* vulkan);

	virtual void run() override;

	reflect::input<vkImageArrayResource> images_in;

	std::vector<video::ITexture*> textures;

	video::IVideoDriver* driver;
	Lightmap_Configuration* configuration;

	bool bFlip = false;

	REFLECT_VKMOD()
};
*/

#endif