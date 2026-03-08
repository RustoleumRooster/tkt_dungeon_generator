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

	virtual void run();
	void createImages(bool random_data = false);

	reflect::output<vkBufferResource> output_tensor;
	reflect::output<vkBufferResource> scratchpad;

	REFLECT_VKMOD()
};

struct Dummy_Consumer : public Vulkan_Module
{
	Dummy_Consumer()

	{
		set_ptrs();
	}

	//TensorDimension input_dimensions{ 1,4,4,4 };

	virtual void run() {}

	reflect::input<vkBufferResource> input_tensor;
	REFLECT_VKMOD()
};

struct Convolution_Module : public Vulkan_Module
{
	TensorDimension input_dimensions{ 16,16,32,32 };
	TensorDimension output_dimensions{ 16,16,16,16 };

	Convolution_Module()
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
		u32 h;
		u32 w;
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

	//std::vector<aligned_float> weights;
	//std::vector<aligned_float> activations;
	//std::vector<aligned_float> outputs;
	void createImages();
	void createDescriptorSets();
	void createDescriptorSetLayout();

	virtual void run();
	void execute();
	void read_results();
	void cleanup();
	/*
	void resize()
	{
		weights.resize(info.k * info.k * input_dimensions.C * output_dimensions.C);
		activations.resize(info.k * info.k * input_dimensions.C);
		outputs.resize(info.k * info.k * output_dimensions.C);
	}*/

	reflect::input<vkBufferResource> weights;
	reflect::input<vkBufferResource> input_tensor;
	reflect::output<vkBufferResource> output_tensor;
	reflect::input<vkBufferResource> scratchpad;

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