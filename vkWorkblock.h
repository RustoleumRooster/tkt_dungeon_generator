#pragma once

#include "Reflection.h"
#include <irrlicht.h>
#include "global.h"
#include "vkModules.h"


class MyDescriptorPool;

struct Create_Tensor_Module;
struct Convolution_Module;
struct Normalization_Module;
struct Activation_Module;

struct Workblock_Module : public Vulkan_Module
{
	TensorDimension input_dimension;
	TensorDimension output_dimension;

	Workblock_Module()
	{
		set_ptrs();
	}

	virtual void build_workflow(std::vector<Vulkan_Module*>&) override {}
	virtual void initialize(Vulkan_App* vulkan) override;
	virtual void setDimensions() override;
	virtual void run() override;

	virtual reflect::input<vkBufferResource>& head_input() { return input_tensor; }
	virtual reflect::output<vkBufferResource>& tail_output() { return output_tensor; }

	reflect::input<vkBufferResource> input_tensor;
	reflect::output<vkBufferResource> output_tensor;

	std::vector<Vulkan_Module*> modules;

	REFLECT_VKMOD()
};


struct Convolution_Block : public Workblock_Module
{
	virtual void build_workflow(std::vector<Vulkan_Module*>&) override;
	virtual reflect::input<vkBufferResource>& head_input() override;
	virtual reflect::output<vkBufferResource>& tail_output() override;

	Create_Tensor_Module* weights_buffer = NULL;
	Convolution_Module* conv = NULL;
	Normalization_Module* norm = NULL;
	Activation_Module* activate = NULL;

	REFLECT_VKMOD()
};
