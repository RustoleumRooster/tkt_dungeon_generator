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

	struct pushconstant_struct
	{
		u32 n_vectors;
	};

	pushconstant_struct pushconstants
	{
		1 //n_vectors
	};

	virtual void build_workflow(std::vector<Vulkan_Module*>&) override;
	virtual void initialize(Vulkan_App* vulkan) override;
	virtual void setDimensions() override;

	virtual void run();

	Create_Tensor_Module* weights_buffer = NULL;
	Convolution_Module* conv = NULL;
	Normalization_Module* norm = NULL;
	Activation_Module* activate = NULL;

	reflect::input<vkBufferResource> input_tensor;
	reflect::output<vkBufferResource> output_tensor;

	reflect::input<vkBufferResource>& head_input();
	reflect::output<vkBufferResource>& tail_output();

	std::vector<Vulkan_Module*> modules;

	REFLECT_VKMOD()
};
