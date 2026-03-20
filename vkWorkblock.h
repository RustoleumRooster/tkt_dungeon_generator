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
struct GroupNorm_Module;
struct Silu_Module;
struct Skip_Module;
struct Add_Module;
struct NNUp_Module;
struct Sigmoid_Module;

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
	virtual reflect::input<vkBufferResource>& gradient_input() { return input_tensor; }
	virtual reflect::output<vkBufferResource>& gradient_output() { return output_tensor; }
	virtual void backward();

	reflect::input<vkBufferResource> input_tensor;
	reflect::output<vkBufferResource> output_tensor;
	reflect::input<vkBufferResource> input_grad;
	reflect::output<vkBufferResource> output_grad;

	std::vector<Vulkan_Module*> modules;

	REFLECT_VKMOD()
};


struct Convolution_Block : public Workblock_Module
{
	virtual void build_workflow(std::vector<Vulkan_Module*>&) override;
	virtual reflect::input<vkBufferResource>& head_input() override;
	virtual reflect::output<vkBufferResource>& tail_output() override;

	Create_Tensor_Module* scratchpad_buffer = NULL;
	Convolution_Module* conv = NULL;
	Normalization_Module* norm = NULL;
	Activation_Module* activate = NULL;

	REFLECT_VKMOD()
};


struct FinalBlock_Module : public Workblock_Module
{
	virtual void build_workflow(std::vector<Vulkan_Module*>&) override;
	virtual reflect::input<vkBufferResource>& head_input() override;
	virtual reflect::output<vkBufferResource>& tail_output() override;

	Create_Tensor_Module* scratchpad_buffer = NULL;
	Convolution_Module* conv		= NULL;
	Sigmoid_Module*     sigmoid		= NULL;

	REFLECT_VKMOD()
};


struct UpscaleBlock_Module : public Workblock_Module
{
	virtual void build_workflow(std::vector<Vulkan_Module*>&) override;
	virtual reflect::input<vkBufferResource>& head_input() override;
	virtual reflect::output<vkBufferResource>& tail_output() override;

	Create_Tensor_Module* scratchpad_buffer = NULL;
	NNUp_Module*          nn_up       = NULL;
	GroupNorm_Module*     group_norm  = NULL;
	Silu_Module*          silu        = NULL;
	Convolution_Module*   conv        = NULL;

	REFLECT_VKMOD()
};


struct ResBlock_Module : public Workblock_Module
{
	virtual void build_workflow(std::vector<Vulkan_Module*>&) override;
	virtual reflect::input<vkBufferResource>& head_input() override;
	virtual reflect::output<vkBufferResource>& tail_output() override;

	Create_Tensor_Module* scratchpad_buffer = NULL;
	Skip_Module*          skip        = NULL;
	GroupNorm_Module*     group_norm  = NULL;
	Convolution_Module*   conv1 = NULL;
	Silu_Module*          silu  = NULL;
	Convolution_Module*   conv2 = NULL;
	Add_Module*           add   = NULL;

	REFLECT_VKMOD()
};
