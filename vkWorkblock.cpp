#include <irrlicht.h>
#include "Reflection.h"
#include "vkWorkblock.h"
#include "vkModules.h"
#include "vkUtilModules.h"

REFLECT_VKMOD_BEGIN(Workblock_Module)
	ALIAS("Workblock")		
	REFLECT_STRUCT_MEMBER(input_tensor)
	REFLECT_STRUCT_MEMBER(output_tensor)
REFLECT_VKMOD_END()

REFLECT_VKMOD_BEGIN(Convolution_Block)
	ALIAS("Convolution Block")
	REFLECT_STRUCT_MEMBER(input_tensor) //Actually belongs to Workblock_Module, but we can safely reflect it here as well
	REFLECT_STRUCT_MEMBER(output_tensor) //Actually belongs to Workblock_Module, but we can safely reflect it here as well
	INHERIT_FROM(Workblock_Module)
REFLECT_VKMOD_END()

//============================================================
// Workblock_Module (abstract base)
//

void Workblock_Module::setDimensions()
{
	for (Vulkan_Module* mod : modules)
		mod->setDimensions();
}

void Workblock_Module::initialize(Vulkan_App* vulkan)
{
	Vulkan_Module::initialize(vulkan);

	for (Vulkan_Module* mod : modules)
		mod->initialize(vulkan);
}

void Workblock_Module::run()
{
	for (Vulkan_Module* mod : modules)
		mod->signaled();

	output_tensor.ready = true;
}

//============================================================
// Convolution_Block
//

void Convolution_Block::build_workflow(std::vector<Vulkan_Module*>& append_list)
{
	u32 N = input_dimension.B;
	u32 C = input_dimension.C;
	u32 H = input_dimension.H;
	u32 W = input_dimension.W;
	u32 k = 4; //hardcoded in shader
	u32 s = 2; //hardcoded in shader

	weights_buffer = new Create_Tensor_Module();
	weights_buffer->dimensions = { C,C,k,k };

	conv = new Convolution_Module();
	conv->pushconstants.img_size_in = H;
	conv->pushconstants.img_size_out = H/2;
	conv->input_dimensions = { N,C,H,W };
	conv->output_dimensions = { N,C, H / 2, W / 2 };

	norm = new Normalization_Module();
	norm->input_dimensions = { N,C, H / 2, W / 2 };

	activate = new Activation_Module();
	activate->input_dimensions = { N,C, H / 2, W / 2 };

	modules.push_back(weights_buffer);
	modules.push_back(conv);
	modules.push_back(norm);
	modules.push_back(activate);

	reflect::connect(&conv->pass_output, &norm->input_tensor);
	reflect::connect(&norm->pass_output, &activate->input_tensor);

	reflect::connect(&norm->mean_buffer, &activate->mean_buffer);
	reflect::connect(&norm->var_buffer, &activate->var_buffer);

	reflect::connect(&weights_buffer->scratchpad, &conv->scratchpad);
	reflect::connect(&weights_buffer->scratchpad, &norm->scratchpad);
	reflect::connect(&weights_buffer->scratchpad, &activate->results_buffer);

	for (Vulkan_Module* mod : modules)
	{
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>& Convolution_Block::head_input()
{
	return conv->input_tensor;
}

reflect::output<vkBufferResource>& Convolution_Block::tail_output()
{
	return activate->pass_output;
}
