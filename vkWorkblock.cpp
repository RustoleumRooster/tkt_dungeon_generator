#include <irrlicht.h>
#include "Reflection.h"
#include "vkWorkblock.h"
#include "vkModules.h"
#include "vkUtilModules.h"

REFLECT_VKMOD_BEGIN(Workblock_Module)
	ALIAS("Workblock Layer")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_STRUCT_MEMBER(input_tensor)
	REFLECT_STRUCT_MEMBER(output_tensor)
REFLECT_VKMOD_END()

void Workblock_Module::setDimensions()
{
	for (Vulkan_Module* mod : modules)
		mod->setDimensions();
}

void Workblock_Module::build_workflow(std::vector<Vulkan_Module*>& append_list)
{
	weights_buffer = new Create_Tensor_Module();
	weights_buffer->dimensions = { 128,128,4,4 };

	conv = new Convolution_Module();
	conv->pushconstants.img_size_in = 64;
	conv->pushconstants.img_size_out = 32;
	conv->input_dimensions = { 16,128,64,64 };
	conv->output_dimensions = { 16,128,32,32 };

	norm = new Normalization_Module();
	norm->input_dimensions = { 16,128,32,32 };

	activate = new Activation_Module();
	activate->input_dimensions = { 16,128,32,32 };

	modules.push_back(weights_buffer);
	modules.push_back(conv);
	modules.push_back(norm);
	modules.push_back(activate);

	reflect::connect(&this->weights_buffer->output_tensor, &this->conv->weights);

	reflect::connect(&conv->pass_output, &norm->input_tensor);
	reflect::connect(&norm->pass_output, &activate->input_tensor);

	reflect::connect(&norm->mean_buffer, &activate->mean_buffer);
	reflect::connect(&norm->var_buffer, &activate->var_buffer);

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

void Workblock_Module::initialize(Vulkan_App* vulkan)
{
	Vulkan_Module::initialize(vulkan);

	for (Vulkan_Module* mod : modules)
	{
		mod->initialize(vulkan);
	}

}

void Workblock_Module::run()
{
	for (Vulkan_Module* mod : modules)
	{
		mod->signaled();
	}

	output_tensor.ready = true;
}

reflect::input<vkBufferResource>& Workblock_Module::head_input()
{
	return conv->input_tensor;
}

reflect::output<vkBufferResource>& Workblock_Module::tail_output()
{
	return activate->pass_output;
}
