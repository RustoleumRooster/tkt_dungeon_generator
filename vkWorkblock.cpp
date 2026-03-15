#include <irrlicht.h>
#include "Reflection.h"
#include "vkWorkblock.h"
#include "vkModules.h"
#include "vkUtilModules.h"
#include "vkGroupNormModule.h"
#include "vkGroupNorm2Module.h"
#include "vkSiluModule.h"
#include "vkSkipModule.h"
#include "vkAddModule.h"

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


//============================================================
// ResBlock_Module
//

REFLECT_VKMOD_BEGIN(ResBlock_Module)
	ALIAS("Res Block")
	INHERIT_FROM(Workblock_Module)
	REFLECT_STRUCT_MEMBER(input_tensor)
	REFLECT_STRUCT_MEMBER(output_tensor)
REFLECT_VKMOD_END()

void ResBlock_Module::build_workflow(std::vector<Vulkan_Module*>& append_list)
{
	u32 N = input_dimension.B;
	u32 C = input_dimension.C;
	u32 H = input_dimension.H;
	u32 W = input_dimension.W;

	scratchpad_buffer = new Create_Tensor_Module();
	scratchpad_buffer->dimensions = { 1,1,1,512 };

	skip = new Skip_Module();
	skip->input_dimensions = { N,C,H,W };

	group_norm = new GroupNorm_Module();
	group_norm->input_dimensions = { N,C,H,W };
	group_norm->pushconstants = { N,C,H,W, 32 };

	group_norm2 = new GroupNorm2_Module();
	group_norm2->input_dimensions = { N,C,H,W };
	group_norm2->pushconstants = { N,C,H,W, 32 };

	conv1 = new Convolution_Module();
	conv1->input_dimensions  = { N,C,H,W };
	conv1->output_dimensions = { N,C,H,W };
	conv1->pushconstants.n          = N;
	conv1->pushconstants.c_in       = C;
	conv1->pushconstants.c_out      = C;
	conv1->pushconstants.img_size_in  = H;
	conv1->pushconstants.img_size_out = H;

	silu = new Silu_Module();
	silu->input_dimensions = { N,C,H,W };
	silu->pushconstants = { N,C,H,W };

	conv2 = new Convolution_Module();
	conv2->input_dimensions  = { N,C,H,W };
	conv2->output_dimensions = { N,C,H,W };
	conv2->pushconstants.n          = N;
	conv2->pushconstants.c_in       = C;
	conv2->pushconstants.c_out      = C;
	conv2->pushconstants.img_size_in  = H;
	conv2->pushconstants.img_size_out = H;

	add = new Add_Module();
	add->input_dimensions = { N,C,H,W };

	modules.push_back(scratchpad_buffer);
	modules.push_back(skip);
	modules.push_back(group_norm);
	modules.push_back(group_norm2);
	modules.push_back(conv1);
	modules.push_back(silu);
	modules.push_back(conv2);
	modules.push_back(add);

	// skip branch
	reflect::connect(&skip->pass_output, &group_norm->input_tensor);

	// normalization chain
	reflect::connect(&group_norm->mean_buffer, &group_norm2->mean_buffer);
	reflect::connect(&group_norm->var_buffer,  &group_norm2->var_buffer);
	reflect::connect(&group_norm->pass_output, &group_norm2->input_tensor);

	// forward pass
	reflect::connect(&group_norm2->pass_output, &conv1->input_tensor);
	reflect::connect(&conv1->pass_output,        &silu->input_tensor);
	reflect::connect(&silu->pass_output,         &conv2->input_tensor);

	// add skip to output
	reflect::connect(&conv2->pass_output,  &add->input_a);
	reflect::connect(&skip->skip_output,   &add->input_b);

	// scratchpad
	reflect::connect(&scratchpad_buffer->scratchpad, &group_norm->scratchpad);
	reflect::connect(&scratchpad_buffer->scratchpad, &group_norm2->scratchpad);
	reflect::connect(&scratchpad_buffer->scratchpad, &conv1->scratchpad);
	reflect::connect(&scratchpad_buffer->scratchpad, &conv2->scratchpad);

	for (Vulkan_Module* mod : modules)
	{
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>& ResBlock_Module::head_input()
{
	return skip->input_tensor;
}

reflect::output<vkBufferResource>& ResBlock_Module::tail_output()
{
	return add->output;
}
