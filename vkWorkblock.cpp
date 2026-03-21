#include <irrlicht.h>
#include "Reflection.h"
#include "vkWorkblock.h"
#include "vkModules.h"
#include "vkUtilModules.h"
#include "vkGroupNormModule.h"
#include "vkSiluModule.h"
#include "vkSkipModule.h"
#include "vkNNUpModule.h"
#include "vkSigmoidModule.h"
#include "vkBCELossModule.h"
#include <cassert>

REFLECT_VKMOD_BEGIN(Workblock_Module)
	ALIAS("Workblock")		
	//Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	//Backward Pass
	REFLECT_VKMOD_GRAD(input_grad)
	REFLECT_VKMOD_GRAD(output_grad)
REFLECT_VKMOD_END()

REFLECT_VKMOD_BEGIN(Convolution_Block)
	ALIAS("Convolution Block")
	//Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor) //Actually belongs to Workblock_Module, but we can safely reflect it here as well
	REFLECT_VKMOD_FEAT(output_tensor)
	//Backward Pass
	REFLECT_VKMOD_GRAD(input_grad) 
	REFLECT_VKMOD_GRAD(output_grad) 
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

void Workblock_Module::forward()
{
	for (Vulkan_Module* mod : modules)
	{
		assert(mod->ready_forward() && "Module not ready...");
		mod->forward();
	}

	output_tensor.ready = true;
}

void Workblock_Module::backward()
{
	for (int i= modules.size()-1; i>=0; i--)
		modules[i]->backward();
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

	conv = new Convolution_Module();
	conv->pushconstants.k = 4;
	conv->pushconstants.s = 2;
	conv->pushconstants.p = 1;
	conv->pushconstants.img_size_in = H;
	conv->pushconstants.img_size_out = H/2;
	conv->pushconstants.c_in = C;
	conv->pushconstants.c_out = C;
	conv->input_dimensions = { N,C,H,W };
	conv->output_dimensions = { N,C, H / 2, W / 2 };

	norm = new Normalization_Module();
	norm->input_dimensions = { N,C, H / 2, W / 2 };

	activate = new Activation_Module();
	activate->input_dimensions = { N,C, H / 2, W / 2 };

	modules.push_back(conv);
	modules.push_back(norm);
	modules.push_back(activate);

	reflect::connect(&conv->output_tensor, &norm->input_tensor);
	reflect::connect(&conv->output_tensor, &activate->input_tensor);

	reflect::connect(&norm->mean_buffer, &activate->mean_buffer);
	reflect::connect(&norm->var_buffer, &activate->var_buffer);

	for (Vulkan_Module* mod : modules)
	{
		mod->depth = this->depth + 1;
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
	return activate->output_tensor;
}


//============================================================
// FinalBlock_Module
//

REFLECT_VKMOD_BEGIN(FinalBlock_Module)
	ALIAS("Final Block")
	INHERIT_FROM(Workblock_Module)
	//Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	//Backward Pass
	REFLECT_VKMOD_GRAD(input_grad)
	REFLECT_VKMOD_GRAD(output_grad)
REFLECT_VKMOD_END()

void FinalBlock_Module::build_workflow(std::vector<Vulkan_Module*>& append_list)
{
	u32 N   = input_dimension.B;
	u32 C   = input_dimension.C;
	u32 H   = input_dimension.H;
	u32 W   = input_dimension.W;

	conv = new Convolution_Module();
	conv->pushconstants.k = 1;
	conv->pushconstants.s = 1;
	conv->pushconstants.p = 0;
	conv->input_dimensions  = { N, C, H, W };
	conv->output_dimensions = { N, 1, H, W };

	sigmoid = new Sigmoid_Module();
	sigmoid->input_dimensions = { N, 1, H, W };
	sigmoid->pushconstants    = { N, 1, H, W };

	modules.push_back(conv);
	modules.push_back(sigmoid);

	reflect::connect(&conv->output_tensor, &sigmoid->input_tensor);


	for (Vulkan_Module* mod : modules)
	{
		mod->depth = this->depth + 1;
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>& FinalBlock_Module::head_input()
{
	return conv->input_tensor;
}

reflect::output<vkBufferResource>& FinalBlock_Module::tail_output()
{
	return sigmoid->output_tensor;
}


//============================================================
// UpscaleBlock_Module
//

REFLECT_VKMOD_BEGIN(UpscaleBlock_Module)
	ALIAS("Upscale Block")
	INHERIT_FROM(Workblock_Module)
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
REFLECT_VKMOD_END()

void UpscaleBlock_Module::build_workflow(std::vector<Vulkan_Module*>& append_list)
{
	u32 N    = input_dimension.B;
	u32 C_in = input_dimension.C;
	u32 H    = input_dimension.H;
	u32 W    = input_dimension.W;
	u32 C_out = output_dimension.C;

	nn_up = new NNUp_Module();
	nn_up->input_dimensions = { N, C_in, H, W };

	group_norm = new GroupNorm_Module();
	group_norm->input_dimensions = { N, C_in, H*2, W*2 };
	group_norm->pushconstants    = { N, C_in, H*2, W*2, 32 };

	silu = new Silu_Module();
	silu->input_dimensions = { N, C_in, H*2, W*2 };
	silu->pushconstants    = { N, C_in, H*2, W*2 };

	conv = new Convolution_Module();
	conv->pushconstants.k = 3;
	conv->pushconstants.s = 1;
	conv->pushconstants.p = 1;
	conv->input_dimensions  = { N, C_in,  H*2, W*2 };
	conv->output_dimensions = { N, C_out, H*2, W*2 };

	modules.push_back(nn_up);
	modules.push_back(group_norm);
	modules.push_back(silu);
	modules.push_back(conv);

	reflect::connect(&nn_up->output,           &group_norm->input_tensor);
	reflect::connect(&group_norm->output_tensor, &silu->input_tensor);
	reflect::connect(&silu->output_tensor,       &conv->input_tensor);

	for (Vulkan_Module* mod : modules)
	{
		mod->depth = this->depth + 1;
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>& UpscaleBlock_Module::head_input()
{
	return nn_up->input;
}

reflect::output<vkBufferResource>& UpscaleBlock_Module::tail_output()
{
	return conv->output_tensor;
}


//============================================================
// ResBlock_Module
//

REFLECT_VKMOD_BEGIN(ResBlock_Module)
	ALIAS("Res Block")
	INHERIT_FROM(Workblock_Module)
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
REFLECT_VKMOD_END()

void ResBlock_Module::build_workflow(std::vector<Vulkan_Module*>& append_list)
{
	u32 N = input_dimension.B;
	u32 C = input_dimension.C;
	u32 H = input_dimension.H;
	u32 W = input_dimension.W;

	group_norm = new GroupNorm_Module();
	group_norm->input_dimensions = { N,C,H,W };
	group_norm->pushconstants    = { N,C,H,W, 32 };

	conv1 = new Convolution_Module();
	conv1->input_dimensions  = { N,C,H,W };
	conv1->output_dimensions = { N,C,H,W };

	silu = new Silu_Module();
	silu->input_dimensions = { N,C,H,W };
	silu->pushconstants = { N,C,H,W };

	conv2 = new Convolution_Module();
	conv2->input_dimensions  = { N,C,H,W };
	conv2->output_dimensions = { N,C,H,W };

	skip = new Skip_Module();
	skip->input_dimensions = { N,C,H,W };

	//modules.push_back(skip);
	modules.push_back(group_norm);
	modules.push_back(conv1);
	modules.push_back(silu);
	modules.push_back(conv2);
	modules.push_back(skip);

	// skip branch
	//reflect::connect(&skip->output_tensor, &group_norm->input_tensor);

	// forward pass
	reflect::connect(&group_norm->output_tensor, &conv1->input_tensor);
	reflect::connect(&conv1->output_tensor,      &silu->input_tensor);
	reflect::connect(&silu->output_tensor,       &conv2->input_tensor);

	// add skip to output
	reflect::connect(&conv2->output_tensor, &skip->input_a);

	for (Vulkan_Module* mod : modules)
	{
		mod->depth = this->depth + 1;
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>& ResBlock_Module::head_input()
{
	return group_norm->input_tensor;
}

reflect::output<vkBufferResource>& ResBlock_Module::tail_output()
{
	return skip->output;
}
