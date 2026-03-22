#include <irrlicht.h>
#include "Reflection.h"
#include "vkWorkblock.h"
#include "vkModules.h"
#include "vkUtilModules.h"
#include "vkGroupNormModule.h"
#include "vkSiluModule.h"
#include "vkGeluModule.h"
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
	log() << GetDynamicReflection()->name << " (forward) :\n";

	for (Vulkan_Module* mod : modules)
	{
		assert(mod->ready_forward() && "Module not ready...");
		mod->forward();
		mod->forward_pass_complete = true;
	}

	output_tensor.ready = true;
}

void Workblock_Module::backward()
{
	log() << GetDynamicReflection()->name << " (backward) :\n";

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
// Convolution_Gelu_Block
//

REFLECT_VKMOD_BEGIN(Convolution_Gelu_Block)
	ALIAS("Convolution Gelu Block")
	INHERIT_FROM(Workblock_Module)
	//Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	//Backward Pass
	REFLECT_VKMOD_GRAD(input_grad)
	REFLECT_VKMOD_GRAD(output_grad)
REFLECT_VKMOD_END()

void Convolution_Gelu_Block::build_workflow(std::vector<Vulkan_Module*>& append_list)
{
	u32 N    = input_dimension.B;
	u32 C_in = input_dimension.C;
	u32 H    = input_dimension.H;
	u32 W    = input_dimension.W;
	u32 C_out = output_dimension.C;

	conv = new Convolution_Module();
	conv->pushconstants.k          = 4;
	conv->pushconstants.s          = 2;
	conv->pushconstants.p          = 1;
	conv->pushconstants.img_size_in  = H;
	conv->pushconstants.img_size_out = H / 2;
	conv->pushconstants.c_in       = C_in;
	conv->pushconstants.c_out      = C_out;
	conv->input_dimensions         = { N, C_in,  H,     W     };
	conv->output_dimensions        = { N, C_out, H / 2, W / 2 };

	group_norm = new GroupNorm_Module();
	group_norm->input_dimensions = { N, C_out, H / 2, W / 2 };
	group_norm->pushconstants    = { N, C_out, H / 2, W / 2, 32 };

	gelu = new Gelu_Module();
	gelu->input_dimensions = { N, C_out, H / 2, W / 2 };
	gelu->pushconstants    = { N, C_out, H / 2, W / 2 };

	modules.push_back(conv);
	modules.push_back(group_norm);
	modules.push_back(gelu);

	reflect::connect(&conv->output_tensor,       &group_norm->input_tensor);
	reflect::connect(&group_norm->output_tensor, &gelu->input_tensor);

	// Backward chain: gelu → group_norm → conv
	reflect::connect(&gelu->grad_output,        &group_norm->grad_input);
	reflect::connect(&group_norm->grad_output,  &conv->grad_input);

	for (Vulkan_Module* mod : modules)
	{
		mod->depth = this->depth + 1;
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>& Convolution_Gelu_Block::head_input()      { return conv->input_tensor; }
reflect::output<vkBufferResource>& Convolution_Gelu_Block::tail_output()    { return gelu->output_tensor; }
reflect::input<vkBufferResource>& Convolution_Gelu_Block::gradient_input()  { return gelu->grad_input; }
reflect::output<vkBufferResource>& Convolution_Gelu_Block::gradient_output(){ return conv->grad_output; }


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

	// Backward chain: sigmoid → conv
	reflect::connect(&sigmoid->grad_output, &conv->grad_input);

	for (Vulkan_Module* mod : modules)
	{
		mod->depth = this->depth + 1;
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>& FinalBlock_Module::head_input()      { return conv->input_tensor; }
reflect::output<vkBufferResource>& FinalBlock_Module::tail_output()    { return sigmoid->output_tensor; }
reflect::input<vkBufferResource>& FinalBlock_Module::gradient_input()  { return sigmoid->grad_input; }
reflect::output<vkBufferResource>& FinalBlock_Module::gradient_output(){ return conv->grad_output; }


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

	reflect::connect(&nn_up->output,             &group_norm->input_tensor);
	reflect::connect(&group_norm->output_tensor, &silu->input_tensor);
	reflect::connect(&silu->output_tensor,       &conv->input_tensor);

	// Backward chain: conv → silu → group_norm → nn_up
	reflect::connect(&conv->grad_output,        &silu->grad_input);
	reflect::connect(&silu->grad_output,        &group_norm->grad_input);
	reflect::connect(&group_norm->grad_output,  &nn_up->grad_input);

	for (Vulkan_Module* mod : modules)
	{
		mod->depth = this->depth + 1;
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>& UpscaleBlock_Module::head_input()      { return nn_up->input; }
reflect::output<vkBufferResource>& UpscaleBlock_Module::tail_output()    { return conv->output_tensor; }
reflect::input<vkBufferResource>& UpscaleBlock_Module::gradient_input()  { return conv->grad_input; }
reflect::output<vkBufferResource>& UpscaleBlock_Module::gradient_output(){ return nn_up->grad_output; }


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

	conv1 = new Convolution_Module();
	conv1->input_dimensions  = { N,C,H,W };
	conv1->output_dimensions = { N,C,H,W };
	conv1->pushconstants.k = 3;
	conv1->pushconstants.s = 1;
	conv1->pushconstants.p = 1;

	group_norm = new GroupNorm_Module();
	group_norm->input_dimensions = { N,C,H,W };
	group_norm->pushconstants = { N,C,H,W, 32 };

	silu = new Silu_Module();
	silu->input_dimensions = { N,C,H,W };
	silu->pushconstants = { N,C,H,W };

	conv2 = new Convolution_Module();
	conv2->input_dimensions  = { N,C,H,W };
	conv2->output_dimensions = { N,C,H,W };
	conv2->pushconstants.k = 3;
	conv2->pushconstants.s = 1;
	conv2->pushconstants.p = 1;

	skip = new Skip_Module();
	skip->input_dimensions = { N,C,H,W };

	// add_grad accumulates the two backward paths:
	//   grad_input_a = skip.grad_output_b  (identity / skip connection gradient)
	//   grad_input_b = group_norm.grad_output (conv path gradient)
	add_grad = new Add_Grad_Module();
	add_grad->input_dimensions = { N,C,H,W };

	// add_grad must run LAST in backward (depends on skip and group_norm outputs),
	// so insert it at index 0 — backward() iterates the modules vector in reverse.
	modules.push_back(add_grad);
	modules.push_back(conv1);
	modules.push_back(group_norm);
	modules.push_back(silu);
	modules.push_back(conv2);
	modules.push_back(skip);

	// skip branch
	//reflect::connect(&skip->output_tensor, &group_norm->input_tensor);

	reflect::connect(&conv1->output_tensor, &group_norm->input_tensor);
	reflect::connect(&group_norm->output_tensor, &silu->input_tensor);
	reflect::connect(&silu->output_tensor, &conv2->input_tensor);

	// add skip to output
	reflect::connect(&conv2->output_tensor, &skip->input_a);

	// Backward chain: skip → conv2 → silu → conv1 → group_norm → add_grad
	// skip.grad_output_a: gradient through the conv path
	// skip.grad_output_b: gradient through the identity path
	// add_grad sums both paths into the final upstream gradient
	reflect::connect(&skip->grad_output_a,       &conv2->grad_input);
	reflect::connect(&conv2->grad_output,        &silu->grad_input);
	reflect::connect(&silu->grad_output,         &group_norm->grad_input);
	reflect::connect(&group_norm->grad_output, &conv1->grad_input);

	reflect::connect(&skip->grad_output_b,       &add_grad->grad_input_a);
	reflect::connect(&conv1->grad_output,   &add_grad->grad_input_b);

	for (Vulkan_Module* mod : modules)
	{
		mod->depth = this->depth + 1;
		mod->is_submodule = true;
		append_list.push_back(mod);
	}
}

reflect::input<vkBufferResource>& ResBlock_Module::head_input()      { return conv1->input_tensor; }
reflect::output<vkBufferResource>& ResBlock_Module::tail_output()    { return skip->output; }
reflect::input<vkBufferResource>& ResBlock_Module::gradient_input()  { return skip->grad_input; }
reflect::output<vkBufferResource>& ResBlock_Module::gradient_output(){ return add_grad->grad_output; }
