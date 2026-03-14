
#include "vkModules.h"
#include "vkUtilModules.h"
#include "vkQuantizeModule.h"
#include "vkWorkblock.h"

Vulkan_Workflow::~Vulkan_Workflow()
{
	for (auto& v : Modules)
	{
		delete v.X;
	}
}

void Vulkan_Workflow::make_default_workflow()
{
	Create_Tensor_Module* create_images = new Create_Tensor_Module();
	create_images->dimensions = { 16,128,64,64 };

	Create_Tensor_Module* weights_buffer = new Create_Tensor_Module();
	weights_buffer->dimensions = { 128,128,4,4 };

	Create_Tensor_Module* create_codebook = new Create_Tensor_Module();
	create_codebook->dimensions = { 1,128,8,8 };

	Workblock_Module* conv_block = new Workblock_Module();
	conv_block->input_dimension = { 16,128,64,64 };
	conv_block->output_dimension = { 16,128,32,32 };

	Convolution_Module* conv_2 = new Convolution_Module();
	conv_2->pushconstants.img_size_in = 32;
	conv_2->pushconstants.img_size_out = 16;
	conv_2->input_dimensions = { 16,128,32,32 };
	conv_2->output_dimensions = { 16,128,16,16 };

	Convolution_Module* conv_3 = new Convolution_Module();
	conv_3->pushconstants.img_size_in = 16;
	conv_3->pushconstants.img_size_out = 8;
	conv_3->input_dimensions = { 16,128,32,32 };
	conv_3->output_dimensions = { 16,128,8,8 };

	Quantize_Module* quantize = new Quantize_Module();
	quantize->input_dimensions = { 16,128,8,8 };
	quantize->codebook_size = { 1,1,512,128 };

	Modules.push_back(VkMod_Reference{ create_images });
	Modules.push_back(VkMod_Reference{ weights_buffer });
	Modules.push_back(VkMod_Reference{ create_codebook });
	Modules.push_back(VkMod_Reference{ conv_block });
	Modules.push_back(VkMod_Reference{ conv_2 });
	Modules.push_back(VkMod_Reference{ conv_3 });
	Modules.push_back(VkMod_Reference{ quantize });

	std::vector<Vulkan_Module*> append_list;

	for (VkMod_Reference& ref : Modules)
		ref.X->build_workflow(append_list);

	for (Vulkan_Module* mod : append_list)
		Modules.push_back(VkMod_Reference{ mod });

	reflect::connect(&create_images->output_tensor, &conv_block->input_tensor);
	reflect::connect(&create_images->output_tensor, &conv_block->head_input());
	reflect::connect(&conv_block->tail_output(), &conv_2->input_tensor);
	reflect::connect(&weights_buffer->output_tensor, &conv_2->weights);

	//3
	reflect::connect(&conv_2->pass_output, &conv_3->input_tensor);
	
	//reflect::connect(&output_buffer_A->output_tensor, &conv_3->output_buffer);
	reflect::connect(&weights_buffer->output_tensor, &conv_3->weights);

	//quantize
	reflect::connect(&conv_3->pass_output, &quantize->input_tensor);
	reflect::connect(&create_codebook->output_tensor, &quantize->codebook);

	//reflect::connect(&weights_buffer->scratchpad, &conv_1->scratchpad);
	reflect::connect(&weights_buffer->scratchpad, &quantize->results_buffer);
	reflect::connect(&weights_buffer->scratchpad, &conv_2->scratchpad);
	reflect::connect(&weights_buffer->scratchpad, &conv_3->scratchpad);
	//reflect::connect(&weights_buffer->scratchpad, &norm_1->scratchpad);
	//reflect::connect(&weights_buffer->scratchpad, &activate_1->results_buffer);

	

	items.clear();
	for (VkMod_Reference& ref : Modules)
		items.push_back(ref.X);
}

void Vulkan_Workflow::initialize_and_run(Vulkan_App* vulkan)
{
	for (VkMod_Reference& mod : Modules)
	{
		mod.X->initialize(vulkan);
	}

	vulkan->pre_run_check();

	for (VkMod_Reference& mod : Modules)
	{
		if(mod.X->is_submodule == false)
			mod.X->signaled();
	}

	vulkan->cleanup();
}