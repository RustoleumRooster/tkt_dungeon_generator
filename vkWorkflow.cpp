
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

	Convolution_Block* conv_block = new Convolution_Block();
	conv_block->input_dimension = { 16,128,64,64 };
	conv_block->output_dimension = { 16,128,32,32 };

	Convolution_Block* conv_block_2 = new Convolution_Block();
	conv_block_2->input_dimension = { 16,128,32,32 };
	conv_block_2->output_dimension = { 16,128,16,16 };

	Convolution_Block* conv_block_3 = new Convolution_Block();
	conv_block_3->input_dimension = { 16,128,16,16 };
	conv_block_3->output_dimension = { 16,128,8,8 };

	Quantize_Module* quantize = new Quantize_Module();
	quantize->input_dimensions = { 16,128,8,8 };
	quantize->codebook_size = { 1,1,512,128 };

	Modules.push_back(VkMod_Reference{ create_images });
	Modules.push_back(VkMod_Reference{ weights_buffer });
	Modules.push_back(VkMod_Reference{ create_codebook });
	Modules.push_back(VkMod_Reference{ conv_block });
	Modules.push_back(VkMod_Reference{ conv_block_2 });
	Modules.push_back(VkMod_Reference{ conv_block_3 });
	Modules.push_back(VkMod_Reference{ quantize });

	std::vector<Vulkan_Module*> append_list;

	for (VkMod_Reference& ref : Modules)
		ref.X->build_workflow(append_list);

	for (Vulkan_Module* mod : append_list)
		Modules.push_back(VkMod_Reference{ mod });

	//1
	reflect::connect(&create_images->output_tensor, &conv_block->input_tensor);
	reflect::connect(&create_images->output_tensor, &conv_block->head_input());

	//2
	reflect::connect(&conv_block->tail_output(), &conv_block_2->input_tensor);
	reflect::connect(&conv_block->tail_output(), &conv_block_2->head_input());

	//3
	reflect::connect(&conv_block_2->tail_output(), &conv_block_3->input_tensor);
	reflect::connect(&conv_block_2->tail_output(), &conv_block_3->head_input());

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