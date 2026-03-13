
#include "vkModules.h"
#include "vkUtilModules.h"

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

	//Create_Tensor_Module* output_buffer_A = new Create_Tensor_Module();
	//output_buffer_A->dimensions = { 16,128,32,32 };

	//Create_Tensor_Module* output_buffer_B = new Create_Tensor_Module();
	//output_buffer_A->dimensions = { 16,128,16,16 }; //smaller

	Create_Tensor_Module* weights_buffer = new Create_Tensor_Module();
	weights_buffer->dimensions = { 128,128,4,4 };

	//Create_Tensor_Module* create_codebook = new Create_Tensor_Module();
	//create_codebook->dimensions = { 1,128,4,4 };

	Convolution_Module* conv_1 = new Convolution_Module();
	conv_1->pushconstants.img_size_in = 64;
	conv_1->pushconstants.img_size_out = 32;
	conv_1->input_dimensions = { 16,128,64,64 };
	conv_1->output_dimensions = { 16,128,32,32 };

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

	Normalization_Module* norm_1 = new Normalization_Module();
	norm_1->input_dimensions = { 16,128,32,32 };

	Activation_Module* activate_1 = new Activation_Module();
	activate_1->input_dimensions = { 16,128,32,32 };

	Modules.push_back(VkMod_Reference{ create_images });
	//Modules.push_back(VkMod_Reference{ output_buffer_A });
	//Modules.push_back(VkMod_Reference{ output_buffer_B });
	Modules.push_back(VkMod_Reference{ weights_buffer });
	Modules.push_back(VkMod_Reference{ conv_1 });
	Modules.push_back(VkMod_Reference{ conv_2 });
	Modules.push_back(VkMod_Reference{ conv_3 });
	Modules.push_back(VkMod_Reference{ norm_1 });
	Modules.push_back(VkMod_Reference{ activate_1 });

	//1
	reflect::connect(&create_images->output_tensor, &conv_1->input_tensor);
	//reflect::connect(&output_buffer_A->output_tensor, &conv_1->output_buffer);
	reflect::connect(&weights_buffer->output_tensor, &conv_1->weights);

	reflect::connect(&conv_1->pass_output, &norm_1->input_tensor);
	reflect::connect(&norm_1->pass_output, &activate_1->input_tensor);

	reflect::connect(&norm_1->mean_buffer, &activate_1->mean_buffer);
	reflect::connect(&norm_1->var_buffer, &activate_1->var_buffer);

	//2
	reflect::connect(&activate_1->pass_output, &conv_2->input_tensor);
	//reflect::connect(&output_buffer_B->output_tensor, &conv_2->output_buffer);
	reflect::connect(&weights_buffer->output_tensor, &conv_2->weights);

	//3
	reflect::connect(&conv_2->pass_output, &conv_3->input_tensor);
	//reflect::connect(&output_buffer_A->output_tensor, &conv_3->output_buffer);
	reflect::connect(&weights_buffer->output_tensor, &conv_3->weights);

	reflect::connect(&weights_buffer->scratchpad, &conv_1->scratchpad);
	reflect::connect(&weights_buffer->scratchpad, &conv_2->scratchpad);
	reflect::connect(&weights_buffer->scratchpad, &conv_3->scratchpad);
	reflect::connect(&weights_buffer->scratchpad, &norm_1->scratchpad);
	reflect::connect(&weights_buffer->scratchpad, &activate_1->results_buffer);

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

	vulkan->run_workflow();
	vulkan->cleanup();
}