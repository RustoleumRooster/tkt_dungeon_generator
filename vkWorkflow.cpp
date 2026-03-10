
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

	Create_Tensor_Module* create_images_out = new Create_Tensor_Module();
	create_images_out->dimensions = { 16,128,32,32 };

	Create_Tensor_Module* create_weights = new Create_Tensor_Module();
	create_weights->dimensions = { 128,128,4,4 };

	Convolution_Module* conv_1 = new Convolution_Module();
	conv_1->pushconstants.img_size_in = 64;
	conv_1->pushconstants.img_size_out = 32;

	Normalization_Module* norm_1 = new Normalization_Module();
	norm_1->input_dimensions = { 16,128,32,32 };

	Activation_Module* activate_1 = new Activation_Module();

	Modules.push_back(VkMod_Reference{ create_images });
	Modules.push_back(VkMod_Reference{ create_images_out });
	Modules.push_back(VkMod_Reference{ create_weights });
	Modules.push_back(VkMod_Reference{ conv_1 });
	Modules.push_back(VkMod_Reference{ norm_1 });
	Modules.push_back(VkMod_Reference{ activate_1 });

	reflect::connect(&create_images->output_tensor, &conv_1->input_tensor);
	reflect::connect(&create_images_out->output_tensor, &conv_1->output_dummy);
	reflect::connect(&create_weights->output_tensor, &conv_1->weights);

	reflect::connect(&conv_1->output_tensor, &norm_1->input_tensor);
	reflect::connect(&conv_1->output_tensor, &activate_1->input_tensor);

	reflect::connect(&norm_1->mean_buffer, &activate_1->mean_buffer);
	reflect::connect(&norm_1->var_buffer, &activate_1->var_buffer);

	reflect::connect(&create_weights->scratchpad, &conv_1->scratchpad);
	reflect::connect(&create_weights->scratchpad, &norm_1->scratchpad);
	reflect::connect(&create_weights->scratchpad, &activate_1->results_buffer);

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