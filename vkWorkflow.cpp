
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
	Create_Tensor_Module* create_weights = new Create_Tensor_Module();
	Convolution_Module* conv_1 = new Convolution_Module();
	Dummy_Consumer* dummy = new Dummy_Consumer();
	
	Modules.push_back(VkMod_Reference{ create_images });
	Modules.push_back(VkMod_Reference{ create_weights });
	Modules.push_back(VkMod_Reference{ conv_1 });
	Modules.push_back(VkMod_Reference{ dummy });
	
	reflect::connect(&create_images->output_tensor, &conv_1->input_tensor);
	reflect::connect(&create_weights->output_tensor, &conv_1->weights);
	reflect::connect(&create_weights->scratchpad, &conv_1->scratchpad);
	reflect::connect(&conv_1->output_tensor, &dummy->input_tensor);

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