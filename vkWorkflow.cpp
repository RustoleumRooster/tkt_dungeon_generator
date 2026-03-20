
#include "vkModules.h"
#include "vkUtilModules.h"
#include "vkQuantizeModule.h"
#include "vkWorkblock.h"
#include "vkBCELossModule.h"
#include <vulkan/vulkan.h>

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

	ResBlock_Module* res_block = new ResBlock_Module();
	res_block->input_dimension  = { 16,128,8,8 };
	res_block->output_dimension = { 16,128,8,8 };

	UpscaleBlock_Module* up_block = new UpscaleBlock_Module();
	up_block->input_dimension  = { 16,128,8,8 };
	up_block->output_dimension = { 16,64,16,16 };

	UpscaleBlock_Module* up_block_2 = new UpscaleBlock_Module();
	up_block_2->input_dimension  = { 16,64,16,16 };
	up_block_2->output_dimension = { 16,32,32,32 };

	UpscaleBlock_Module* up_block_3 = new UpscaleBlock_Module();
	up_block_3->input_dimension  = { 16,32,32,32 };
	up_block_3->output_dimension = { 16,16,64,64 };

	FinalBlock_Module* final_block = new FinalBlock_Module();
	final_block->input_dimension  = { 16,16,64,64 };
	final_block->output_dimension = { 16,1,64,64 };

	Create_Tensor_Module* ground_truth = new Create_Tensor_Module();
	ground_truth->dimensions = { 16,1,64,64 };

	BCE_Loss_Module* bce_loss = new BCE_Loss_Module();
	bce_loss->input_dimensions = { 16,1,64,64 };

	Modules.push_back(VkMod_Reference{ create_images });
	Modules.push_back(VkMod_Reference{ conv_block });
	Modules.push_back(VkMod_Reference{ conv_block_2 });
	Modules.push_back(VkMod_Reference{ conv_block_3 });
	Modules.push_back(VkMod_Reference{ quantize });
	Modules.push_back(VkMod_Reference{ res_block });
	Modules.push_back(VkMod_Reference{ up_block });
	Modules.push_back(VkMod_Reference{ up_block_2 });
	Modules.push_back(VkMod_Reference{ up_block_3 });
	Modules.push_back(VkMod_Reference{ final_block });
	Modules.push_back(VkMod_Reference{ ground_truth });
	Modules.push_back(VkMod_Reference{ bce_loss });

	backward_pass_head = bce_loss;

	std::vector<Vulkan_Module*> append_list;

	for (VkMod_Reference& ref : Modules)
		ref.X->build_workflow(append_list);

	for (Vulkan_Module* mod : append_list)
		Modules.push_back(VkMod_Reference{ mod });
	
	//=====================================================
	// Encoder
	//
	
	//1
	reflect::connect(&create_images->output_tensor, &conv_block->input_tensor);
	reflect::connect(&create_images->output_tensor, &conv_block->head_input());

	//2
	reflect::connect(&conv_block->tail_output(), &conv_block_2->input_tensor);
	reflect::connect(&conv_block->tail_output(), &conv_block_2->head_input());

	//3
	reflect::connect(&conv_block_2->tail_output(), &conv_block_3->input_tensor);
	reflect::connect(&conv_block_2->tail_output(), &conv_block_3->head_input());

	//=====================================================
	// Quantize
	//

	reflect::connect(&conv_block_3->tail_output(), &quantize->input_tensor);

	//=====================================================
	// Decoder
	//

	reflect::connect(&quantize->output, &res_block->input_tensor);
	reflect::connect(&quantize->output, &res_block->head_input());

	reflect::connect(&res_block->tail_output(), &up_block->input_tensor);
	reflect::connect(&res_block->tail_output(), &up_block->head_input());

	reflect::connect(&up_block->tail_output(), &up_block_2->input_tensor);
	reflect::connect(&up_block->tail_output(), &up_block_2->head_input());

	reflect::connect(&up_block_2->tail_output(), &up_block_3->input_tensor);
	reflect::connect(&up_block_2->tail_output(), &up_block_3->head_input());

	reflect::connect(&up_block_3->tail_output(), &final_block->input_tensor);
	reflect::connect(&up_block_3->tail_output(), &final_block->head_input());

	//=====================================================
	// Loss
	//

	reflect::connect(&final_block->tail_output(), &bce_loss->predictions);
	reflect::connect(&ground_truth->output_tensor, &bce_loss->ground_truth);

	items.clear();
	for (VkMod_Reference& ref : Modules)
		items.push_back(ref.X);
}

void Vulkan_Workflow::initialize_and_run(Vulkan_App* vulkan)
{
	for (VkMod_Reference& mod : Modules)
	{
		if(!mod.X->is_submodule)
			mod.X->initialize(vulkan);
	}

	for (VkMod_Reference& mod : Modules)
	{
		mod.X->setDimensions();
	}

	plan_memory(vulkan);

	vulkan->pre_run_check();

	for (VkMod_Reference& mod : Modules)
	{
		if(mod.X->is_submodule == false)
			mod.X->signaled();
	}

	if (backward_pass_head)
	{
		if(backward_pass_head->ready_backward())
			backward_pass_head->backward();
	}

	vulkan->cleanup();

}

void Vulkan_Workflow::plan_memory(Vulkan_App* vulkan)
{
	const VkDeviceSize alignment = 256;

	VkDeviceSize feature_total = 0;
	VkDeviceSize param_total   = 0;
	VkDeviceSize grad_total    = 0;
	VkDeviceSize other_total   = 0;

	for (VkMod_Reference& ref : Modules)
	{
		Vulkan_Module* mod = ref.X;
		reflect::TypeDescriptor_Struct* td = mod->GetDynamicReflection();
		if (!td) continue;

		if (td->inherited_type == &Workblock_Module::Reflection)
			continue;

		for (reflect::Member& m : td->members)
		{
			reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;
			if (!m_tD) continue;

			if (m_tD->inherited_type == &reflect::output_type::Reflection)
			{
				reflect::output_type* out = (reflect::output_type*)m.get(mod);
				VkDeviceSize sz = out->aligned_size(alignment);

				if (m.flags & REFLECT_VKMOD_COMPONENT_GRAD)
					grad_total += sz;
				else if (m.flags & REFLECT_VKMOD_COMPONENT_FEAT)
					feature_total += sz;
				else if (m.flags & REFLECT_VKMOD_COMPONENT_PARAM)
					param_total += sz;
				else
					other_total += sz;
			}
		}
	}

	auto mb = [](VkDeviceSize b) { return b / (1024.0 * 1024.0); };

	std::cout << "[Memory Plan]\n";
	std::cout << "  Feature maps : " << mb(feature_total) << " MB  (" << feature_total << " B)\n";
	std::cout << "  Parameters   : " << mb(param_total)   << " MB  (" << param_total   << " B)\n";
	std::cout << "  Gradients    : " << mb(grad_total)    << " MB  (" << grad_total    << " B)\n";
	std::cout << "  Other        : " << mb(other_total)   << " MB  (" << other_total   << " B)\n";

	auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	if (feature_total > 0)
	{
		feature_buffer = vulkan->create_buffer(feature_total, usage);
		std::cout << "  feature_buffer created\n";
	}
	if (param_total > 0)
	{
		param_buffer = vulkan->create_buffer(param_total, usage);
		std::cout << "  param_buffer created\n";
	}
	if (grad_total > 0)
	{
		grad_buffer = vulkan->create_buffer(grad_total, usage);
		std::cout << "  grad_buffer created\n";
	}
	if (other_total > 0)
	{
		other_buffer = vulkan->create_buffer(other_total, usage);
		std::cout << "  other_buffer created\n";
	}

	VkDeviceSize feature_t = 0;
	VkDeviceSize param_t = 0;
	VkDeviceSize grad_t = 0;
	VkDeviceSize other_t = 0;

	for (VkMod_Reference& ref : Modules)
	{
		Vulkan_Module* mod = ref.X;
		reflect::TypeDescriptor_Struct* td = mod->GetDynamicReflection();
		if (!td) continue;

		for (reflect::Member& m : td->members)
		{
			reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;
			if (!m_tD) continue;

			if (m_tD == &reflect::output<vkBufferResource>::Reflection)
			{
				reflect::output<vkBufferResource>* out = (reflect::output<vkBufferResource>*)m.get(mod);
				VkDeviceSize sz = out->aligned_size(alignment);

				if (m.flags & REFLECT_VKMOD_COMPONENT_GRAD)
				{
					out->X->Buffer = grad_buffer->Buffer;
					out->X->BufferInfo.buffer = grad_buffer->Buffer;
					out->X->BufferInfo.offset = grad_t;
					out->X->BufferInfo.range = sz;
					out->X->BufferMemory = grad_buffer->BufferMemory;
					grad_t += sz;

				}
				else if (m.flags & REFLECT_VKMOD_COMPONENT_FEAT)
				{
					out->X->Buffer = feature_buffer->Buffer;
					out->X->BufferInfo.buffer = feature_buffer->Buffer;
					out->X->BufferInfo.offset = feature_t;
					out->X->BufferInfo.range = sz;
					out->X->BufferMemory = feature_buffer->BufferMemory;
					feature_t += sz;

				}
				else if (m.flags & REFLECT_VKMOD_COMPONENT_PARAM)
				{
					out->X->Buffer = param_buffer->Buffer;
					out->X->BufferInfo.buffer = param_buffer->Buffer;
					out->X->BufferInfo.offset = param_t;
					out->X->BufferInfo.range = sz;
					out->X->BufferMemory = param_buffer->BufferMemory;
					param_t += sz;

				}
				else
				{
					out->X->Buffer = other_buffer->Buffer;
					out->X->BufferInfo.buffer = other_buffer->Buffer;
					out->X->BufferInfo.offset = other_t;
					out->X->BufferInfo.range = sz;
					out->X->BufferMemory = other_buffer->BufferMemory;
					other_t += sz;
				}
			}
		}
	}
}