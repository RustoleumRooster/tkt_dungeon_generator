
#include "vkModules.h"
#include "vkUtilModules.h"
#include "vkQuantizeModule.h"
#include "vkWorkblock.h"
#include "vkBCELossModule.h"
#include <vulkan/vulkan.h>
#include "vkGroupNormModule.h"
#include "vkSkipModule.h"
#include <cassert>

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

	reflect::connect(&quantize->output_tensor, &res_block->input_tensor);
	reflect::connect(&quantize->output_tensor, &res_block->head_input());

	reflect::output<vkBufferResource>* in = dynamic_cast<reflect::output<vkBufferResource>*>(res_block->group_norm->input_tensor.src_output);

	if (in)
		reflect::connect(in, &res_block->skip->input_b);


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
		if (!mod.X->is_submodule)
		{
			mod.X->initialize(vulkan);
			mod.X->setDimensions();
		}
	}

	plan_memory(vulkan);


	for (VkMod_Reference& mod : Modules)
	{
		if (mod.X->is_submodule == false)
		{
			assert(mod.X->ready_forward() && "Module not ready...");
			mod.X->forward();
			mod.X->forward_pass_complete = true;
		}
	}
	/*
	if (backward_pass_head)
	{
		if(backward_pass_head->ready_backward())
			backward_pass_head->backward();
	}
	*/
	vulkan->cleanup();

}

void Vulkan_Workflow::plan_memory(Vulkan_App* vulkan)
{
	const VkDeviceSize alignment = 256;

	// Query total device-local memory
	VkPhysicalDeviceMemoryProperties memProps{};
	vkGetPhysicalDeviceMemoryProperties(vulkan->m_device->getPhysicalDevice(), &memProps);

	VkDeviceSize device_local_total = 0;
	for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
	{
		if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
			device_local_total += memProps.memoryHeaps[i].size;
	}

	auto mb = [](VkDeviceSize b) { return b / (1024.0 * 1024.0); };
	std::cout << "[Memory Plan]  Device-local memory: " << mb(device_local_total) << " MB\n";

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

				assert(sz > 0 && "plan_memory: output buffer has zero size");
				assert(sz < device_local_total && "plan_memory: single buffer exceeds total device memory");

				if (m.flags & REFLECT_VKMOD_COMPONENT_GRAD)
				{
					//it's a reusable buffer, just size it up to the largest size needed!
					grad_total = std::max(grad_total, sz);
				}
				else if (m.flags & REFLECT_VKMOD_COMPONENT_FEAT)
					feature_total += sz;
				else
					other_total += sz;
			}
			else if (m_tD->inherited_type == &reflect::parameter_type::Reflection)
			{
				reflect::parameter_type* out = (reflect::parameter_type*)m.get(mod);
				VkDeviceSize sz = out->aligned_size(alignment);

				param_total += sz;
			}
		}
	}

	VkDeviceSize grand_total = feature_total + param_total*2 + grad_total + other_total;
	assert(grand_total <= (device_local_total * 9 / 10) &&
	       "plan_memory: total allocation exceeds 90% of device-local memory");

	std::cout << "  Feature maps : " << mb(feature_total) << " MB\n";
	std::cout << "  Parameters   : " << mb(param_total)   << " MB\n";
	std::cout << "  Param Grads  : " << mb(param_total)   << " MB\n";
	std::cout << "  Grad (tmp)   : " << mb(grad_total)    << " MB\n";
	std::cout << "  Other        : " << mb(other_total)   << " MB\n";
	std::cout << "  Total        : " << mb(grand_total)   << " MB  ("
	          << mb(device_local_total * 9 / 10) << " MB budget)\n";

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
		param_grad_buffer = vulkan->create_buffer(param_total, usage);
		std::cout << "  param_grad_buffer created\n";
	}
	if (grad_total > 0)
	{
		grad_tmp_buffer = vulkan->create_buffer(grad_total, usage);
		std::cout << "  grad_buffer created\n";
	}
	if (other_total > 0)
	{
		other_buffer = vulkan->create_buffer(other_total, usage);
		std::cout << "  other_buffer created\n";
	}

	VkDeviceSize feature_t = 0;
	VkDeviceSize param_t = 0;
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
					// grad buffers are reused each backward pass -- all share the same pool, no offset
					out->X = vulkan->create_buffer_slice(grad_tmp_buffer, 0, sz);
				}
				else if (m.flags & REFLECT_VKMOD_COMPONENT_FEAT)
				{
					out->X = vulkan->create_buffer_slice(feature_buffer, feature_t, sz);
					feature_t += sz;
				}
				else
				{
					out->X = vulkan->create_buffer_slice(other_buffer, other_t, sz);
					other_t += sz;
				}
			}
			else if (m_tD == &reflect::parameter<vkBufferResource>::Reflection)
			{
				reflect::parameter<vkBufferResource>* p = (reflect::parameter<vkBufferResource>*)m.get(mod);
				VkDeviceSize sz = p->aligned_size(alignment);

				p->X = vulkan->create_buffer_slice(param_buffer,      param_t, sz);
				p->Y = vulkan->create_buffer_slice(param_grad_buffer, param_t, sz);

				param_t += sz;
			}
		}
	}

	// Pass 3: wire input.X = src_output.X for every input member
	for (VkMod_Reference& ref : Modules)
	{
		Vulkan_Module* mod = ref.X;
		reflect::TypeDescriptor_Struct* td = mod->GetDynamicReflection();
		if (!td) continue;

		for (reflect::Member& m : td->members)
		{
			reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;
			if (!m_tD) continue;

			if (m_tD->inherited_type == &reflect::input_type::Reflection &&
				!(m.flags & REFLECT_VKMOD_COMPONENT_GRAD))
			{
				reflect::input<vkBufferResource>* in = (reflect::input<vkBufferResource>*)m.get(mod);

				assert(in->src_output != nullptr && "plan_memory pass 3: input not connected");

				reflect::output<vkBufferResource>* src = (reflect::output<vkBufferResource>*)in->src_output;

				assert(src->X != nullptr && "plan_memory pass 3: src_output has no buffer");

				in->X = src->X;
			}
		}
	}
}