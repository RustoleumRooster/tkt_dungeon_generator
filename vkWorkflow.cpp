
#include "vkModules.h"
#include "vkUtilModules.h"
#include "vkQuantizeModule.h"
#include "vkWorkblock.h"
#include "vkBCELossModule.h"
#include "vkOptimizationModule.h"
#include <vulkan/vulkan.h>
#include "vkGroupNormModule.h"
#include "vkSkipModule.h"
#include "vkLayerNormModule.h"
#include "vkPosEmbedModule.h"
#include <cassert>
#include <chrono>

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

	Convolution_Gelu_Block* conv_block = new Convolution_Gelu_Block();
	conv_block->input_dimension = { 16,128,64,64 };
	conv_block->output_dimension = { 16,128,32,32 };

	Convolution_Gelu_Block* conv_block_2 = new Convolution_Gelu_Block();
	conv_block_2->input_dimension = { 16,128,32,32 };
	conv_block_2->output_dimension = { 16,128,16,16 };

	Convolution_Gelu_Block* conv_block_3 = new Convolution_Gelu_Block();
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

	optimizer = new Optimization_Module();

	// Optimizer goes first so the reverse backward loop runs it last,
	// after all gradients have been accumulated.
	Modules.push_back(VkMod_Reference{ optimizer });
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
	reflect::connect(&quantize->output_tensor, &res_block->skip->input_b);


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

	//=====================================================
	// Backward Pass
	//
	// Pattern: each block gets two connects from the upstream gradient output —
	//   one to the workblock-level input_grad (satisfies plan_memory pass 3 assert)
	//   one to gradient_input() (the actual sub-module entry point for backward compute)
	//

	// Loss → Decoder
	reflect::connect(&bce_loss->gradient_out,         &final_block->input_grad);
	reflect::connect(&bce_loss->gradient_out,         &final_block->gradient_input());

	reflect::connect(&final_block->gradient_output(),  &up_block_3->input_grad);
	reflect::connect(&final_block->gradient_output(),  &up_block_3->gradient_input());

	reflect::connect(&up_block_3->gradient_output(),   &up_block_2->input_grad);
	reflect::connect(&up_block_3->gradient_output(),   &up_block_2->gradient_input());

	reflect::connect(&up_block_2->gradient_output(),   &up_block->input_grad);
	reflect::connect(&up_block_2->gradient_output(),   &up_block->gradient_input());

	reflect::connect(&up_block->gradient_output(),     &res_block->input_grad);
	reflect::connect(&up_block->gradient_output(),     &res_block->gradient_input());

	// Decoder → Quantize
	// res_block.gradient_output() = add_grad.grad_output
	//   = skip.grad_output_b (identity path) + group_norm.grad_output (conv path)
	reflect::connect(&res_block->gradient_output(),    &quantize->grad_input);

	// Quantize → Encoder
	reflect::connect(&quantize->grad_output,           &conv_block_3->input_grad);
	reflect::connect(&quantize->grad_output,           &conv_block_3->gradient_input());

	reflect::connect(&conv_block_3->gradient_output(), &conv_block_2->input_grad);
	reflect::connect(&conv_block_3->gradient_output(), &conv_block_2->gradient_input());

	reflect::connect(&conv_block_2->gradient_output(), &conv_block->input_grad);
	reflect::connect(&conv_block_2->gradient_output(), &conv_block->gradient_input());

	items.clear();
	for (VkMod_Reference& ref : Modules)
		items.push_back(ref.X);
}

void Vulkan_Workflow::make_transformer_workflow()
{
	//=================================================
	//Encoder
	//

	Create_Tensor_Module* create_images = new Create_Tensor_Module();
	create_images->dimensions = { 16,128,64,64 };

	Convolution_Gelu_Block* conv_block = new Convolution_Gelu_Block();
	conv_block->input_dimension = { 16,128,64,64 };
	conv_block->output_dimension = { 16,128,32,32 };

	Convolution_Gelu_Block* conv_block_2 = new Convolution_Gelu_Block();
	conv_block_2->input_dimension = { 16,128,32,32 };
	conv_block_2->output_dimension = { 16,128,16,16 };

	Convolution_Gelu_Block* conv_block_3 = new Convolution_Gelu_Block();
	conv_block_3->input_dimension = { 16,128,16,16 };
	conv_block_3->output_dimension = { 16,128,8,8 };

	Quantize_Module* quantize = new Quantize_Module();
	quantize->input_dimensions = { 16,128,8,8 };
	quantize->codebook_size = { 1,1,512,128 };

	//=================================================
	// Transformer
	//

	PosEmbed_Module* pos_embed = new PosEmbed_Module();
	pos_embed->input_dimensions = { 16,64,1,128 };

	LayerNorm_Module* layer_norm = new LayerNorm_Module();
	layer_norm->input_dimensions = { 16,64,1,128 };

	optimizer = new Optimization_Module();

	// Optimizer goes first so the reverse backward loop runs it last,
	// after all gradients have been accumulated.
	Modules.push_back(VkMod_Reference{ optimizer });
	Modules.push_back(VkMod_Reference{ create_images });
	Modules.push_back(VkMod_Reference{ conv_block });
	Modules.push_back(VkMod_Reference{ conv_block_2 });
	Modules.push_back(VkMod_Reference{ conv_block_3 });
	Modules.push_back(VkMod_Reference{ quantize });
	Modules.push_back(VkMod_Reference{ layer_norm });
	Modules.push_back(VkMod_Reference{ pos_embed });

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
	// Transformer
	//

	reflect::connect(&quantize->output_tensor, &pos_embed->input_tensor);

	reflect::connect(&pos_embed->output_tensor, &layer_norm->input_tensor);

	

	//=====================================================
	// Loss
	//

	

	//=====================================================
	// Backward Pass
	//
	// Pattern: each block gets two connects from the upstream gradient output —
	//   one to the workblock-level input_grad (satisfies plan_memory pass 3 assert)
	//   one to gradient_input() (the actual sub-module entry point for backward compute)
	//

	// Loss → Decoder
	//TODO

	// Quantize → Encoder
	reflect::connect(&quantize->grad_output, &conv_block_3->input_grad);
	reflect::connect(&quantize->grad_output, &conv_block_3->gradient_input());

	reflect::connect(&conv_block_3->gradient_output(), &conv_block_2->input_grad);
	reflect::connect(&conv_block_3->gradient_output(), &conv_block_2->gradient_input());

	reflect::connect(&conv_block_2->gradient_output(), &conv_block->input_grad);
	reflect::connect(&conv_block_2->gradient_output(), &conv_block->gradient_input());

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

	if (optimizer && param_total_bytes > 0)
	{
		optimizer->params_buf             = param_buffer;
		optimizer->grads_buf              = param_grad_buffer;
		optimizer->m_buf                  = param_m_buffer;
		optimizer->v_buf                  = param_v_buffer;
		optimizer->pushconstants.n_elements = (u32)(param_total_bytes / sizeof(float));
	}

	for (VkMod_Reference& mod : Modules)
		mod.X->initialize_parameters();

	// Warn about any reflected parameters that were not initialized.
	for (VkMod_Reference& ref : Modules)
	{
		Vulkan_Module* mod = ref.X;
		reflect::TypeDescriptor_Struct* td = mod->GetDynamicReflection();
		if (!td) continue;

		for (reflect::Member& m : td->members)
		{
			reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;
			if (!m_tD) continue;

			if (m_tD->inherited_type == &reflect::parameter_type::Reflection)
			{
				reflect::parameter_type* p = (reflect::parameter_type*)m.get(mod);
				if (!p->initialized)
					std::cout << "[WARNING] Parameter '" << m.name
					          << "' on module '" << td->name << "' was not initialized.\n";
			}
		}
	}

	for (VkMod_Reference& mod : Modules)
	{
		//if (!mod.X->is_submodule)
			mod.X->startup();
	}

	const int n_passes = 1;

	for (int pass = 0; pass < n_passes; pass++)
	{
		std::cout << "\n[Pass " << pass + 1 << " / " << n_passes << "]\n";

		// Reset completion flags so ready_forward/ready_backward pass again.
		for (VkMod_Reference& mod : Modules)
		{
			mod.X->forward_pass_complete  = false;
			mod.X->backward_pass_complete = false;
		}

		// Advance Adam bias-correction terms each step after the first.
		if (optimizer && pass > 0)
		{
			optimizer->pushconstants.B1_t *= 0.9f;
			optimizer->pushconstants.B2_t *= 0.999f;
		}

		auto pass_start = std::chrono::high_resolution_clock::now();

		for (VkMod_Reference& mod : Modules)
		{
			if (mod.X->is_submodule == false)
			{
				assert(mod.X->ready_forward() && "Module not ready...");
				auto t0 = std::chrono::high_resolution_clock::now();
				mod.X->forward();
				auto t1 = std::chrono::high_resolution_clock::now();
				mod.X->elapsed_forward = std::chrono::duration<float, std::milli>(t1 - t0).count();
				mod.X->total_forward  += mod.X->elapsed_forward;
				mod.X->forward_pass_complete = true;
			}
		}
		if(false)	//backward pass disabled
		for (int i = (int)Modules.size() - 1; i >= 0; i--)
		{
			Vulkan_Module* mod = Modules[i].X;
			if (mod->is_submodule == false)
			{
				assert(mod->ready_backward() && "Module not ready for backward...");
				auto t0 = std::chrono::high_resolution_clock::now();
				mod->backward();
				auto t1 = std::chrono::high_resolution_clock::now();
				mod->elapsed_backward = std::chrono::duration<float, std::milli>(t1 - t0).count();
				mod->total_backward  += mod->elapsed_backward;
				mod->backward_pass_complete = true;
			}
		}

		auto pass_end = std::chrono::high_resolution_clock::now();
		float pass_ms = std::chrono::duration<float, std::milli>(pass_end - pass_start).count();
		std::cout << "[Pass " << pass + 1 << " complete] " << pass_ms << " ms, (average " << pass_ms / float(pass+1) <<" ms)\n";
	}

	for (VkMod_Reference& mod : Modules)
	{
		//if (!mod.X->is_submodule)
			mod.X->cleanup_passes();
	}

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
				VkDeviceSize sz = (m.flags & REFLECT_VKMOD_COMPONENT_UINT_TYPE)
				                ? out->aligned_size<u32>(alignment)
				                : out->aligned_size<f32>(alignment);

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
				VkDeviceSize sz = out->aligned_size<f32>(alignment);

				param_total += sz;
			}
		}
	}

	param_total_bytes = param_total;
	VkDeviceSize grand_total = feature_total + param_total*4 + grad_total*2 + other_total;
	assert(grand_total <= (device_local_total * 9 / 10) &&
	       "plan_memory: total allocation exceeds 90% of device-local memory");

	std::cout << "  Feature maps : " << mb(feature_total) << " MB\n";
	std::cout << "  Parameters   : " << mb(param_total)   << " MB\n";
	std::cout << "  Param Grads  : " << mb(param_total)   << " MB\n";
	std::cout << "  Param M (1st): " << mb(param_total)   << " MB\n";
	std::cout << "  Param V (2nd): " << mb(param_total)   << " MB\n";
	std::cout << "  Grad (ping)  : " << mb(grad_total)    << " MB\n";
	std::cout << "  Grad (pong)  : " << mb(grad_total)    << " MB\n";
	std::cout << "  Other        : " << mb(other_total)   << " MB\n";
	std::cout << "  Total        : " << mb(grand_total)   << " MB  ("
	          << mb(device_local_total * 9 / 10) << " MB budget)\n\n";

	auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	if (feature_total > 0)
	{
		feature_buffer = vulkan->create_buffer(feature_total, usage);
	}
	if (param_total > 0)
	{
		param_buffer = vulkan->create_buffer(param_total, usage);
		param_grad_buffer = vulkan->create_buffer(param_total, usage);
		param_m_buffer = vulkan->create_buffer(param_total, usage);
		param_v_buffer = vulkan->create_buffer(param_total, usage);
	}
	if (grad_total > 0)
	{
		grad_tmp_buffer   = vulkan->create_buffer(grad_total, usage);
		grad_tmp_buffer_b = vulkan->create_buffer(grad_total, usage);
	}
	if (other_total > 0)
	{
		other_buffer = vulkan->create_buffer(other_total, usage);
	}

	VkDeviceSize      feature_t        = 0;
	VkDeviceSize      param_t          = 0;
	VkDeviceSize      other_t          = 0;
	int               grad_ping        = 0;     // alternates 0/1 to assign ping vs pong buffer
	vkBufferResource* last_grad_buf    = NULL;  // buffer used by the most recent grad allocation
	VkDeviceSize      last_grad_sz     = 0;     // aligned size of that allocation

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
				VkDeviceSize sz = (m.flags & REFLECT_VKMOD_COMPONENT_UINT_TYPE)
				                ? out->aligned_size<u32>(alignment)
				                : out->aligned_size<f32>(alignment);

				if (m.flags & REFLECT_VKMOD_COMPONENT_GRAD)
				{
					// Special case: Skip_Module writes grad_output_a and grad_output_b in
					// the same dispatch. Pack grad_output_b immediately after grad_output_a
					// in the same buffer so both outputs share one ping/pong slot and the
					// input (grad_input) stays on the opposite buffer.
					if (td == &Skip_Module::Reflection && strcmp(m.name, "grad_output_b") == 0)
					{
						assert(last_grad_buf && "Skip grad_output_b has no preceding grad_output_a");
						assert(last_grad_sz + sz <= grad_total && "Skip grad_output_b overflows grad_tmp buffer");
						out->X = vulkan->create_buffer_slice(last_grad_buf, last_grad_sz, sz);
					}
					else
					{
						// Ping-pong: alternate between two same-sized grad buffers so that
						// consecutive backward steps never bind the same VkBuffer for both
						// read (grad_input) and write (grad_output) in the same dispatch.
						vkBufferResource* buf = (grad_ping % 2 == 0) ? grad_tmp_buffer : grad_tmp_buffer_b;
						out->X = vulkan->create_buffer_slice(buf, 0, sz);
						last_grad_buf = buf;
						last_grad_sz  = sz;
						grad_ping++;
					}
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
				VkDeviceSize sz = p->aligned_size<f32>(alignment);

				p->X = vulkan->create_buffer_slice(param_buffer,      param_t, sz);
				p->Y = vulkan->create_buffer_slice(param_grad_buffer, param_t, sz);
				p->M = vulkan->create_buffer_slice(param_m_buffer,    param_t, sz);
				p->V = vulkan->create_buffer_slice(param_v_buffer,    param_t, sz);

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
			}/* BACKWARD PASS TEMPORARILY DISABLED
			else if (m_tD->inherited_type == &reflect::input_type::Reflection &&
				(m.flags & REFLECT_VKMOD_COMPONENT_GRAD))
			{
				reflect::input<vkBufferResource>* in = (reflect::input<vkBufferResource>*)m.get(mod);

				assert(in->src_output != nullptr && "plan_memory pass 3: input gradient not connected");

				reflect::output<vkBufferResource>* src = (reflect::output<vkBufferResource>*)in->src_output;

				assert(src->X != nullptr && "plan_memory pass 3: gradient src_output has no buffer");

				in->X = src->X;
			}*/
		}

	}
}