#include <irrlicht.h>
#include "vkModules.h"
#include "vkQuantizeModule.h"
#include "soa.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"
#include <chrono>

std::chrono::steady_clock::time_point q_startTime;
std::chrono::steady_clock::time_point q_currentTime;
float q_passedTime;

#define Q_START_TIMER() q_startTime = std::chrono::high_resolution_clock::now();
#define Q_PRINT_TIMER() q_currentTime = std::chrono::high_resolution_clock::now(); \
    q_passedTime = std::chrono::duration<float, std::chrono::seconds::period>(q_currentTime - q_startTime).count(); \
    log() << q_passedTime << "\n";

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Quantize Module
//

REFLECT_VKMOD_BEGIN(Quantize_Module)
	ALIAS("Quantize Layer")
	INHERIT_FROM(Vulkan_Module)
	//Forward Pass (features)
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	REFLECT_VKMOD_FEAT(indices_buffer)
		REFLECT_VKMOD_UINT_TYPE()
	REFLECT_VKMOD_FEAT(loss_buffer)
	REFLECT_VKMOD_FEAT(commit_loss) //single scalar value
	REFLECT_VKMOD_PARAM(codebook)
	REFLECT_VKMOD_PARAM(ema_count)
	REFLECT_VKMOD_PARAM(ema_sum)
	//Backward Pass (gradients)
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void Quantize_Module::Pass::createPipeline(MyDevice* device, const char* spv,
                                           VkPushConstantRange pushconstant)
{
	VkPipelineLayoutCreateInfo info{};
	info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	info.setLayoutCount         = 1;
	info.pSetLayouts            = &descriptorSetLayout->getDescriptorSetLayout();
	info.pPushConstantRanges    = &pushconstant;
	info.pushConstantRangeCount = 1;

	vkCreatePipelineLayout(device->getDevice(), &info, nullptr, &pipelineLayout);
	pipeline = new ComputePipeline(device, spv, pipelineLayout);
}

void Quantize_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Quantize_Module::initialize_parameters()
{
	// Codebook layout: { 1, 1, n_codes, dim }
	// Initialize each code vector with N(0, 1/sqrt(dim)) so they land on the
	// unit hypersphere in expectation — a reasonable starting spread for VQ.
	u32 n_codes = codebook.dimensions.H;
	u32 dim     = codebook.dimensions.W;
	u32 n       = n_codes * dim;

	float std_dev = 1.0f / std::sqrt(float(dim));

	std::mt19937 rng(std::random_device{}());
	std::normal_distribution<float> dist(0.0f, std_dev);

	std::vector<float> data(n);
	for (u32 i = 0; i < n; i++)
		data[i] = dist(rng);

	upload_to_buffer(codebook.X, data);
	codebook.initialized = true;
	log() << "Quantize codebook initialized (" << n_codes << " codes, dim=" << dim << ", std=" << std_dev << ")\n";

	// Zero-initialize EMA state
	std::vector<float> zeros_k(n_codes, 0.0f);
	std::vector<float> zeros_kd(n_codes * dim, 0.0f);
	upload_to_buffer(ema_count.X, zeros_k);
	upload_to_buffer(ema_sum.X,   zeros_kd);
	ema_count.initialized = true;
	ema_sum.initialized   = true;
}

void Quantize_Module::setDimensions()
{
	codebook.dimensions      = codebook_size;
	output_tensor.dimensions = input_dimensions;
	// one u32 index per spatial position per batch element
	indices_buffer.dimensions = { input_dimensions.B, 1, input_dimensions.H, input_dimensions.W };
	// one f32 commitment distance per spatial position per batch element
	loss_buffer.dimensions    = { input_dimensions.B, 1, input_dimensions.H, input_dimensions.W };
	// scalar mean over all positions
	commit_loss.dimensions    = { 1, 1, 1, 1 };
	commit_pushconstants.n_elements  = input_dimensions.B * input_dimensions.H * input_dimensions.W;
	bwd_pushconstants.n_elements     = input_dimensions.B * input_dimensions.H * input_dimensions.W;
	bwd_pushconstants.n_channels     = input_dimensions.C;
	grad_output.dimensions           = input_dimensions;
	pushconstants.n_vectors          = codebook_size.H;

	// EMA state buffers
	ema_count.dimensions = { 1, 1, codebook_size.H, 1 };           // float[K]
	ema_sum.dimensions   = { 1, 1, codebook_size.H, codebook_size.W }; // float[K*D]
	ema_pushconstants.K            = codebook_size.H;
	ema_pushconstants.D            = codebook_size.W;
	ema_pushconstants.n_positions  = input_dimensions.B * input_dimensions.H * input_dimensions.W;
	ema_pushconstants.HW           = input_dimensions.H * input_dimensions.W;
}

void Quantize_Module::forward()
{
	// bindings: input_tensor(0), codebook(1), output_tensor(2), indices_buffer(3), loss_buffer(4)
	fwd_pass.bindings.resize(5);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = codebook.X->getDescriptorSetLayout(1);
	fwd_pass.bindings[2] = output_tensor.X->getDescriptorSetLayout(2);
	fwd_pass.bindings[3] = indices_buffer.X->getDescriptorSetLayout(3);
	fwd_pass.bindings[4] = loss_buffer.X->getDescriptorSetLayout(4);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	fwd_pass.createPipeline(m_device, "shaders/quantize.spv", push_constant);

	output_tensor.ready = true;

	{
		MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
		fwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, codebook.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, output_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(3, indices_buffer.X->getDescriptorBufferInfo());
		writer.writeBuffer(4, loss_buffer.X->getDescriptorBufferInfo());
		writer.build(fwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = input_dimensions.B * input_dimensions.H * input_dimensions.W;

	log() << "Quantize forward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	indices_buffer.ready = true;
	loss_buffer.ready    = true;
	fwd_pass.cleanup(m_device->getDevice());

	// Commit Loss Pass: mean reduction over loss_buffer -> commit_loss scalar
	// bindings: loss_buffer(0), commit_loss(1)
	commit_loss_pass.bindings.resize(2);
	commit_loss_pass.bindings[0] = loss_buffer.X->getDescriptorSetLayout(0);
	commit_loss_pass.bindings[1] = commit_loss.X->getDescriptorSetLayout(1);
	commit_loss_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, commit_loss_pass.bindings);

	VkPushConstantRange commit_push_constant;
	commit_push_constant.offset     = 0;
	commit_push_constant.size       = sizeof(commit_pushconstant_struct);
	commit_push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	commit_loss_pass.createPipeline(m_device, "shaders/commit_loss.spv", commit_push_constant);

	{
		MyDescriptorWriter writer(*commit_loss_pass.descriptorSetLayout, *m_DescriptorPool);
		commit_loss_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, loss_buffer.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, commit_loss.X->getDescriptorBufferInfo());
		writer.build(commit_loss_pass.descriptorSets[0]);
	}

	{
		VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

		// zero the accumulator before dispatching
		vkCmdFillBuffer(commandBuffer, commit_loss.X->Buffer, 0, VK_WHOLE_SIZE, 0);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
			commit_loss_pass.pipeline->getPipeline());

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
			commit_loss_pass.pipelineLayout, 0, 1, &commit_loss_pass.descriptorSets[0], 0, 0);

		uint32_t n_WorkGroups_x = (commit_pushconstants.n_elements + 255) / 256;

		log() << "Commit loss: " << n_WorkGroups_x << " workgroups\n";

		vkCmdPushConstants(commandBuffer, commit_loss_pass.pipelineLayout,
			VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(commit_pushconstant_struct), &commit_pushconstants);

		vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

		m_device->endSingleTimeCommands(commandBuffer);
	}

	m_DescriptorPool->freeDescriptorsSets(commit_loss_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	commit_loss.ready = true;
	commit_loss_pass.cleanup(m_device->getDevice());
}

void Quantize_Module::reset_dead_codes()
{
	u32 K           = ema_pushconstants.K;
	u32 D           = ema_pushconstants.D;
	u32 n_positions = ema_pushconstants.n_positions;  // B * H * W
	u32 HW          = ema_pushconstants.HW;

	// 1. Read ema_count back from GPU
	VkDeviceSize count_bytes = K * sizeof(float);
	std::vector<float> counts(K);
	{
		MyBufferObject staging(m_device, count_bytes, 1,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);
		m_device->copyBuffer(ema_count.X->Buffer, staging.getBuffer(), count_bytes);
		staging.readFromBuffer(counts.data());
	}

	// 2. Find dead codes
	std::vector<u32> dead;
	for (u32 k = 0; k < K; k++)
		if (counts[k] < 2.0f)
			dead.push_back(k);

	if (dead.empty()) return;

	log() << "reset_dead_codes: " << dead.size() << " / " << K << " below threshold\n";

	// 3. Read z_e, codebook, and ema_sum from GPU
	VkDeviceSize ze_bytes = (VkDeviceSize)n_positions * D * sizeof(float);
	VkDeviceSize cb_bytes = (VkDeviceSize)K * D * sizeof(float);

	std::vector<float> ze(n_positions * D);
	std::vector<float> codebook_data(K * D);
	std::vector<float> ema_sum_data(K * D);

	{
		MyBufferObject staging(m_device, ze_bytes, 1,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);
		m_device->copyBuffer(input_tensor.X->Buffer, staging.getBuffer(), ze_bytes);
		staging.readFromBuffer(ze.data());
	}
	{
		MyBufferObject staging(m_device, cb_bytes, 1,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);
		m_device->copyBuffer(codebook.X->Buffer, staging.getBuffer(), cb_bytes);
		staging.readFromBuffer(codebook_data.data());
	}
	{
		MyBufferObject staging(m_device, cb_bytes, 1,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);
		m_device->copyBuffer(ema_sum.X->Buffer, staging.getBuffer(), cb_bytes);
		staging.readFromBuffer(ema_sum_data.data());
	}

	// 4. Overwrite each dead code with a randomly sampled z_e vector
	for (u32 k : dead)
	{
		u32 pos = (u32)(random_number() % n_positions);
		u32 n   = pos / HW;
		u32 hw  = pos % HW;

		for (u32 d = 0; d < D; d++)
		{
			float val           = ze[n * D * HW + d * HW + hw];
			codebook_data[k * D + d] = val;
			ema_sum_data[k * D + d]  = val;
		}
		counts[k] = 1.0f;
	}

	// 5. Upload patched buffers back to GPU
	upload_to_buffer(codebook.X,  codebook_data);
	upload_to_buffer(ema_sum.X,   ema_sum_data);
	upload_to_buffer(ema_count.X, counts);
}

void Quantize_Module::backward()
{
	// bindings: grad_input(0)   — dL/d_quantized from downstream
	//           input_tensor(1) — z_e, encoder output (for commitment gradient)
	//           output_tensor(2)— e, quantized output  (for commitment gradient)
	//           grad_output(3)  — dL/dz_e = STE + 2*beta*(z_e - e)/n_elements
	bwd_pass.bindings.resize(4);
	bwd_pass.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass.bindings[1] = input_tensor.X->getDescriptorSetLayout(1);
	bwd_pass.bindings[2] = output_tensor.X->getDescriptorSetLayout(2);
	bwd_pass.bindings[3] = grad_output.X->getDescriptorSetLayout(3);
	bwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(bwd_pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	bwd_pass.createPipeline(m_device, "shaders/quantize_grad.spv", push_constant);

	grad_output.ready = true;

	{
		MyDescriptorWriter writer(*bwd_pass.descriptorSetLayout, *m_DescriptorPool);
		bwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, input_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, output_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(3, grad_output.X->getDescriptorBufferInfo());
		writer.build(bwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipelineLayout, 0, 1, &bwd_pass.descriptorSets[0], 0, 0);

	uint32_t total_elements  = bwd_pushconstants.n_elements * bwd_pushconstants.n_channels;
	uint32_t n_WorkGroups_x  = (total_elements + 255) / 256;

	log() << "Quantize backward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, bwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bwd_pushconstant_struct), &bwd_pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(bwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	bwd_pass.cleanup(m_device->getDevice());

	// EMA codebook update pass
	// bindings: codebook(0), ema_count(1), ema_sum(2), indices_buffer(3), input_tensor/z_e(4)
	ema_pass.bindings.resize(5);
	ema_pass.bindings[0] = codebook.X->getDescriptorSetLayout(0);
	ema_pass.bindings[1] = ema_count.X->getDescriptorSetLayout(1);
	ema_pass.bindings[2] = ema_sum.X->getDescriptorSetLayout(2);
	ema_pass.bindings[3] = indices_buffer.X->getDescriptorSetLayout(3);
	ema_pass.bindings[4] = input_tensor.X->getDescriptorSetLayout(4);
	ema_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, ema_pass.bindings);

	VkPushConstantRange ema_push_constant;
	ema_push_constant.offset     = 0;
	ema_push_constant.size       = sizeof(ema_pushconstant_struct);
	ema_push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	ema_pass.createPipeline(m_device, "shaders/quantize_ema.spv", ema_push_constant);

	{
		MyDescriptorWriter writer(*ema_pass.descriptorSetLayout, *m_DescriptorPool);
		ema_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, codebook.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, ema_count.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, ema_sum.X->getDescriptorBufferInfo());
		writer.writeBuffer(3, indices_buffer.X->getDescriptorBufferInfo());
		writer.writeBuffer(4, input_tensor.X->getDescriptorBufferInfo());
		writer.build(ema_pass.descriptorSets[0]);
	}

	{
		VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
			ema_pass.pipeline->getPipeline());

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
			ema_pass.pipelineLayout, 0, 1, &ema_pass.descriptorSets[0], 0, 0);

		log() << "Quantize EMA: " << ema_pushconstants.K << " workgroups\n";

		vkCmdPushConstants(commandBuffer, ema_pass.pipelineLayout,
			VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ema_pushconstant_struct), &ema_pushconstants);

		vkCmdDispatch(commandBuffer, ema_pushconstants.K, 1, 1);

		m_device->endSingleTimeCommands(commandBuffer);
	}

	m_DescriptorPool->freeDescriptorsSets(ema_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	ema_pass.cleanup(m_device->getDevice());
}
