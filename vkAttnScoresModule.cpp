#include <irrlicht.h>
#include "vkModules.h"
#include "vkAttnScoresModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Attn_Scores_Module
//

REFLECT_VKMOD_BEGIN(Attn_Scores_Module)
	ALIAS("Attention Scores")
	INHERIT_FROM(Vulkan_Module)
	// Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	// Backward Pass
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void Attn_Scores_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Attn_Scores_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Attn_Scores_Module::setDimensions()
{
	u32 B = input_dimensions.B;
	u32 M = input_dimensions.C;        // seq_len = 64
	u32 D = input_dimensions.W / 3;    // embed_dim (W = 3*D from QKV)
	u32 K = D / num_heads;             // head_dim

	output_tensor.dimensions = { B, num_heads, M, M };
	grad_output.dimensions   = input_dimensions;

	pushconstants.M         = M;
	pushconstants.N         = M;
	pushconstants.K         = K;
	pushconstants.D         = D;
	pushconstants.num_heads = num_heads;
}

void Attn_Scores_Module::startup()
{
	VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };

	// fwd_pass: qkv(0), scores(1)
	fwd_pass.bindings.resize(2);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = output_tensor.X->getDescriptorSetLayout(1);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);
	fwd_pass.createPipeline(m_device, "shaders/attn_scores.spv", r);
	output_tensor.ready = true;

	// bwd_pass_q: dS(0), qkv(1), grad_qkv(2) — writes Q slice of grad_output
	bwd_pass_q.bindings.resize(3);
	bwd_pass_q.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass_q.bindings[1] = input_tensor.X->getDescriptorSetLayout(1);
	bwd_pass_q.bindings[2] = grad_output.X->getDescriptorSetLayout(2);
	bwd_pass_q.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_q.bindings);
	bwd_pass_q.createPipeline(m_device, "shaders/attn_scores_grad_q.spv", r);

	// bwd_pass_k: dS(0), qkv(1), grad_qkv(2) — writes K slice of grad_output
	bwd_pass_k.bindings.resize(3);
	bwd_pass_k.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass_k.bindings[1] = input_tensor.X->getDescriptorSetLayout(1);
	bwd_pass_k.bindings[2] = grad_output.X->getDescriptorSetLayout(2);
	bwd_pass_k.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_k.bindings);
	bwd_pass_k.createPipeline(m_device, "shaders/attn_scores_grad_k.spv", r);
	grad_output.ready = true;

	log() << "Attn_Scores startup complete\n";
}

void Attn_Scores_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
	bwd_pass_q.cleanup(m_device->getDevice());
	bwd_pass_k.cleanup(m_device->getDevice());
}

//============================================================
// Dispatch helper
//

void Attn_Scores_Module::dispatch_forward(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
	fwd_pass.descriptorSets.resize(1);
	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, output_tensor.X->getDescriptorBufferInfo());
	writer.build(fwd_pass.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	// dispatch (N/16, M/16, B * n_heads) — one tile per workgroup
	uint32_t wg_x = pushconstants.N / 16;
	uint32_t wg_y = pushconstants.M / 16;
	uint32_t wg_z = input_dimensions.B * num_heads;
	log() << "Attn_Scores dispatch ("
	      << wg_x << ", " << wg_y << ", " << wg_z << ")\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
}

void Attn_Scores_Module::dispatch_backward_q(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_q.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_q.descriptorSets.resize(1);
	writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, grad_output.X->getDescriptorBufferInfo());
	writer.build(bwd_pass_q.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_q.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_q.pipelineLayout, 0, 1, &bwd_pass_q.descriptorSets[0], 0, 0);

	// dQ_h = dS_h @ K_h — output [M x K] per (batch, head): dispatch (K/16, M/16, B * num_heads)
	uint32_t wg_x = pushconstants.K / 16;
	uint32_t wg_y = pushconstants.M / 16;
	uint32_t wg_z = input_dimensions.B * num_heads;
	log() << "Attn_Scores bwd Q: dispatch ("
	      << wg_x << ", " << wg_y << ", " << wg_z << ")\n";

	vkCmdPushConstants(cmd, bwd_pass_q.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
}

void Attn_Scores_Module::dispatch_backward_k(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_k.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_k.descriptorSets.resize(1);
	writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, grad_output.X->getDescriptorBufferInfo());
	writer.build(bwd_pass_k.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_k.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_k.pipelineLayout, 0, 1, &bwd_pass_k.descriptorSets[0], 0, 0);

	// dK_h = dS_h^T @ Q_h — output [N x K] per (batch, head): dispatch (K/16, N/16, B * num_heads)
	uint32_t wg_x = pushconstants.K / 16;
	uint32_t wg_y = pushconstants.N / 16;
	uint32_t wg_z = input_dimensions.B * num_heads;
	log() << "Attn_Scores bwd K: dispatch ("
	      << wg_x << ", " << wg_y << ", " << wg_z << ")\n";

	vkCmdPushConstants(cmd, bwd_pass_k.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
}

//============================================================
// forward / backward
//

void Attn_Scores_Module::forward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_forward(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
}

void Attn_Scores_Module::backward()
{
	// dQ_h = dS_h @ K_h / sqrt(K)    → Q slice of grad_output
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_backward_q(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(bwd_pass_q.descriptorSets);

	// dK_h = dS_h^T @ Q_h / sqrt(K)  → K slice of grad_output
	cmd = m_device->beginSingleTimeCommands();
	dispatch_backward_k(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(bwd_pass_k.descriptorSets);
}
