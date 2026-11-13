#include <irrlicht.h>
#include "vkModules.h"
#include "vkWeightedSumVModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// WeightedSumV_Module
//

REFLECT_VKMOD_BEGIN(WeightedSumV_Module)
	ALIAS("Weighted Sum V")
	INHERIT_FROM(Vulkan_Module)
	// Forward Pass
	REFLECT_VKMOD_FEAT(qkv_tensor)
	REFLECT_VKMOD_FEAT(scores_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	// Backward Pass
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_scores)
	REFLECT_VKMOD_GRAD(grad_qkv)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void WeightedSumV_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void WeightedSumV_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void WeightedSumV_Module::setDimensions()
{
	u32 B = input_dimensions.B;
	u32 M = input_dimensions.C;      // seq_len
	u32 D = input_dimensions.W / 3;  // embed_dim
	u32 K = D / num_heads;           // head_dim
	u32 N = M;                       // key positions == query positions

	output_tensor.dimensions = { B, M, D, 1 };
	grad_scores.dimensions   = { B, num_heads, M, N };
	grad_qkv.dimensions      = input_dimensions;

	pushconstants.M         = M;
	pushconstants.N         = N;
	pushconstants.K         = K;
	pushconstants.D         = D;
	pushconstants.num_heads = num_heads;
}

void WeightedSumV_Module::startup()
{
	VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };

	// qkv(0), scores(1), output(2)
	fwd_pass.bindings.resize(3);
	fwd_pass.bindings[0] = qkv_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = scores_tensor.X->getDescriptorSetLayout(1);
	fwd_pass.bindings[2] = output_tensor.X->getDescriptorSetLayout(2);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);
	fwd_pass.createPipeline(m_device, "shaders/weighted_sum_V.spv", r);
	output_tensor.ready = true;

	// dO(0), scores(1), grad_qkv(2)
	bwd_pass_v.bindings.resize(3);
	bwd_pass_v.bindings[0] = scores_tensor.X->getDescriptorSetLayout(0);
	bwd_pass_v.bindings[1] = grad_input.X->getDescriptorSetLayout(1);
	bwd_pass_v.bindings[2] = grad_qkv.X->getDescriptorSetLayout(2);
	bwd_pass_v.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_v.bindings);
	bwd_pass_v.createPipeline(m_device, "shaders/weighted_sum_V_grad_v.spv", r);

	// dO(0), qkv(1), grad_scores(2)
	bwd_pass_a.bindings.resize(3);
	bwd_pass_a.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass_a.bindings[1] = qkv_tensor.X->getDescriptorSetLayout(1);
	bwd_pass_a.bindings[2] = grad_scores.X->getDescriptorSetLayout(2);
	bwd_pass_a.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_a.bindings);
	bwd_pass_a.createPipeline(m_device, "shaders/weighted_sum_V_grad_a.spv", r);

	log() << "WeightedSumV startup complete\n";
}

void WeightedSumV_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
	bwd_pass_v.cleanup(m_device->getDevice());
	bwd_pass_a.cleanup(m_device->getDevice());
}

//============================================================
// Dispatch helper
//

void WeightedSumV_Module::dispatch_forward(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
	fwd_pass.descriptorSets.resize(1);
	writer.writeBuffer(0, qkv_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, scores_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, output_tensor.X->getDescriptorBufferInfo());
	writer.build(fwd_pass.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	// dispatch (K/16, M/16, B * num_heads) — output tile is [M x K] per (batch, head)
	uint32_t wg_x = pushconstants.K / 16;
	uint32_t wg_y = pushconstants.M / 16;
	uint32_t wg_z = input_dimensions.B * num_heads;
	log() << "WeightedSumV fwd: dispatch ("
	      << wg_x << ", " << wg_y << ", " << wg_z << ")\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
}

void WeightedSumV_Module::dispatch_backward_v(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_v.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_v.descriptorSets.resize(1);
	writer.writeBuffer(0, scores_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, grad_qkv.X->getDescriptorBufferInfo());
	writer.build(bwd_pass_v.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_v.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_v.pipelineLayout, 0, 1, &bwd_pass_v.descriptorSets[0], 0, 0);

	// dispatch (K/16, M/16, B * num_heads) — output tile is [M x K] per (batch, head)
	uint32_t wg_x = pushconstants.K / 16;
	uint32_t wg_y = pushconstants.M / 16;
	uint32_t wg_z = input_dimensions.B * num_heads;
	log() << "WeightedSumV bwd 1: dispatch ("
		<< wg_x << ", " << wg_y << ", " << wg_z << ")\n";

	vkCmdPushConstants(cmd, bwd_pass_v.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
}

void WeightedSumV_Module::dispatch_backward_a(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_a.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_a.descriptorSets.resize(1);
	writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, qkv_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, grad_scores.X->getDescriptorBufferInfo());
	writer.build(bwd_pass_a.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_a.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_a.pipelineLayout, 0, 1, &bwd_pass_a.descriptorSets[0], 0, 0);

	// dispatch (K/16, M/16, B * num_heads) — output tile is [M x K] per (batch, head)
	uint32_t wg_x = pushconstants.N / 16;
	uint32_t wg_y = pushconstants.M / 16;
	uint32_t wg_z = input_dimensions.B * num_heads;
	log() << "WeightedSumV bwd 2: dispatch ("
		<< wg_x << ", " << wg_y << ", " << wg_z << ")\n";

	vkCmdPushConstants(cmd, bwd_pass_a.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
}

//============================================================
// forward / backward
//

void WeightedSumV_Module::forward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_forward(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
}

void WeightedSumV_Module::backward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_backward_v(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(bwd_pass_v.descriptorSets);

	cmd = m_device->beginSingleTimeCommands();
	dispatch_backward_a(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(bwd_pass_a.descriptorSets);
}
