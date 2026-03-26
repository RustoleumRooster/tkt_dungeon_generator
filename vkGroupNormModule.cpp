#include <irrlicht.h>
#include "vkModules.h"
#include "vkGroupNormModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// GroupNorm Module
//

REFLECT_VKMOD_BEGIN(GroupNorm_Module)
	ALIAS("Group Norm Layer")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_MEMBER(mean_buffer)
	REFLECT_VKMOD_MEMBER(invstd_buffer)
	REFLECT_VKMOD_MEMBER(int_sums_buffer)
	REFLECT_VKMOD_PARAM(parameters)
	REFLECT_VKMOD_FEAT(output_tensor)
	REFLECT_VKMOD_FEAT(xnorm_buffer)
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void GroupNorm_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void GroupNorm_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void GroupNorm_Module::setDimensions()
{
	mean_buffer.dimensions     = { 1, 1, input_dimensions.B, pushconstants.num_groups };
	invstd_buffer.dimensions   = { 1, 1, input_dimensions.B, pushconstants.num_groups };
	int_sums_buffer.dimensions = { 1, 1, input_dimensions.B, pushconstants.num_groups * 2 };
	parameters.dimensions      = { 1, 1, 2, input_dimensions.C }; // gamma, beta per channel
	output_tensor.dimensions   = input_dimensions;
	xnorm_buffer.dimensions    = input_dimensions;
	grad_output.dimensions     = input_dimensions;

	pushconstants.n = input_dimensions.B;
	pushconstants.c = input_dimensions.C;
	pushconstants.h = input_dimensions.H;
	pushconstants.w = input_dimensions.W;
}

void GroupNorm_Module::initialize_parameters()
{
	// parameters layout: { 1, 1, 2, C }
	//   row 0: gamma (scale) — initialize to 1
	//   row 1: beta  (shift) — initialize to 0
	u32 C = parameters.dimensions.W;

	std::vector<float> data(2 * C, 0.0f);
	for (u32 i = 0; i < C; i++)
		data[i] = 1.0f;  // gamma
	// beta already 0 from fill

	upload_to_buffer(parameters.X, data);
	parameters.initialized = true;
	log() << "GroupNorm parameters initialized (gamma=1, beta=0)\n";
}

void GroupNorm_Module::startup()
{
	VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };

	// --- fwd_pass_A: groupnorm.spv — computes mean/invstd ---
	// bindings: input_tensor(0), mean_buffer(1), invstd_buffer(2)
	fwd_pass_A.bindings.resize(3);
	fwd_pass_A.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass_A.bindings[1] = mean_buffer.X->getDescriptorSetLayout(1);
	fwd_pass_A.bindings[2] = invstd_buffer.X->getDescriptorSetLayout(2);
	fwd_pass_A.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass_A.bindings);
	fwd_pass_A.createPipeline(m_device, "shaders/groupnorm.spv", r);
	mean_buffer.ready   = true;
	invstd_buffer.ready = true;

	// --- fwd_pass_B: groupnorm2.spv — applies gamma/beta, writes xnorm ---
	// bindings: input_tensor(0), parameters(1), mean_buffer(2), invstd_buffer(3),
	//           output_tensor(4), xnorm_buffer(5)
	fwd_pass_B.bindings.resize(6);
	fwd_pass_B.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass_B.bindings[1] = parameters.X->getDescriptorSetLayout(1);
	fwd_pass_B.bindings[2] = mean_buffer.X->getDescriptorSetLayout(2);
	fwd_pass_B.bindings[3] = invstd_buffer.X->getDescriptorSetLayout(3);
	fwd_pass_B.bindings[4] = output_tensor.X->getDescriptorSetLayout(4);
	fwd_pass_B.bindings[5] = xnorm_buffer.X->getDescriptorSetLayout(5);
	fwd_pass_B.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass_B.bindings);
	fwd_pass_B.createPipeline(m_device, "shaders/groupnorm2.spv", r);
	output_tensor.ready = true;
	xnorm_buffer.ready  = true;

	// --- bwd_pass_1: groupnorm_grad_1.spv — dγ and dβ gradients ---
	// bindings: grad_input(0), xnorm_buffer(1), param_grad(2)
	bwd_pass_1.bindings.resize(3);
	bwd_pass_1.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass_1.bindings[1] = xnorm_buffer.X->getDescriptorSetLayout(1);
	bwd_pass_1.bindings[2] = parameters.Y->getDescriptorSetLayout(2);
	bwd_pass_1.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_1.bindings);
	bwd_pass_1.createPipeline(m_device, "shaders/groupnorm_grad_1.spv", r);

	// --- bwd_pass_2: groupnorm_grad_2.spv — intermediate sums per (n, g) ---
	// bindings: grad_input(0), xnorm_buffer(1), int_sums_buffer(2)
	bwd_pass_2.bindings.resize(3);
	bwd_pass_2.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass_2.bindings[1] = xnorm_buffer.X->getDescriptorSetLayout(1);
	bwd_pass_2.bindings[2] = int_sums_buffer.X->getDescriptorSetLayout(2);
	bwd_pass_2.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_2.bindings);
	bwd_pass_2.createPipeline(m_device, "shaders/groupnorm_grad_2.spv", r);
	int_sums_buffer.ready = true;

	// --- bwd_pass_4: groupnorm_grad_3.spv — dL/dx ---
	// bindings: grad_input(0), xnorm_buffer(1), invstd_buffer(2),
	//           int_sums_buffer(3), parameters(4), grad_output(5)
	bwd_pass_4.bindings.resize(6);
	bwd_pass_4.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass_4.bindings[1] = xnorm_buffer.X->getDescriptorSetLayout(1);
	bwd_pass_4.bindings[2] = invstd_buffer.X->getDescriptorSetLayout(2);
	bwd_pass_4.bindings[3] = int_sums_buffer.X->getDescriptorSetLayout(3);
	bwd_pass_4.bindings[4] = parameters.X->getDescriptorSetLayout(4);
	bwd_pass_4.bindings[5] = grad_output.X->getDescriptorSetLayout(5);
	bwd_pass_4.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_4.bindings);
	bwd_pass_4.createPipeline(m_device, "shaders/groupnorm_grad_3.spv", r);
	grad_output.ready = true;

	log() << "GroupNorm startup complete\n";
}

void GroupNorm_Module::cleanup_passes()
{
	fwd_pass_A.cleanup(m_device->getDevice());
	fwd_pass_B.cleanup(m_device->getDevice());
	bwd_pass_1.cleanup(m_device->getDevice());
	bwd_pass_2.cleanup(m_device->getDevice());
	bwd_pass_4.cleanup(m_device->getDevice());
}

//============================================================
// Dispatch helpers
//

void GroupNorm_Module::dispatch_forward_A(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*fwd_pass_A.descriptorSetLayout, *m_DescriptorPool);
	fwd_pass_A.descriptorSets.resize(1);
	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, mean_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, invstd_buffer.X->getDescriptorBufferInfo());
	writer.build(fwd_pass_A.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass_A.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass_A.pipelineLayout, 0, 1, &fwd_pass_A.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = pushconstants.num_groups * pushconstants.n;
	log() << "GroupNorm A: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(cmd, fwd_pass_A.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WorkGroups_x, 1, 1);
}

void GroupNorm_Module::dispatch_forward_B(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*fwd_pass_B.descriptorSetLayout, *m_DescriptorPool);
	fwd_pass_B.descriptorSets.resize(1);
	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, parameters.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, mean_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(3, invstd_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(4, output_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(5, xnorm_buffer.X->getDescriptorBufferInfo());
	writer.build(fwd_pass_B.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass_B.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass_B.pipelineLayout, 0, 1, &fwd_pass_B.descriptorSets[0], 0, 0);

	uint32_t n_elements     = pushconstants.n * pushconstants.c * pushconstants.h * pushconstants.w;
	uint32_t n_WorkGroups_x = (n_elements + 255) / 256;
	log() << "GroupNorm B: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(cmd, fwd_pass_B.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WorkGroups_x, 1, 1);
}

void GroupNorm_Module::dispatch_backward_1(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_1.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_1.descriptorSets.resize(1);
	writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, xnorm_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, parameters.Y->getDescriptorBufferInfo());
	writer.build(bwd_pass_1.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_1.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_1.pipelineLayout, 0, 1, &bwd_pass_1.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = pushconstants.c;
	log() << "GroupNorm bwd_1 (dγ/dβ): " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(cmd, bwd_pass_1.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WorkGroups_x, 1, 1);
}

void GroupNorm_Module::dispatch_backward_2(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_2.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_2.descriptorSets.resize(1);
	writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, xnorm_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, int_sums_buffer.X->getDescriptorBufferInfo());
	writer.build(bwd_pass_2.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_2.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_2.pipelineLayout, 0, 1, &bwd_pass_2.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = pushconstants.n * pushconstants.num_groups;
	log() << "GroupNorm bwd_2 (sums): " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(cmd, bwd_pass_2.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WorkGroups_x, 1, 1);
}

void GroupNorm_Module::dispatch_backward_4(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_4.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_4.descriptorSets.resize(1);
	writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, xnorm_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, invstd_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(3, int_sums_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(4, parameters.X->getDescriptorBufferInfo());
	writer.writeBuffer(5, grad_output.X->getDescriptorBufferInfo());
	writer.build(bwd_pass_4.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_4.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_4.pipelineLayout, 0, 1, &bwd_pass_4.descriptorSets[0], 0, 0);

	uint32_t n_elements     = pushconstants.n * pushconstants.c * pushconstants.h * pushconstants.w;
	uint32_t n_WorkGroups_x = (n_elements + 255) / 256;
	log() << "GroupNorm bwd_4 (dL/dx): " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(cmd, bwd_pass_4.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WorkGroups_x, 1, 1);
}

//============================================================
// forward / backward
//

void GroupNorm_Module::forward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_forward_A(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(fwd_pass_A.descriptorSets);

	cmd = m_device->beginSingleTimeCommands();
	dispatch_forward_B(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(fwd_pass_B.descriptorSets);
}

void GroupNorm_Module::backward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_backward_1(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(bwd_pass_1.descriptorSets);

	cmd = m_device->beginSingleTimeCommands();
	dispatch_backward_2(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(bwd_pass_2.descriptorSets);

	cmd = m_device->beginSingleTimeCommands();
	dispatch_backward_4(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(bwd_pass_4.descriptorSets);
}
