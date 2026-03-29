#include <irrlicht.h>
#include "vkModules.h"
#include "vkPosEmbedModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// PosEmbed Module
//

REFLECT_VKMOD_BEGIN(PosEmbed_Module)
	ALIAS("Positional Embedding")
	INHERIT_FROM(Vulkan_Module)
	// Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_PARAM(parameters)
	REFLECT_VKMOD_FEAT(output_tensor)
	// Backward Pass
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void PosEmbed_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void PosEmbed_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void PosEmbed_Module::setDimensions()
{
	u32 R = input_dimensions.C;  // rows
	u32 C = input_dimensions.H;  // cols
	u32 D = input_dimensions.W;  // embedding dim

	output_tensor.dimensions = input_dimensions;
	parameters.dimensions    = { 1, 1, R + C, D };  // row embeds then col embeds
	grad_output.dimensions   = input_dimensions;

	pushconstants.B = input_dimensions.B;
	pushconstants.R = R;
	pushconstants.C = C;
	pushconstants.D = D;
}

void PosEmbed_Module::initialize_parameters()
{
	u32 R = input_dimensions.C;
	u32 C = input_dimensions.H;
	u32 D = input_dimensions.W;

	// Small random initialization — N(0, 0.02)
	std::mt19937 rng(42);
	std::normal_distribution<float> dist(0.0f, 0.02f);

	std::vector<float> data((R + C) * D);
	for (auto& v : data) v = dist(rng);

	upload_to_buffer(parameters.X, data);
	parameters.initialized = true;
	log() << "PosEmbed parameters initialized (R=" << R << " C=" << C << " D=" << D << ")\n";
}

void PosEmbed_Module::startup()
{
	VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };

	// fwd_pass: input(0), params(1), output(2)
	fwd_pass.bindings.resize(3);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = parameters.X->getDescriptorSetLayout(1);
	fwd_pass.bindings[2] = output_tensor.X->getDescriptorSetLayout(2);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);
	fwd_pass.createPipeline(m_device, "shaders/pos_embed.spv", r);
	output_tensor.ready = true;

	// bwd_pass_x: grad_input(0), grad_output(1)
	bwd_pass_x.bindings.resize(2);
	bwd_pass_x.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass_x.bindings[1] = grad_output.X->getDescriptorSetLayout(1);
	bwd_pass_x.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_x.bindings);
	bwd_pass_x.createPipeline(m_device, "shaders/pos_embed_grad_x.spv", r);
	grad_output.ready = true;

	// bwd_pass_p: grad_input(0), param_grad(1)
	bwd_pass_p.bindings.resize(2);
	bwd_pass_p.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass_p.bindings[1] = parameters.Y->getDescriptorSetLayout(1);
	bwd_pass_p.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_p.bindings);
	bwd_pass_p.createPipeline(m_device, "shaders/pos_embed_grad_p.spv", r);

	log() << "PosEmbed startup complete\n";
}

void PosEmbed_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
	bwd_pass_x.cleanup(m_device->getDevice());
	bwd_pass_p.cleanup(m_device->getDevice());
}

//============================================================
// Dispatch helpers
//

void PosEmbed_Module::dispatch_forward(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
	fwd_pass.descriptorSets.resize(1);
	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, parameters.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, output_tensor.X->getDescriptorBufferInfo());
	writer.build(fwd_pass.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WGs = pushconstants.B * pushconstants.R * pushconstants.C;
	log() << "PosEmbed fwd: " << n_WGs << " workgroups\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WGs, 1, 1);
}

void PosEmbed_Module::dispatch_backward_x(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_x.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_x.descriptorSets.resize(1);
	writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, grad_output.X->getDescriptorBufferInfo());
	writer.build(bwd_pass_x.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_x.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_x.pipelineLayout, 0, 1, &bwd_pass_x.descriptorSets[0], 0, 0);

	uint32_t n_WGs = pushconstants.B * pushconstants.R * pushconstants.C;
	log() << "PosEmbed bwd_x (pass-through): " << n_WGs << " workgroups\n";

	vkCmdPushConstants(cmd, bwd_pass_x.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WGs, 1, 1);
}

void PosEmbed_Module::dispatch_backward_p(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_p.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_p.descriptorSets.resize(1);
	writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, parameters.Y->getDescriptorBufferInfo());
	writer.build(bwd_pass_p.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_p.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_p.pipelineLayout, 0, 1, &bwd_pass_p.descriptorSets[0], 0, 0);

	uint32_t n_WGs = pushconstants.R + pushconstants.C;  // one WG per embedding row
	log() << "PosEmbed bwd_p (param grad): " << n_WGs << " workgroups\n";

	vkCmdPushConstants(cmd, bwd_pass_p.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WGs, 1, 1);
}

//============================================================
// forward / backward
//

void PosEmbed_Module::forward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_forward(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
}

void PosEmbed_Module::backward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_backward_x(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(bwd_pass_x.descriptorSets);

	cmd = m_device->beginSingleTimeCommands();
	dispatch_backward_p(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(bwd_pass_p.descriptorSets);
}
