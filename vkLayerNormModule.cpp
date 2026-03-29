#include <irrlicht.h>
#include "vkModules.h"
#include "vkLayerNormModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// LayerNorm Module
//

REFLECT_VKMOD_BEGIN(LayerNorm_Module)
	ALIAS("Layer Norm")
	INHERIT_FROM(Vulkan_Module)
	//Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_MEMBER(invstd_buffer)
	REFLECT_VKMOD_MEMBER(xnorm_buffer)
	REFLECT_VKMOD_PARAM(parameters)
	REFLECT_VKMOD_FEAT(output_tensor)
	//Backward Pass
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void LayerNorm_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void LayerNorm_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void LayerNorm_Module::setDimensions()
{
	// invstd: one scalar per token
	invstd_buffer.dimensions  = { input_dimensions.B, input_dimensions.C, 1, 1 };
	xnorm_buffer.dimensions   = input_dimensions;
	output_tensor.dimensions  = input_dimensions;
	// gamma + beta interleaved: [d*2+0] = gamma[d], [d*2+1] = beta[d]
	parameters.dimensions     = { 1, 1, 1, input_dimensions.W * 2 };
	grad_output.dimensions    = input_dimensions;

	pushconstants.B = input_dimensions.B;
	pushconstants.T = input_dimensions.C;
	pushconstants.D = input_dimensions.W;
}

void LayerNorm_Module::initialize_parameters()
{
	u32 D = input_dimensions.W;
	std::vector<float> data(D * 2, 0.0f);
	for (u32 d = 0; d < D; d++)
		data[d * 2 + 0] = 1.0f;  // gamma = 1, beta = 0

	upload_to_buffer(parameters.X, data);
	parameters.initialized = true;
	log() << "LayerNorm parameters initialized (gamma=1, beta=0)\n";
}

void LayerNorm_Module::startup()
{
	VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };

	// fwd_pass: input(0), invstd(1), xnorm(2), parameters(3), output(4)
	fwd_pass.bindings.resize(5);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = invstd_buffer.X->getDescriptorSetLayout(1);
	fwd_pass.bindings[2] = xnorm_buffer.X->getDescriptorSetLayout(2);
	fwd_pass.bindings[3] = parameters.X->getDescriptorSetLayout(3);
	fwd_pass.bindings[4] = output_tensor.X->getDescriptorSetLayout(4);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);
	fwd_pass.createPipeline(m_device, "shaders/layernorm.spv", r);
	invstd_buffer.ready = true;
	xnorm_buffer.ready  = true;
	output_tensor.ready = true;

	// bwd_pass_x: grad_input(0), xnorm(1), invstd(2), parameters(3), grad_output(4)
	bwd_pass_x.bindings.resize(5);
	bwd_pass_x.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass_x.bindings[1] = xnorm_buffer.X->getDescriptorSetLayout(1);
	bwd_pass_x.bindings[2] = invstd_buffer.X->getDescriptorSetLayout(2);
	bwd_pass_x.bindings[3] = parameters.X->getDescriptorSetLayout(3);
	bwd_pass_x.bindings[4] = grad_output.X->getDescriptorSetLayout(4);
	bwd_pass_x.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_x.bindings);
	bwd_pass_x.createPipeline(m_device, "shaders/layernorm_grad_x.spv", r);
	grad_output.ready = true;

	// bwd_pass_p: grad_input(0), xnorm(1), param_grad(2)
	bwd_pass_p.bindings.resize(3);
	bwd_pass_p.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass_p.bindings[1] = xnorm_buffer.X->getDescriptorSetLayout(1);
	bwd_pass_p.bindings[2] = parameters.Y->getDescriptorSetLayout(2);
	bwd_pass_p.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_p.bindings);
	bwd_pass_p.createPipeline(m_device, "shaders/layernorm_grad_p.spv", r);

	log() << "LayerNorm startup complete\n";
}

void LayerNorm_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
	bwd_pass_x.cleanup(m_device->getDevice());
	bwd_pass_p.cleanup(m_device->getDevice());
}

//============================================================
// Dispatch helpers
//

void LayerNorm_Module::dispatch_forward(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
	fwd_pass.descriptorSets.resize(1);
	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, invstd_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, xnorm_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(3, parameters.X->getDescriptorBufferInfo());
	writer.writeBuffer(4, output_tensor.X->getDescriptorBufferInfo());
	writer.build(fwd_pass.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WGs = pushconstants.B * pushconstants.T;
	log() << "LayerNorm fwd: " << n_WGs << " workgroups\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WGs, 1, 1);
}

void LayerNorm_Module::dispatch_backward_x(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_x.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_x.descriptorSets.resize(1);
	writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, xnorm_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, invstd_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(3, parameters.X->getDescriptorBufferInfo());
	writer.writeBuffer(4, grad_output.X->getDescriptorBufferInfo());
	writer.build(bwd_pass_x.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_x.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_x.pipelineLayout, 0, 1, &bwd_pass_x.descriptorSets[0], 0, 0);

	uint32_t n_WGs = pushconstants.B * pushconstants.T;
	log() << "LayerNorm bwd_x (dL/dx): " << n_WGs << " workgroups\n";

	vkCmdPushConstants(cmd, bwd_pass_x.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WGs, 1, 1);
}

void LayerNorm_Module::dispatch_backward_p(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*bwd_pass_p.descriptorSetLayout, *m_DescriptorPool);
	bwd_pass_p.descriptorSets.resize(1);
	writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, xnorm_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, parameters.Y->getDescriptorBufferInfo());
	writer.build(bwd_pass_p.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass_p.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_p.pipelineLayout, 0, 1, &bwd_pass_p.descriptorSets[0], 0, 0);

	uint32_t n_WGs = pushconstants.D;  // one WG per dimension
	log() << "LayerNorm bwd_p (dγ/dβ): " << n_WGs << " workgroups\n";

	vkCmdPushConstants(cmd, bwd_pass_p.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WGs, 1, 1);
}

//============================================================
// forward / backward
//

void LayerNorm_Module::forward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_forward(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
}

void LayerNorm_Module::backward()
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
