#include <irrlicht.h>
#include "vkModules.h"
#include "vkSigmoidModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Sigmoid Module
//

REFLECT_VKMOD_BEGIN(Sigmoid_Module)
	ALIAS("Sigmoid Layer")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void Sigmoid_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Sigmoid_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Sigmoid_Module::setDimensions()
{
	pushconstants.n = input_dimensions.B;
	pushconstants.c = input_dimensions.C;
	pushconstants.h = input_dimensions.H;
	pushconstants.w = input_dimensions.W;

	output_tensor.dimensions = input_dimensions;
	grad_output.dimensions   = input_dimensions;
}

void Sigmoid_Module::startup()
{
	// --- fwd_pass: sigmoid.spv ---
	// bindings: input_tensor(0), output_tensor(1)
	fwd_pass.bindings.resize(2);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = output_tensor.X->getDescriptorSetLayout(1);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);

	{
		VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };
		fwd_pass.createPipeline(m_device, "shaders/sigmoid.spv", r);
	}

	output_tensor.ready = true;

	// --- bwd_pass: sigmoid_grad.spv ---
	// bindings: output_tensor(0) [sigmoid(x) values], grad_input(1), grad_output(2)
	bwd_pass.bindings.resize(3);
	bwd_pass.bindings[0] = output_tensor.X->getDescriptorSetLayout(0);
	bwd_pass.bindings[1] = grad_input.X->getDescriptorSetLayout(1);
	bwd_pass.bindings[2] = grad_output.X->getDescriptorSetLayout(2);
	bwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass.bindings);

	{
		VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };
		bwd_pass.createPipeline(m_device, "shaders/sigmoid_grad.spv", r);
	}

	grad_output.ready = true;

	log() << "Sigmoid startup complete\n";
}

void Sigmoid_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
	bwd_pass.cleanup(m_device->getDevice());
}

void Sigmoid_Module::forward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();

	{
		MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
		fwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, output_tensor.X->getDescriptorBufferInfo());
		writer.build(fwd_pass.descriptorSets[0]);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = (pushconstants.n * pushconstants.c * pushconstants.h * pushconstants.w + 255) / 256;
	log() << "Sigmoid forward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());

	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
}

void Sigmoid_Module::backward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();

	{
		MyDescriptorWriter writer(*bwd_pass.descriptorSetLayout, *m_DescriptorPool);
		bwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, output_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, grad_input.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, grad_output.X->getDescriptorBufferInfo());
		writer.build(bwd_pass.descriptorSets[0]);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipelineLayout, 0, 1, &bwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = (pushconstants.n * pushconstants.c * pushconstants.h * pushconstants.w + 255) / 256;
	log() << "Sigmoid backward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(cmd, bwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());

	m_DescriptorPool->freeDescriptorsSets(bwd_pass.descriptorSets);
}
