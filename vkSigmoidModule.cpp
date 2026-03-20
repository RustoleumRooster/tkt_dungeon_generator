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
	REFLECT_VKMOD_MEMBER(input_tensor)
	REFLECT_VKMOD_MEMBER(output_tensor)
	REFLECT_VKMOD_MEMBER(grad_input)
	REFLECT_VKMOD_MEMBER(grad_output)
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
	// backward pass: grad_output same shape as input
	grad_output.dimensions = input_dimensions;
}

void Sigmoid_Module::run()
{
	forward();
}

void Sigmoid_Module::forward()
{
	// binding: input_tensor(0) — in-place operation
	fwd_pass.bindings.resize(1);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	fwd_pass.createPipeline(m_device, "shaders/sigmoid.spv", push_constant);

	output_tensor.ready = true;
	//output_tensor.X = input_tensor.X;

	{
		MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
		fwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
		writer.build(fwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_elements    = pushconstants.n * pushconstants.c * pushconstants.h * pushconstants.w;
	uint32_t n_WorkGroups_x = (n_elements + 255) / 256;

	log() << "Sigmoid forward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	fwd_pass.cleanup(m_device->getDevice());
}

void Sigmoid_Module::backward()
{
	// bindings: output_tensor(0) [sigmoid(x) values], grad_input(1), grad_output(2)
	// grad: dL/dx = grad_input * output_tensor * (1 - output_tensor)
	bwd_pass.bindings.resize(3);
	bwd_pass.bindings[0] = output_tensor.X->getDescriptorSetLayout(0);
	bwd_pass.bindings[1] = grad_input.X->getDescriptorSetLayout(1);
	bwd_pass.bindings[2] = grad_output.X->getDescriptorSetLayout(2);
	bwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	bwd_pass.createPipeline(m_device, "shaders/sigmoid_grad.spv", push_constant);

	grad_output.ready = true;

	{
		MyDescriptorWriter writer(*bwd_pass.descriptorSetLayout, *m_DescriptorPool);
		bwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, output_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, grad_input.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, grad_output.X->getDescriptorBufferInfo());
		writer.build(bwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipelineLayout, 0, 1, &bwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_elements     = pushconstants.n * pushconstants.c * pushconstants.h * pushconstants.w;
	uint32_t n_WorkGroups_x = (n_elements + 255) / 256;

	log() << "Sigmoid backward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, bwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(bwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	bwd_pass.cleanup(m_device->getDevice());
}
