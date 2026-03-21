#include <irrlicht.h>
#include "vkModules.h"
#include "vkSkipModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Skip (Add) Module
//

REFLECT_VKMOD_BEGIN(Skip_Module)
	ALIAS("Skip Module")
	INHERIT_FROM(Vulkan_Module)
//Forward Pass
	REFLECT_VKMOD_FEAT(input_a)
	REFLECT_VKMOD_FEAT(input_b)
	REFLECT_VKMOD_FEAT(output)
//Backward Pass
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output_a)
	REFLECT_VKMOD_GRAD(grad_output_b)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void Skip_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Skip_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Skip_Module::setDimensions()
{
	output.dimensions       = input_dimensions;
	grad_output_a.dimensions = input_dimensions;
	grad_output_b.dimensions = input_dimensions;

	pushconstants.n_elements = input_dimensions.B * input_dimensions.C *
	                           input_dimensions.H * input_dimensions.W;
}

void Skip_Module::forward()
{
	// bindings: input_a(0), input_b(1), output(2)
	fwd_pass.bindings.resize(3);
	fwd_pass.bindings[0] = input_a.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = input_b.X->getDescriptorSetLayout(1);
	fwd_pass.bindings[2] = output.X->getDescriptorSetLayout(2);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	fwd_pass.createPipeline(m_device, "shaders/skip.spv", push_constant);

	output.ready = true;

	{
		MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
		fwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, input_a.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, input_b.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, output.X->getDescriptorBufferInfo());
		writer.build(fwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = (pushconstants.n_elements + 255) / 256;

	log() << "Skip forward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	fwd_pass.cleanup(m_device->getDevice());
}

void Skip_Module::backward()
{
	// Add backward: dL/dA = dL/dB = dL/dOutput — copy grad_input to both outputs
	// bindings: grad_input(0), grad_output_a(1), grad_output_b(2)
	bwd_pass.bindings.resize(3);
	bwd_pass.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass.bindings[1] = grad_output_a.X->getDescriptorSetLayout(1);
	bwd_pass.bindings[2] = grad_output_b.X->getDescriptorSetLayout(2);
	bwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	bwd_pass.createPipeline(m_device, "shaders/skip_grad.spv", push_constant);

	grad_output_a.ready = true;
	grad_output_b.ready = true;

	{
		MyDescriptorWriter writer(*bwd_pass.descriptorSetLayout, *m_DescriptorPool);
		bwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, grad_output_a.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, grad_output_b.X->getDescriptorBufferInfo());
		writer.build(bwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipelineLayout, 0, 1, &bwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = (pushconstants.n_elements + 255) / 256;

	log() << "Skip backward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, bwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(bwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	bwd_pass.cleanup(m_device->getDevice());
}
