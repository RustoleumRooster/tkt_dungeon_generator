#include <irrlicht.h>
#include "vkModules.h"
#include "vkBCELossModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// BCE Loss Module
//

REFLECT_VKMOD_BEGIN(BCE_Loss_Module)
	ALIAS("BCE Loss")
	INHERIT_FROM(Vulkan_Module)
REFLECT_VKMOD_FORWARD_PASS()
	REFLECT_VKMOD_MEMBER(predictions)
	REFLECT_VKMOD_MEMBER(ground_truth)
	REFLECT_VKMOD_MEMBER(loss)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
REFLECT_VKMOD_BACKWARD_PASS()
	REFLECT_VKMOD_MEMBER(gradient_out)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void BCE_Loss_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void BCE_Loss_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void BCE_Loss_Module::setDimensions()
{
	loss.dimensions = { 1,1,1,1 };

	pushconstants.n_elements = input_dimensions.B *
	                           input_dimensions.H *
	                           input_dimensions.W;

	gradient_out.dimensions = { input_dimensions.B, 1,
	                             input_dimensions.H, input_dimensions.W };
}

void BCE_Loss_Module::run()
{
	forward();
}

void BCE_Loss_Module::forward()
{
	// descriptor set layout: predictions(0), ground_truth(1), loss(2)
	fwd_pass.bindings.resize(3);
	fwd_pass.bindings[0] = predictions.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = ground_truth.X->getDescriptorSetLayout(1);
	fwd_pass.bindings[2] = loss.X->getDescriptorSetLayout(2);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	fwd_pass.createPipeline(m_device, "shaders/bce_loss.spv", push_constant);

	loss.ready = true;

	// allocate descriptor sets
	{
		MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
		fwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, predictions.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, ground_truth.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, loss.X->getDescriptorBufferInfo());
		writer.build(fwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	// zero the loss accumulator before dispatching
	vkCmdFillBuffer(commandBuffer, loss.X->Buffer, 0, VK_WHOLE_SIZE, 0);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = (pushconstants.n_elements + 255) / 256;

	log() << "BCE forward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	fwd_pass.cleanup(m_device->getDevice());
}

void BCE_Loss_Module::backward()
{
	// bindings: predictions(0), ground_truth(1), gradient_out(2)
	bwd_pass.bindings.resize(3);
	bwd_pass.bindings[0] = predictions.X->getDescriptorSetLayout(0);
	bwd_pass.bindings[1] = ground_truth.X->getDescriptorSetLayout(1);
	bwd_pass.bindings[2] = gradient_out.X->getDescriptorSetLayout(2);
	bwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	bwd_pass.createPipeline(m_device, "shaders/bce_grad.spv", push_constant);

	gradient_out.ready = true;

	// allocate descriptor sets
	{
		MyDescriptorWriter writer(*bwd_pass.descriptorSetLayout, *m_DescriptorPool);
		bwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, predictions.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, ground_truth.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, gradient_out.X->getDescriptorBufferInfo());
		writer.build(bwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipelineLayout, 0, 1, &bwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = (pushconstants.n_elements + 255) / 256;

	log() << "BCE backward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, bwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(bwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	bwd_pass.cleanup(m_device->getDevice());
}
