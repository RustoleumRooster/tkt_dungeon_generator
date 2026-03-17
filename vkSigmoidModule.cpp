#include <irrlicht.h>
#include "vkModules.h"
#include "vkSigmoidModule.h"
#include "soa.h"
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
	REFLECT_STRUCT_MEMBER(input_tensor)
	REFLECT_STRUCT_MEMBER(pass_output)
	REFLECT_VKMOD_MEMBER_OUTPUT_IN_PLACE(input_tensor, pass_output)
REFLECT_VKMOD_END()

void Sigmoid_Module::run()
{
	createDescriptorSetLayout();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	createComputePipeline("shaders/sigmoid.spv", push_constant);

	execute();

	pass_output.ready = true;
	pass_output.X = input_tensor.X;

	cleanup();
}

void Sigmoid_Module::setDimensions()
{
	pushconstants.n = input_dimensions.B;
	pushconstants.c = input_dimensions.C;
	pushconstants.h = input_dimensions.H;
	pushconstants.w = input_dimensions.W;
}

void Sigmoid_Module::createDescriptorSets()
{
	MyDescriptorWriter writer(*descriptorSetLayout, *m_DescriptorPool);

	descriptorSets.resize(1);

	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());

	writer.build(descriptorSets[0]);
}

void Sigmoid_Module::createDescriptorSetLayout()
{
	bindings.resize(1);
	bindings[0] = input_tensor.X->getDescriptorSetLayout(0);

	descriptorSetLayout = new MyDescriptorSetLayout(m_device, bindings);
}

void Sigmoid_Module::execute()
{
	createDescriptorSets();

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		&descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = (pushconstants.n * pushconstants.c * pushconstants.h * pushconstants.w) / 256;
	uint32_t n_WorkGroups_y = 1;
	uint32_t n_WorkGroups_z = 1;

	log() << "(" << n_WorkGroups_x << " / " << n_WorkGroups_y << ")\n";

	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, n_WorkGroups_y, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());
}

void Sigmoid_Module::cleanup()
{
	descriptorSetLayout->cleanup();

	pipeline->cleanup();

	vkDestroyPipelineLayout(m_device->getDevice(), pipelineLayout, nullptr);
}
