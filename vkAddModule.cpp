#include <irrlicht.h>
#include "vkModules.h"
#include "vkAddModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Add Module
//

REFLECT_VKMOD_BEGIN(Add_Module)
	ALIAS("Add")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_VKMOD_FEAT(input_a)
	REFLECT_VKMOD_FEAT(input_b)
	REFLECT_VKMOD_FEAT(output)
REFLECT_VKMOD_END()

void Add_Module::setDimensions()
{
	output.dimensions = input_dimensions;
	pushconstants.n_elements = input_dimensions.B * input_dimensions.C *
	                           input_dimensions.H * input_dimensions.W;
}

void Add_Module::run()
{
	createDescriptorSetLayout();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	createComputePipeline("shaders/add.spv", push_constant);

	output.ready = true;

	execute();
	cleanup();
}

void Add_Module::createDescriptorSets()
{
	MyDescriptorWriter writer(*descriptorSetLayout, *m_DescriptorPool);

	descriptorSets.resize(1);

	writer.writeBuffer(0, input_a.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, input_b.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, output.X->getDescriptorBufferInfo());

	writer.build(descriptorSets[0]);
}

void Add_Module::createDescriptorSetLayout()
{
	bindings.resize(3);
	bindings[0] = input_a.X->getDescriptorSetLayout(0);
	bindings[1] = input_b.X->getDescriptorSetLayout(1);
	bindings[2] = output.X->getDescriptorSetLayout(2);

	descriptorSetLayout = new MyDescriptorSetLayout(m_device, bindings);
}

void Add_Module::execute()
{
	createDescriptorSets();

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		&descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = pushconstants.n_elements % 256 == 0 ? pushconstants.n_elements / 256 : (pushconstants.n_elements + 1) / 256;

	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());
}

void Add_Module::cleanup()
{
	descriptorSetLayout->cleanup();

	pipeline->cleanup();

	vkDestroyPipelineLayout(m_device->getDevice(), pipelineLayout, nullptr);
}
