#include <irrlicht.h>
#include "vkModules.h"
#include "vkNNUpModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Nearest-Neighbor 2x Upscale Module
//

REFLECT_VKMOD_BEGIN(NNUp_Module)
	ALIAS("NNUp")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_VKMOD_MEMBER(input)
	REFLECT_VKMOD_MEMBER(output)
REFLECT_VKMOD_END()

void NNUp_Module::setDimensions()
{
	// Output is 2x in H and W
	output.dimensions = {
		input_dimensions.B,
		input_dimensions.C,
		input_dimensions.H * 2,
		input_dimensions.W * 2
	};

	pushconstants.C          = input_dimensions.C;
	pushconstants.H          = output.dimensions.H;
	pushconstants.W          = output.dimensions.W;
	pushconstants.n_elements = output.dimensions.B * output.dimensions.C *
	                           output.dimensions.H * output.dimensions.W;
}

void NNUp_Module::run()
{
	createDescriptorSetLayout();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	createComputePipeline("shaders/NN_up.spv", push_constant);

	output.ready = true;

	execute();
	cleanup();
}

void NNUp_Module::createDescriptorSets()
{
	MyDescriptorWriter writer(*descriptorSetLayout, *m_DescriptorPool);

	descriptorSets.resize(1);

	writer.writeBuffer(0, input.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, output.X->getDescriptorBufferInfo());

	writer.build(descriptorSets[0]);
}

void NNUp_Module::createDescriptorSetLayout()
{
	bindings.resize(2);
	bindings[0] = input.X->getDescriptorSetLayout(0);
	bindings[1] = output.X->getDescriptorSetLayout(1);

	descriptorSetLayout = new MyDescriptorSetLayout(m_device, bindings);
}

void NNUp_Module::execute()
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

void NNUp_Module::cleanup()
{
	descriptorSetLayout->cleanup();

	pipeline->cleanup();

	vkDestroyPipelineLayout(m_device->getDevice(), pipelineLayout, nullptr);
}
