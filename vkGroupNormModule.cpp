#include <irrlicht.h>
#include "vkModules.h"
#include "vkGroupNormModule.h"
#include "soa.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"
#include <chrono>

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// GroupNorm Module
//

REFLECT_VKMOD_BEGIN(GroupNorm_Module)
	ALIAS("Group Norm Layer")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_VKMOD_MEMBER(input_tensor)
	REFLECT_VKMOD_MEMBER(pass_output)
	REFLECT_VKMOD_MEMBER(mean_buffer)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
	REFLECT_VKMOD_MEMBER(var_buffer)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
	REFLECT_VKMOD_MEMBER(scratchpad)
	REFLECT_VKMOD_MEMBER_OUTPUT_IN_PLACE(input_tensor, pass_output)
REFLECT_VKMOD_END()

void GroupNorm_Module::run()
{
	createDescriptorSetLayout();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	createComputePipeline("shaders/groupnorm.spv", push_constant);

	execute();

	mean_buffer.ready = true;
	var_buffer.ready = true;
	pass_output.ready = true;
	pass_output.X = input_tensor.X;

	cleanup();
}

void GroupNorm_Module::setDimensions()
{
	mean_buffer.dimensions = { 1,1,input_dimensions.B, pushconstants.num_groups };
	var_buffer.dimensions  = { 1,1,input_dimensions.B, pushconstants.num_groups };

	pushconstants.n = input_dimensions.B;
	pushconstants.c = input_dimensions.C;
	pushconstants.h = input_dimensions.H;
	pushconstants.w = input_dimensions.W;
}

void GroupNorm_Module::createDescriptorSets()
{
	MyDescriptorWriter writer(*descriptorSetLayout, *m_DescriptorPool);

	descriptorSets.resize(1);

	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, mean_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, var_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(3, scratchpad.X->getDescriptorBufferInfo());

	writer.build(descriptorSets[0]);
}

void GroupNorm_Module::createDescriptorSetLayout()
{
	bindings.resize(4);
	bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	bindings[1] = mean_buffer.X->getDescriptorSetLayout(1);
	bindings[2] = var_buffer.X->getDescriptorSetLayout(2);
	bindings[3] = scratchpad.X->getDescriptorSetLayout(3);

	descriptorSetLayout = new MyDescriptorSetLayout(m_device, bindings);
}

void GroupNorm_Module::execute()
{
	createDescriptorSets();

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		&descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = pushconstants.num_groups * pushconstants.n;
	uint32_t n_WorkGroups_y = 1;
	uint32_t n_WorkGroups_z = 1;

	log() << "(" << n_WorkGroups_x << " / " << n_WorkGroups_y << ")\n";

	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, n_WorkGroups_y, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());
}

void GroupNorm_Module::cleanup()
{
	descriptorSetLayout->cleanup();

	pipeline->cleanup();

	vkDestroyPipelineLayout(m_device->getDevice(), pipelineLayout, nullptr);
}
