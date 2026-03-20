#include <irrlicht.h>
#include "vkModules.h"
#include "vkGroupNormModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// GroupNorm Module
//

REFLECT_VKMOD_BEGIN(GroupNorm_Module)
	ALIAS("Group Norm Layer")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_MEMBER(mean_buffer)
	REFLECT_VKMOD_MEMBER(var_buffer)
	REFLECT_VKMOD_PARAM(parameters)
	REFLECT_VKMOD_FEAT(output_tensor)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void GroupNorm_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void GroupNorm_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void GroupNorm_Module::setDimensions()
{
	mean_buffer.dimensions = { 1, 1, input_dimensions.B, pushconstants.num_groups };
	var_buffer.dimensions  = { 1, 1, input_dimensions.B, pushconstants.num_groups };
	parameters.dimensions  = { 1, 1, 2, input_dimensions.C }; // gamma, beta per channel
	output_tensor.dimensions = input_dimensions;

	pushconstants.n = input_dimensions.B;
	pushconstants.c = input_dimensions.C;
	pushconstants.h = input_dimensions.H;
	pushconstants.w = input_dimensions.W;
}

void GroupNorm_Module::run()
{
	forward_A();
	forward_B();
}

void GroupNorm_Module::forward_A()
{
	// bindings: input_tensor(0), mean_buffer(1), var_buffer(2)
	// computes per-group mean and variance
	fwd_pass_A.bindings.resize(3);
	fwd_pass_A.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass_A.bindings[1] = mean_buffer.X->getDescriptorSetLayout(1);
	fwd_pass_A.bindings[2] = var_buffer.X->getDescriptorSetLayout(2);
	fwd_pass_A.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass_A.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	fwd_pass_A.createPipeline(m_device, "shaders/groupnorm.spv", push_constant);

	mean_buffer.ready = true;
	var_buffer.ready  = true;
	output_tensor.ready = true;
	output_tensor.X     = input_tensor.X;

	{
		MyDescriptorWriter writer(*fwd_pass_A.descriptorSetLayout, *m_DescriptorPool);
		fwd_pass_A.descriptorSets.resize(1);
		writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, mean_buffer.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, var_buffer.X->getDescriptorBufferInfo());
		writer.build(fwd_pass_A.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass_A.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass_A.pipelineLayout, 0, 1, &fwd_pass_A.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = pushconstants.num_groups * pushconstants.n;

	log() << "GroupNorm A: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, fwd_pass_A.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(fwd_pass_A.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	fwd_pass_A.cleanup(m_device->getDevice());
}

void GroupNorm_Module::forward_B()
{
	// bindings: input_tensor(0), parameters(1), mean_buffer(2), var_buffer(3)
	// applies normalization with learned gamma/beta in-place
	fwd_pass_B.bindings.resize(4);
	fwd_pass_B.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass_B.bindings[1] = parameters.X->getDescriptorSetLayout(1);
	fwd_pass_B.bindings[2] = mean_buffer.X->getDescriptorSetLayout(2);
	fwd_pass_B.bindings[3] = var_buffer.X->getDescriptorSetLayout(3);
	fwd_pass_B.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass_B.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	fwd_pass_B.createPipeline(m_device, "shaders/groupnorm2.spv", push_constant);

	{
		MyDescriptorWriter writer(*fwd_pass_B.descriptorSetLayout, *m_DescriptorPool);
		fwd_pass_B.descriptorSets.resize(1);
		writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, parameters.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, mean_buffer.X->getDescriptorBufferInfo());
		writer.writeBuffer(3, var_buffer.X->getDescriptorBufferInfo());
		writer.build(fwd_pass_B.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass_B.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass_B.pipelineLayout, 0, 1, &fwd_pass_B.descriptorSets[0], 0, 0);

	uint32_t n_elements     = pushconstants.n * pushconstants.c * pushconstants.h * pushconstants.w;
	uint32_t n_WorkGroups_x = (n_elements + 255) / 256;

	log() << "GroupNorm B: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, fwd_pass_B.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(fwd_pass_B.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	fwd_pass_B.cleanup(m_device->getDevice());
}

void GroupNorm_Module::backward()
{
	// stub
}
