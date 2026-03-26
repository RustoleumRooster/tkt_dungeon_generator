#include <irrlicht.h>
#include "vkModules.h"
#include "vkOptimizationModule.h"
#include <vulkan/vulkan.h>

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Optimization_Module
//

REFLECT_VKMOD_BEGIN(Optimization_Module)
	ALIAS("Adam Optimizer")
	INHERIT_FROM(Vulkan_Module)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void Optimization_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Optimization_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// backward
//

void Optimization_Module::startup()
{
	// --- bwd_pass: adam.spv ---
	// bindings: params(0), grads(1), m(2), v(3)
	bwd_pass.bindings.resize(4);
	bwd_pass.bindings[0] = params_buf->getDescriptorSetLayout(0);
	bwd_pass.bindings[1] = grads_buf->getDescriptorSetLayout(1);
	bwd_pass.bindings[2] = m_buf->getDescriptorSetLayout(2);
	bwd_pass.bindings[3] = v_buf->getDescriptorSetLayout(3);
	bwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass.bindings);

	{
		VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };
		bwd_pass.createPipeline(m_device, "shaders/adam.spv", r);
	}

	log() << "Adam startup complete\n";
}

void Optimization_Module::cleanup_passes()
{
	bwd_pass.cleanup(m_device->getDevice());
}

void Optimization_Module::backward()
{
	// B1_t and B2_t are updated per-step by the workflow before this call.
	{
		MyDescriptorWriter writer(*bwd_pass.descriptorSetLayout, *m_DescriptorPool);
		bwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, params_buf->getDescriptorBufferInfo());
		writer.writeBuffer(1, grads_buf->getDescriptorBufferInfo());
		writer.writeBuffer(2, m_buf->getDescriptorBufferInfo());
		writer.writeBuffer(3, v_buf->getDescriptorBufferInfo());
		writer.build(bwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipelineLayout, 0, 1, &bwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = (pushconstants.n_elements + 255) / 256;
	log() << "Adam optimizer: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(cmd, bwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());

	m_DescriptorPool->freeDescriptorsSets(bwd_pass.descriptorSets);
}
