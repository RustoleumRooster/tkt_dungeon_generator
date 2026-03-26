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
	REFLECT_VKMOD_FEAT(input)
	REFLECT_VKMOD_FEAT(output)
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void NNUp_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void NNUp_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void NNUp_Module::setDimensions()
{
	output.dimensions = {
		input_dimensions.B,
		input_dimensions.C,
		input_dimensions.H * 2,
		input_dimensions.W * 2
	};

	grad_output.dimensions = input_dimensions;

	pushconstants.C          = input_dimensions.C;
	pushconstants.H          = output.dimensions.H;
	pushconstants.W          = output.dimensions.W;
	pushconstants.n_elements = output.dimensions.B * output.dimensions.C *
	                           output.dimensions.H * output.dimensions.W;
}

void NNUp_Module::startup()
{
	// --- fwd_pass: NN_up.spv ---
	// bindings: input(0), output(1)
	fwd_pass.bindings.resize(2);
	fwd_pass.bindings[0] = input.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = output.X->getDescriptorSetLayout(1);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);

	{
		VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };
		fwd_pass.createPipeline(m_device, "shaders/NN_up.spv", r);
	}

	output.ready = true;

	// --- bwd_pass: NN_up_grad.spv ---
	// bindings: grad_input(0) [dL/d_output, large], grad_output(1) [dL/d_input, small]
	bwd_pass.bindings.resize(2);
	bwd_pass.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass.bindings[1] = grad_output.X->getDescriptorSetLayout(1);
	bwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass.bindings);

	{
		VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };
		bwd_pass.createPipeline(m_device, "shaders/NN_up_grad.spv", r);
	}

	grad_output.ready = true;

	log() << "NNUp startup complete\n";
}

void NNUp_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
	bwd_pass.cleanup(m_device->getDevice());
}

void NNUp_Module::forward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();

	{
		MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
		fwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, input.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, output.X->getDescriptorBufferInfo());
		writer.build(fwd_pass.descriptorSets[0]);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = (pushconstants.n_elements + 255) / 256;
	log() << "NNUp forward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());

	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
}

void NNUp_Module::backward()
{
	// n_elements for backward = input element count (small tensor);
	// H/W in push constants are the OUTPUT (large) dimensions — shader divides by 2 internally
	uint32_t n_elements_in  = input_dimensions.B * input_dimensions.C *
	                          input_dimensions.H * input_dimensions.W;
	pushconstant_struct bwd_pc = pushconstants;
	bwd_pc.n_elements = n_elements_in;

	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();

	{
		MyDescriptorWriter writer(*bwd_pass.descriptorSetLayout, *m_DescriptorPool);
		bwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, grad_output.X->getDescriptorBufferInfo());
		writer.build(bwd_pass.descriptorSets[0]);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipelineLayout, 0, 1, &bwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = (n_elements_in + 255) / 256;
	log() << "NNUp backward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(cmd, bwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &bwd_pc);
	vkCmdDispatch(cmd, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());

	m_DescriptorPool->freeDescriptorsSets(bwd_pass.descriptorSets);
}
