#include <irrlicht.h>
#include "vkModules.h"
#include "vkSoftmaxModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Softmax Module
//

REFLECT_VKMOD_BEGIN(Softmax_Module)
	ALIAS("Softmax")
	INHERIT_FROM(Vulkan_Module)
	// Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	// Backward Pass
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void Softmax_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Softmax_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Softmax_Module::setDimensions()
{
	output_tensor.dimensions = input_dimensions;
	grad_output.dimensions   = input_dimensions;

	pushconstants.B         = input_dimensions.B;
	pushconstants.num_heads = input_dimensions.C;
	pushconstants.M         = input_dimensions.H;
	pushconstants.N         = input_dimensions.W;
}

void Softmax_Module::startup()
{
	VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };

	fwd_pass.bindings.resize(2);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = output_tensor.X->getDescriptorSetLayout(1);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);
	fwd_pass.createPipeline(m_device, "shaders/softmax.spv", r);
	output_tensor.ready = true;

	log() << "Softmax startup complete\n";
}

void Softmax_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
}

//============================================================
// Dispatch helper
//

void Softmax_Module::dispatch_forward(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
	fwd_pass.descriptorSets.resize(1);
	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, output_tensor.X->getDescriptorBufferInfo());
	writer.build(fwd_pass.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	// one workgroup per row: B * num_heads * M rows total
	uint32_t n_WGs = pushconstants.B * pushconstants.num_heads * pushconstants.M;
	log() << "Softmax fwd: " << n_WGs << " workgroups\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_WGs, 1, 1);
}

//============================================================
// forward / backward
//

void Softmax_Module::forward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_forward(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
}

void Softmax_Module::backward()
{
	// TODO
}
