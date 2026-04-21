#include <irrlicht.h>
#include "vkModules.h"
#include "vkVocabSoftmaxModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Vocab_Softmax_Module
//

REFLECT_VKMOD_BEGIN(Vocab_Softmax_Module)
	ALIAS("Vocab Softmax")
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

void Vocab_Softmax_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Vocab_Softmax_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Vocab_Softmax_Module::setDimensions()
{
	u32 B         = input_dimensions.B;
	u32 num_heads = input_dimensions.C;
	u32 M         = input_dimensions.H;  // seq_len
	u32 N         = input_dimensions.W;  // vocab_size

	output_tensor.dimensions = input_dimensions;
	grad_output.dimensions   = input_dimensions;

	pushconstants.B         = B;
	pushconstants.num_heads = num_heads;
	pushconstants.M         = M;
	pushconstants.N         = N;
}

void Vocab_Softmax_Module::startup()
{
	VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };

	fwd_pass.bindings.resize(2);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = output_tensor.X->getDescriptorSetLayout(1);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);
	fwd_pass.createPipeline(m_device, "shaders/softmax_vocab.spv", r);
	output_tensor.ready = true;

	log() << "Vocab Softmax startup complete\n";
}

void Vocab_Softmax_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
}

//============================================================
// forward / backward
//

void Vocab_Softmax_Module::forward()
{
	MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
	fwd_pass.descriptorSets.resize(1);
	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, output_tensor.X->getDescriptorBufferInfo());
	writer.build(fwd_pass.descriptorSets[0]);

	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	// one workgroup per row: B * num_heads * M rows total
	uint32_t n_workgroups = pushconstants.B * pushconstants.num_heads * pushconstants.M;
	log() << "Vocab Softmax fwd: " << n_workgroups << " workgroups\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_workgroups, 1, 1);

	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
}

void Vocab_Softmax_Module::backward()
{
	// TODO: softmax backward — dL/dx_i = p_i * (dL/dy_i - sum_j(dL/dy_j * p_j))
}
