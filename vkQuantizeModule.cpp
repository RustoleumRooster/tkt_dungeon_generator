#include <irrlicht.h>
#include "vkModules.h"
#include "vkQuantizeModule.h"
#include "soa.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"
#include <chrono>

std::chrono::steady_clock::time_point q_startTime;
std::chrono::steady_clock::time_point q_currentTime;
float q_passedTime;

#define Q_START_TIMER() q_startTime = std::chrono::high_resolution_clock::now();
#define Q_PRINT_TIMER() q_currentTime = std::chrono::high_resolution_clock::now(); \
    q_passedTime = std::chrono::duration<float, std::chrono::seconds::period>(q_currentTime - q_startTime).count(); \
    log() << q_passedTime << "\n";

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Quantize Module
//

REFLECT_VKMOD_BEGIN(Quantize_Module)
	ALIAS("Quantize Layer")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	REFLECT_VKMOD_PARAM(codebook)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void Quantize_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Quantize_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Quantize_Module::setDimensions()
{
	codebook.dimensions      = codebook_size;
	output_tensor.dimensions = input_dimensions;
	pushconstants.n_vectors  = codebook_size.H;
}

void Quantize_Module::forward()
{
	// bindings: input_tensor(0), codebook(1), output_tensor(2)
	fwd_pass.bindings.resize(3);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = codebook.X->getDescriptorSetLayout(1);
	fwd_pass.bindings[2] = output_tensor.X->getDescriptorSetLayout(2);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	fwd_pass.createPipeline(m_device, "shaders/quantize.spv", push_constant);

	output_tensor.ready = true;

	{
		MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
		fwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, codebook.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, output_tensor.X->getDescriptorBufferInfo());
		writer.build(fwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = input_dimensions.B * input_dimensions.H * input_dimensions.W;

	log() << "Quantize forward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	fwd_pass.cleanup(m_device->getDevice());
}
