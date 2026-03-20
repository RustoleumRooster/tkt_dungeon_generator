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
	REFLECT_VKMOD_PARAM(codebook)
	REFLECT_VKMOD_FEAT(output)
REFLECT_VKMOD_END()

void Quantize_Module::run()
{
	createDescriptorSetLayout();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	createComputePipeline("shaders/quantize.spv", push_constant);

	output.ready = true;

	execute();
	cleanup();
}

void Quantize_Module::setDimensions()
{
	codebook.dimensions = codebook_size;
	output.dimensions = { this->input_dimensions.B,this->input_dimensions.C,this->input_dimensions.H,this->input_dimensions.W };
	pushconstants.n_vectors = this->codebook_size.H;
}

void Quantize_Module::createDescriptorSets()
{
	MyDescriptorWriter writer(*descriptorSetLayout, *m_DescriptorPool);

	descriptorSets.resize(1);

	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, codebook.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, output.X->getDescriptorBufferInfo());

	writer.build(descriptorSets[0]);
}

void Quantize_Module::createDescriptorSetLayout()
{
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	bindings.resize(3);
	bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	bindings[1] = codebook.X->getDescriptorSetLayout(1);
	bindings[2] = output.X->getDescriptorSetLayout(2);

	descriptorSetLayout = new MyDescriptorSetLayout(m_device, bindings);
}

void Quantize_Module::execute()
{
	Q_START_TIMER()
	createDescriptorSets();

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		&descriptorSets[0], 0, 0);

	uint32_t n_threads = input_dimensions.B * input_dimensions.C * input_dimensions.H * input_dimensions.W;

	uint32_t n_WorkGroups_x = input_dimensions.B * input_dimensions.H * input_dimensions.W;
	uint32_t n_WorkGroups_y = 1;
	uint32_t n_WorkGroups_z = 1;

	log() << "(" << n_WorkGroups_x << " / " << n_WorkGroups_y << ")\n";

	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, n_WorkGroups_y, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	Q_PRINT_TIMER(Quantize)
}

void Quantize_Module::cleanup()
{
	descriptorSetLayout->cleanup();

	pipeline->cleanup();

	vkDestroyPipelineLayout(m_device->getDevice(), pipelineLayout, nullptr);
}
