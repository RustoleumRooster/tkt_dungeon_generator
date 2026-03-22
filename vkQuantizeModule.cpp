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
	//Forward Pass (features)
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_FEAT(output_tensor)
	REFLECT_VKMOD_FEAT(indices_buffer)
		REFLECT_VKMOD_UINT_TYPE()
	REFLECT_VKMOD_FEAT(loss_buffer)
	REFLECT_VKMOD_PARAM(codebook)
	//Backward Pass (gradients)
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output)
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
	// one u32 index per spatial position per batch element
	indices_buffer.dimensions = { input_dimensions.B, 1, input_dimensions.H, input_dimensions.W };
	// one f32 commitment distance per spatial position per batch element
	loss_buffer.dimensions    = { input_dimensions.B, 1, input_dimensions.H, input_dimensions.W };
	grad_output.dimensions    = input_dimensions;
	pushconstants.n_vectors  = codebook_size.H;
}

void Quantize_Module::forward()
{
	// bindings: input_tensor(0), codebook(1), output_tensor(2), indices_buffer(3), loss_buffer(4)
	fwd_pass.bindings.resize(5);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = codebook.X->getDescriptorSetLayout(1);
	fwd_pass.bindings[2] = output_tensor.X->getDescriptorSetLayout(2);
	fwd_pass.bindings[3] = indices_buffer.X->getDescriptorSetLayout(3);
	fwd_pass.bindings[4] = loss_buffer.X->getDescriptorSetLayout(4);
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
		writer.writeBuffer(3, indices_buffer.X->getDescriptorBufferInfo());
		writer.writeBuffer(4, loss_buffer.X->getDescriptorBufferInfo());
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

	indices_buffer.ready = true;
	loss_buffer.ready    = true;
	fwd_pass.cleanup(m_device->getDevice());
}

void Quantize_Module::backward()
{
	// bindings: grad_input(0)    — dL/d_quantized_output from downstream
	//           indices_buffer(1) — nearest codebook index per position (saved from forward)
	//           grad_output(2)   — dL/d_encoder_output (straight-through: copy of grad_input)
	//           codebook.Y(3)    — codebook weight gradients (accumulate at selected index)
	bwd_pass.bindings.resize(4);
	bwd_pass.bindings[0] = grad_input.X->getDescriptorSetLayout(0);
	bwd_pass.bindings[1] = indices_buffer.X->getDescriptorSetLayout(1);
	bwd_pass.bindings[2] = grad_output.X->getDescriptorSetLayout(2);
	bwd_pass.bindings[3] = codebook.Y->getDescriptorSetLayout(3);
	bwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	bwd_pass.createPipeline(m_device, "shaders/quantize_grad.spv", push_constant);

	grad_output.ready = true;

	{
		MyDescriptorWriter writer(*bwd_pass.descriptorSetLayout, *m_DescriptorPool);
		bwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, grad_input.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, indices_buffer.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, grad_output.X->getDescriptorBufferInfo());
		writer.writeBuffer(3, codebook.Y->getDescriptorBufferInfo());
		writer.build(bwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipelineLayout, 0, 1, &bwd_pass.descriptorSets[0], 0, 0);

	uint32_t n_WorkGroups_x = input_dimensions.B * input_dimensions.H * input_dimensions.W;

	log() << "Quantize backward: " << n_WorkGroups_x << " workgroups\n";

	vkCmdPushConstants(commandBuffer, bwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(bwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	bwd_pass.cleanup(m_device->getDevice());
}
