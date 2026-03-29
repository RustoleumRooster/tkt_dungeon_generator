#include <irrlicht.h>
#include "vkModules.h"
#include "vkProjectionQKVModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Projection_QKV Module
//

REFLECT_VKMOD_BEGIN(Projection_QKV_Module)
	ALIAS("QKV Projection")
	INHERIT_FROM(Vulkan_Module)
	// Forward Pass
	REFLECT_VKMOD_FEAT(input_tensor)
	REFLECT_VKMOD_PARAM(parameters)
	REFLECT_VKMOD_FEAT(output_tensor)
	// Backward Pass
	REFLECT_VKMOD_GRAD(grad_input)
	REFLECT_VKMOD_GRAD(grad_output)
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void Projection_QKV_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Projection_QKV_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Projection_QKV_Module::setDimensions()
{
	u32 B = input_dimensions.B;
	u32 M = input_dimensions.C;  // seq_len
	u32 K = input_dimensions.W;  // embedding_dim
	u32 N = K * 3;               // Q + K + V projections

	output_tensor.dimensions = { B, M, 1, N };
	parameters.dimensions    = { 1, 1, K, N };  // weight matrix [K x N]
	grad_output.dimensions   = { B, M, 1, K };

	pushconstants.M = M;
	pushconstants.N = N;
	pushconstants.K = K;
}

void Projection_QKV_Module::initialize_parameters()
{
	u32 K = input_dimensions.W;
	u32 N = K * 3;

	// N(0, 1/sqrt(K)) — standard scaled init for projection weights
	std::mt19937 rng(42);
	std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt((float)K));

	std::vector<float> data(K * N);
	for (auto& v : data) v = dist(rng);

	upload_to_buffer(parameters.X, data);
	parameters.initialized = true;
	log() << "QKV projection parameters initialized (" << K << "x" << N << ")\n";
}

void Projection_QKV_Module::startup()
{
	VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };

	// fwd_pass: input(0), weight(1), output(2)
	fwd_pass.bindings.resize(3);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = parameters.X->getDescriptorSetLayout(1);
	fwd_pass.bindings[2] = output_tensor.X->getDescriptorSetLayout(2);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);
	fwd_pass.createPipeline(m_device, "shaders/matmul.spv", r);
	output_tensor.ready = true;

	log() << "QKV projection startup complete\n";
}

void Projection_QKV_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
}

//============================================================
// Dispatch helpers
//

void Projection_QKV_Module::dispatch_forward(VkCommandBuffer cmd)
{
	MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
	fwd_pass.descriptorSets.resize(1);
	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, parameters.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, output_tensor.X->getDescriptorBufferInfo());
	writer.build(fwd_pass.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	// dispatch (N/16, M/16, B)
	uint32_t wg_x = pushconstants.N / 16;
	uint32_t wg_y = pushconstants.M / 16;
	uint32_t wg_z = input_dimensions.B;
	log() << "QKV fwd: dispatch (" << wg_x << ", " << wg_y << ", " << wg_z << ")\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
}

//============================================================
// forward / backward
//

void Projection_QKV_Module::forward()
{
	VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
	dispatch_forward(cmd);
	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
}

void Projection_QKV_Module::backward()
{
	// TODO: matmul backward
	// dL/dW = input^T @ grad_input   — [K x M] * [M x N] = [K x N]
	// dL/dX = grad_input @ W^T       — [M x N] * [N x K] = [M x K]
}
