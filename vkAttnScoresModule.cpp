#include <irrlicht.h>
#include "vkModules.h"
#include "vkAttnScoresModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Attn_Scores_Module
//

REFLECT_VKMOD_BEGIN(Attn_Scores_Module)
	ALIAS("Attention Scores")
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

void Attn_Scores_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Attn_Scores_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Attn_Scores_Module::setDimensions()
{
	u32 B = input_dimensions.B;
	u32 M = input_dimensions.C;        // seq_len = 64
	u32 D = input_dimensions.W / 3;    // embed_dim (W = 3*D from QKV)
	u32 K = D / num_heads;             // head_dim

	output_tensor.dimensions = { B, num_heads, M, M };
	grad_output.dimensions   = input_dimensions;

	pushconstants.M         = M;
	pushconstants.N         = M;
	pushconstants.K         = K;
	pushconstants.D         = D;
	pushconstants.num_heads = num_heads;
}

void Attn_Scores_Module::startup()
{
	VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };

	// fwd_pass: qkv(0), scores(1)
	fwd_pass.bindings.resize(2);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = output_tensor.X->getDescriptorSetLayout(1);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);
	fwd_pass.createPipeline(m_device, "shaders/attn_scores.spv", r);
	output_tensor.ready = true;

	log() << "Attn_Scores startup complete\n";
}

void Attn_Scores_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
}

//============================================================
// Dispatch helper
//

void Attn_Scores_Module::dispatch_forward(VkCommandBuffer cmd, u32 head_n)
{
	MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
	fwd_pass.descriptorSets.resize(1);
	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, output_tensor.X->getDescriptorBufferInfo());
	writer.build(fwd_pass.descriptorSets[0]);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fwd_pass.pipeline->getPipeline());
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	pushconstants.head_n = head_n;

	// dispatch (N/16, M/16, B) — one tile per workgroup
	uint32_t wg_x = pushconstants.N / 16;
	uint32_t wg_y = pushconstants.M / 16;
	uint32_t wg_z = input_dimensions.B;
	log() << "Attn_Scores head " << head_n << ": dispatch ("
	      << wg_x << ", " << wg_y << ", " << wg_z << ")\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
}

//============================================================
// forward / backward
//

void Attn_Scores_Module::forward()
{
	for (u32 h = 0; h < num_heads; h++)
	{
		VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
		dispatch_forward(cmd, h);
		m_device->endSingleTimeCommands(cmd);
		vkDeviceWaitIdle(m_device->getDevice());
		m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
	}
}

void Attn_Scores_Module::backward()
{
	// TODO
}
