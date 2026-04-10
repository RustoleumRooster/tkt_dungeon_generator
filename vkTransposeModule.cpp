#include <irrlicht.h>
#include "vkModules.h"
#include "vkTransposeModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Transpose_NCHW_NHWC_Module
//

REFLECT_VKMOD_BEGIN(Transpose_NCHW_NHWC_Module)
	ALIAS("Transpose NCHW→NHWC")
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

void Transpose_NCHW_NHWC_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Transpose_NCHW_NHWC_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Transpose_NCHW_NHWC_Module::setDimensions()
{
	u32 B = input_dimensions.B;
	u32 C = input_dimensions.C;  // embed dim (128)
	u32 H = input_dimensions.H;  // height    (8)
	u32 W = input_dimensions.W;  // width     (8)

	// Output is [B x H x W x C], stored as TensorDimension {B, H, W, C}
	output_tensor.dimensions = { B, H, W, C };
	grad_output.dimensions   = input_dimensions;  // backward transpose is NHWC → NCHW

	pushconstants.B = B;
	pushconstants.C = C;
	pushconstants.H = H;
	pushconstants.W = W;
}

void Transpose_NCHW_NHWC_Module::startup()
{
	VkPushConstantRange r{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct) };

	fwd_pass.bindings.resize(2);
	fwd_pass.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = output_tensor.X->getDescriptorSetLayout(1);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);
	fwd_pass.createPipeline(m_device, "shaders/transpose_NCHW_to_NHWC.spv", r);
	output_tensor.ready = true;

	log() << "Transpose NCHW→NHWC startup complete\n";
}

void Transpose_NCHW_NHWC_Module::cleanup_passes()
{
	fwd_pass.cleanup(m_device->getDevice());
}

//============================================================
// forward / backward
//

void Transpose_NCHW_NHWC_Module::forward()
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

	uint32_t n_workgroups = (pushconstants.B * pushconstants.C *
	                         pushconstants.H * pushconstants.W + 255) / 256;
	log() << "Transpose NCHW→NHWC fwd: " << n_workgroups << " workgroups\n";

	vkCmdPushConstants(cmd, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);
	vkCmdDispatch(cmd, n_workgroups, 1, 1);

	m_device->endSingleTimeCommands(cmd);
	vkDeviceWaitIdle(m_device->getDevice());
	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);
}

void Transpose_NCHW_NHWC_Module::backward()
{
	// TODO: backward is NHWC → NCHW (same shader with input/output swapped)
}
