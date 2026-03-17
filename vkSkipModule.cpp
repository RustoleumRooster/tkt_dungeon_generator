#include <irrlicht.h>
#include "vkModules.h"
#include "vkSkipModule.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

//============================================================
// Skip Module
//

REFLECT_VKMOD_BEGIN(Skip_Module)
	ALIAS("Skip")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_STRUCT_MEMBER(input_tensor)
	REFLECT_STRUCT_MEMBER(pass_output)
	REFLECT_VKMOD_MEMBER_OUTPUT_IN_PLACE(input_tensor, pass_output)
	REFLECT_STRUCT_MEMBER(skip_output)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
REFLECT_VKMOD_END()

void Skip_Module::setDimensions()
{
	skip_output.dimensions = input_dimensions;
}

void Skip_Module::run()
{
	VkDeviceSize bufferSize = sizeof(float) *
		input_dimensions.B * input_dimensions.C *
		input_dimensions.H * input_dimensions.W;

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	VkBufferCopy copyRegion{};
	copyRegion.size = bufferSize;
	vkCmdCopyBuffer(commandBuffer, input_tensor.X->Buffer, skip_output.X->Buffer, 1, &copyRegion);

	m_device->endSingleTimeCommands(commandBuffer);

	pass_output.X = input_tensor.X;
	pass_output.ready = true;
	skip_output.ready = true;
}
