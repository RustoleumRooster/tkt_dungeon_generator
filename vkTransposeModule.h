#pragma once
#ifndef _VK_TRANSPOSE_MODULE_H_
#define _VK_TRANSPOSE_MODULE_H_

#include <irrlicht.h>
#include <vector>
#include "vkDevice.h"
#include "vkModules.h"
#include <vulkan/vulkan.h>
#include "vkDescriptors.h"
#include "vkBufferObject.h"
#include "vkComputePipeline.h"
#include "reflect_custom_types.h"

class MyDescriptorPool;

// Transposes [B x C x H x W] → [B x H x W x C]
// Converts NCHW (quantize output) to NHWC (attention block input).

struct Transpose_NCHW_NHWC_Module : public Vulkan_Module
{
	// input_dimensions: [B x C x H x W]  e.g. [16 x 128 x 8 x 8]
	TensorDimension input_dimensions{ 16, 128, 8, 8 };

	Transpose_NCHW_NHWC_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		u32 B;
		u32 C;  // channel / embed dim
		u32 H;  // height
		u32 W;  // width
	};

	pushconstant_struct pushconstants{ 16, 128, 8, 8 };

	struct Pass
	{
		VkPipelineLayout                          pipelineLayout;
		MyDescriptorSetLayout*                    descriptorSetLayout = NULL;
		ComputePipeline*                          pipeline            = NULL;
		std::vector<VkDescriptorSet>              descriptorSets;
		std::vector<VkDescriptorSetLayoutBinding> bindings;

		void createPipeline(MyDevice*, const char* spv, VkPushConstantRange);
		void cleanup(VkDevice);
	};

	Pass fwd_pass;

	virtual void setDimensions() override;
	virtual void startup() override;
	virtual void forward() override;
	virtual void backward() override;
	virtual void cleanup_passes() override;

	reflect::input<vkBufferResource>  input_tensor;   // [B x C x H x W]
	reflect::output<vkBufferResource> output_tensor;  // [B x H x W x C]
	reflect::input<vkBufferResource>  grad_input;
	reflect::output<vkBufferResource> grad_output;    // TODO: transpose back (NHWC → NCHW)

	REFLECT_VKMOD()
};

#endif
