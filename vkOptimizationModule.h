#pragma once
#ifndef _VK_OPTIMIZATION_MODULE_H_
#define _VK_OPTIMIZATION_MODULE_H_

#include <irrlicht.h>
#include <vector>
#include "vkDevice.h"
#include "vkModules.h"
#include <vulkan/vulkan.h>
#include "vkDescriptors.h"
#include "vkBufferObject.h"
#include "vkComputePipeline.h"

class MyDescriptorPool;

struct Optimization_Module : public Vulkan_Module
{
	Optimization_Module()
	{
		set_ptrs();
	}

	struct pushconstant_struct
	{
		float lr;        // learning rate
		float B1_t;      // beta1^t  (caller updates each step)
		float B2_t;      // beta2^t
		u32   n_elements;
	};

	pushconstant_struct pushconstants
	{
		0.001f,  // lr
		0.9f,    // B1_t  (= beta1 at step 1)
		0.999f,  // B2_t  (= beta2 at step 1)
		0        // n_elements — set by workflow after plan_memory
	};

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

	Pass bwd_pass;

	// Set by Vulkan_Workflow after plan_memory().
	// Points at the full packed parameter / moment buffers.
	vkBufferResource* params_buf = NULL;
	vkBufferResource* grads_buf  = NULL;
	vkBufferResource* m_buf      = NULL;
	vkBufferResource* v_buf      = NULL;

	virtual void backward() override;

	REFLECT_VKMOD()
};

#endif
