#pragma once

#include "vkWorkblock.h"
#include "vkProjectionUpModule.h"
#include "vkProjectionDownModule.h"
#include "vkGeluModule.h"

struct FFN_Block : public Workblock_Module
{
	virtual void build_workflow(std::vector<Vulkan_Module*>&) override;
	virtual reflect::input<vkBufferResource>&  head_input()      override;
	virtual reflect::output<vkBufferResource>& tail_output()     override;
	virtual reflect::input<vkBufferResource>&  gradient_input()  override;
	virtual reflect::output<vkBufferResource>& gradient_output() override;

	Projection_Up_Module*   proj_up   = NULL;
	Gelu_Module*            gelu      = NULL;
	Projection_Down_Module* proj_down = NULL;

	REFLECT_VKMOD()
};
