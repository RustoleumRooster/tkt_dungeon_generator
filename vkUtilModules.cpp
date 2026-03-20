#include <irrlicht.h>
//#include "LightMaps.h"
#include "vkModules.h"
#include "vkUtilModules.h"
//#include "csg_classes.h"
//#include "utils.h"
//#include "geometry_scene.h"
#include "soa.h"
//#include "my_reflected_nodes.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"
#include <chrono>
//#include "vkAreaLightModule.h"
//#include "vkBouncedLightModule.h"

std::chrono::steady_clock::time_point startTime;
std::chrono::steady_clock::time_point currentTime;
float passedTime;

#define START_TIMER() startTime = std::chrono::high_resolution_clock::now();
#define PRINT_TIMER(text) currentTime = std::chrono::high_resolution_clock::now(); \
    passedTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count(); \
    log() <<  passedTime << " s \n";

using namespace irr;
using namespace core;
using namespace std;

extern IrrlichtDevice* device;

//==========================================
// Create Tensors
//

REFLECT_VKMOD_BEGIN(Create_Tensor_Module)
	ALIAS("Create Tensor")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_VKMOD_MEMBER(output_tensor)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
	REFLECT_VKMOD_MEMBER(scratchpad)
REFLECT_VKMOD_END()

void Create_Tensor_Module::run()
{
	//if (!load_resources())
	//	return;

	createImages(true);

	output_tensor.ready = true;
}

void Create_Tensor_Module::initialize(Vulkan_App* vulkan)
{
	Vulkan_Module::initialize(vulkan);
}

void Create_Tensor_Module::setDimensions()
{
	output_tensor.dimensions = dimensions;
}

void Create_Tensor_Module::createImages(bool random_data)
{
	/*
	int n_indices = dimensions.B * dimensions.C * dimensions.H * dimensions.W;
	VkDeviceSize bufferSize = sizeof(float) * n_indices;

	output_tensor.X = vulkan->create_buffer(bufferSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		&output_tensor);
		*/
	//////////////////////

	VkDeviceSize sc_bufferSize = sizeof(aligned_vec3) * 512;

	scratchpad.X = vulkan->create_buffer(sc_bufferSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	scratchpad.ready = true;
}

//=======================================================
// Convolution Module
//

REFLECT_VKMOD_BEGIN(Convolution_Module)
	ALIAS("Convolution Layer")
	INHERIT_FROM(Vulkan_Module)
REFLECT_VKMOD_FORWARD_PASS()
	REFLECT_VKMOD_MEMBER(pass_output)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
	REFLECT_VKMOD_MEMBER(weights)
	REFLECT_VKMOD_MEMBER(input_tensor)
	REFLECT_VKMOD_MEMBER(scratchpad)
REFLECT_VKMOD_BACKWARD_PASS()
	REFLECT_VKMOD_MEMBER(grad_input)
	REFLECT_VKMOD_MEMBER(grad_output)
	REFLECT_VKMOD_MEMBER(grad_weights)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
REFLECT_VKMOD_END()

//============================================================
// Pass helpers
//

void Convolution_Module::Pass::createPipeline(MyDevice* device, const char* spv,
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

void Convolution_Module::Pass::cleanup(VkDevice device)
{
	descriptorSetLayout->cleanup();
	pipeline->cleanup();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

//============================================================
// Module
//

void Convolution_Module::setDimensions()
{
	pass_output.dimensions    = output_dimensions;
	weights.dimensions        = { input_dimensions.C, output_dimensions.C, pushconstants.k, pushconstants.k };
	grad_output.dimensions    = input_dimensions;
	grad_weights.dimensions   = weights.dimensions;

	pushconstants.c_in         = input_dimensions.C;
	pushconstants.c_out        = output_dimensions.C;
	pushconstants.img_size_in  = input_dimensions.H;
	pushconstants.img_size_out = output_dimensions.H;
	pushconstants.n            = input_dimensions.B;
}

void Convolution_Module::run()
{
	forward();
}

void Convolution_Module::forward()
{
	// bindings: weights(0), input_tensor(1), pass_output(2), scratchpad(3)
	fwd_pass.bindings.resize(4);
	fwd_pass.bindings[0] = weights.X->getDescriptorSetLayout(0);
	fwd_pass.bindings[1] = input_tensor.X->getDescriptorSetLayout(1);
	fwd_pass.bindings[2] = pass_output.X->getDescriptorSetLayout(2);
	fwd_pass.bindings[3] = scratchpad.X->getDescriptorSetLayout(3);
	fwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, fwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	const char* spv = "shaders/conv128.spv";
	if (output_dimensions.C <= 32)
		spv = "shaders/conv32.spv";
	else if (output_dimensions.C <= 64)
		spv = "shaders/conv64.spv";

	fwd_pass.createPipeline(m_device, spv, push_constant);

	pass_output.ready = true;

	{
		MyDescriptorWriter writer(*fwd_pass.descriptorSetLayout, *m_DescriptorPool);
		fwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, weights.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, input_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, pass_output.X->getDescriptorBufferInfo());
		writer.writeBuffer(3, scratchpad.X->getDescriptorBufferInfo());
		writer.build(fwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		fwd_pass.pipelineLayout, 0, 1, &fwd_pass.descriptorSets[0], 0, 0);

	// one workgroup per output pixel; batch in z
	uint32_t n_WorkGroups_x = output_dimensions.W;
	uint32_t n_WorkGroups_y = output_dimensions.H;
	uint32_t n_WorkGroups_z = output_dimensions.B;

	log() << "Conv forward: (" << n_WorkGroups_z << " x " << n_WorkGroups_x << " x " << n_WorkGroups_y << ")\n";

	vkCmdPushConstants(commandBuffer, fwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, n_WorkGroups_y, n_WorkGroups_z);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(fwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	fwd_pass.cleanup(m_device->getDevice());
}

void Convolution_Module::backward()
{
	backward_A();
	backward_B();
}

void Convolution_Module::backward_A()
{
	// bindings: weights(0), grad_input(1), grad_output(2)
	// shader computes dL/d(input) given upstream gradient grad_input and weights
	bwd_pass.bindings.resize(3);
	bwd_pass.bindings[0] = weights.X->getDescriptorSetLayout(0);
	bwd_pass.bindings[1] = grad_input.X->getDescriptorSetLayout(1);
	bwd_pass.bindings[2] = grad_output.X->getDescriptorSetLayout(2);
	bwd_pass.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	bwd_pass.createPipeline(m_device, "shaders/conv_grad.spv", push_constant);

	grad_output.ready = true;

	{
		MyDescriptorWriter writer(*bwd_pass.descriptorSetLayout, *m_DescriptorPool);
		bwd_pass.descriptorSets.resize(1);
		writer.writeBuffer(0, weights.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, grad_input.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, grad_output.X->getDescriptorBufferInfo());
		writer.build(bwd_pass.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass.pipelineLayout, 0, 1, &bwd_pass.descriptorSets[0], 0, 0);

	// dispatch over input spatial dims (gradient flows back to input)
	uint32_t n_WorkGroups_x = input_dimensions.W;
	uint32_t n_WorkGroups_y = input_dimensions.H;
	uint32_t n_WorkGroups_z = input_dimensions.B;

	log() << "Conv backward: (" << n_WorkGroups_z << " x " << n_WorkGroups_x << " x " << n_WorkGroups_y << ")\n";

	vkCmdPushConstants(commandBuffer, bwd_pass.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, n_WorkGroups_y, n_WorkGroups_z);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(bwd_pass.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	bwd_pass.cleanup(m_device->getDevice());
}

void Convolution_Module::backward_B()
{
	// bindings: input_tensor(0), grad_input(1), grad_weights(2)
	// shader computes dL/dW: each thread owns one weight and sums over (N, oh, ow)
	bwd_pass_B.bindings.resize(3);
	bwd_pass_B.bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	bwd_pass_B.bindings[1] = grad_input.X->getDescriptorSetLayout(1);
	bwd_pass_B.bindings[2] = grad_weights.X->getDescriptorSetLayout(2);
	bwd_pass_B.descriptorSetLayout = new MyDescriptorSetLayout(m_device, bwd_pass_B.bindings);

	VkPushConstantRange push_constant;
	push_constant.offset     = 0;
	push_constant.size       = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	bwd_pass_B.createPipeline(m_device, "shaders/conv_grad_b.spv", push_constant);

	grad_weights.ready = true;

	{
		MyDescriptorWriter writer(*bwd_pass_B.descriptorSetLayout, *m_DescriptorPool);
		bwd_pass_B.descriptorSets.resize(1);
		writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
		writer.writeBuffer(1, grad_input.X->getDescriptorBufferInfo());
		writer.writeBuffer(2, grad_weights.X->getDescriptorBufferInfo());
		writer.build(bwd_pass_B.descriptorSets[0]);
	}

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_B.pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		bwd_pass_B.pipelineLayout, 0, 1, &bwd_pass_B.descriptorSets[0], 0, 0);

	uint32_t n_weights      = pushconstants.k * pushconstants.k * pushconstants.c_in * pushconstants.c_out;
	uint32_t n_WorkGroups_x = (n_weights + 255) / 256;

	log() << "Conv backward_B: " << n_WorkGroups_x << " workgroups (" << n_weights << " weights)\n";

	vkCmdPushConstants(commandBuffer, bwd_pass_B.pipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, 1, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(bwd_pass_B.descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	bwd_pass_B.cleanup(m_device->getDevice());
}


//============================================================
// Normalization Module
//

REFLECT_VKMOD_BEGIN(Normalization_Module)
	ALIAS("Normalization Layer")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_VKMOD_MEMBER(input_tensor)
	REFLECT_VKMOD_MEMBER(pass_output)
	REFLECT_VKMOD_MEMBER(mean_buffer)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
	REFLECT_VKMOD_MEMBER(var_buffer)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
	REFLECT_VKMOD_MEMBER(scratchpad)
	REFLECT_VKMOD_MEMBER_OUTPUT_IN_PLACE(input_tensor, pass_output)
REFLECT_VKMOD_END()

void Normalization_Module::run()
{
	//createBuffer();
	createDescriptorSetLayout();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	createComputePipeline("shaders/mean_var.spv", push_constant);

	execute();
	//read_results();
	cleanup();

	mean_buffer.ready = true;
	var_buffer.ready = true;
	pass_output.ready = true;
	pass_output.X = input_tensor.X;
}

void Normalization_Module::setDimensions()
{
	mean_buffer.dimensions = { 1,1,1,this->input_dimensions.C };
	var_buffer.dimensions = { 1,1,1,this->input_dimensions.C };
}

void Normalization_Module::createBuffer()
{
	int n_indices = 128;
	VkDeviceSize bufferSize = sizeof(float) * n_indices;

	mean_buffer.X = vulkan->create_buffer(bufferSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	mean_buffer.X->range = bufferSize;

	var_buffer.X = vulkan->create_buffer(bufferSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	var_buffer.X->range = bufferSize;
}

void Normalization_Module::createDescriptorSets()
{
	MyDescriptorWriter writer(*descriptorSetLayout, *m_DescriptorPool);

	descriptorSets.resize(1);

	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, mean_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, var_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(3, scratchpad.X->getDescriptorBufferInfo());

	writer.build(descriptorSets[0]);
}

void Normalization_Module::createDescriptorSetLayout()
{
	bindings.resize(4);
	bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	bindings[1] = mean_buffer.X->getDescriptorSetLayout(1);
	bindings[2] = var_buffer.X->getDescriptorSetLayout(2);
	bindings[3] = scratchpad.X->getDescriptorSetLayout(3);

	descriptorSetLayout = new MyDescriptorSetLayout(m_device, bindings);
}

void Normalization_Module::execute()
{
	START_TIMER()
	createDescriptorSets();

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		&descriptorSets[0], 0, 0);

	uint32_t work_length = 1;
	uint32_t work_height = 1;

	uint32_t n_WorkGroups_x = 128; //one per channel
	uint32_t n_WorkGroups_y = 1;
	uint32_t n_WorkGroups_z = 1;

	log() << "(" << n_WorkGroups_x << " / " << n_WorkGroups_y << ")\n";


	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, n_WorkGroups_y, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());
	PRINT_TIMER(Calc Mean and Var)
}

void Normalization_Module::read_results()
{
	aligned_vec3* hit_results = NULL;

	uint16_t bSize = 256 * 2;
	VkDeviceSize bufferSize = sizeof(aligned_vec3) * bSize;

	hit_results = new aligned_vec3[bSize];
	for (int i = 0; i < bSize; i++) {
		hit_results[i].V = vector3df{ 0,0,0 };
	}

	MyBufferObject stagingBuffer(m_device, sizeof(aligned_vec3), 256 * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);

	m_device->copyBuffer(scratchpad.X->Buffer, stagingBuffer.getBuffer(), sizeof(aligned_vec3) * 256 * 2);

	stagingBuffer.readFromBuffer((void*)hit_results);

	for (int i = 0; i < 256; i++)
	{
		//cout << hit_results[i].V.X << " ";
		//graph.lines.push_back(line3df(hit_results[i].V, hit_results[256 + i].V));
	}

	log() << "\n";

	for (int i = 0; i < 10; i++)
	{
		//cout PRINTV(hit_results[i].V) << "\n";
	}

	delete[] hit_results;
	
}

void Normalization_Module::cleanup()
{
	descriptorSetLayout->cleanup();

	pipeline->cleanup();

	vkDestroyPipelineLayout(m_device->getDevice(), pipelineLayout, nullptr);
}


//============================================================
// Activation Module
//

REFLECT_VKMOD_BEGIN(Activation_Module)
	ALIAS("Activation Layer")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_VKMOD_MEMBER(input_tensor)
	REFLECT_VKMOD_MEMBER(pass_output)
	REFLECT_VKMOD_MEMBER(mean_buffer)
	REFLECT_VKMOD_MEMBER(var_buffer)
	REFLECT_VKMOD_MEMBER(parameters)
	REFLECT_VKMOD_MEMBER(results_buffer)
	REFLECT_VKMOD_MEMBER_OUTPUT_IN_PLACE(input_tensor, pass_output)
REFLECT_VKMOD_END()

void Activation_Module::setDimensions()
{
	parameters.dimensions = { 1,1,2,input_dimensions.C }; //two params per channel (gamma, beta)
}

void Activation_Module::run()
{
	createDescriptorSetLayout();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	createComputePipeline("shaders/activate.spv", push_constant);

	pass_output.ready = true;
	pass_output.X = input_tensor.X;

	execute();
	cleanup();
}

void Activation_Module::createDescriptorSets()
{
	MyDescriptorWriter writer(*descriptorSetLayout, *m_DescriptorPool);

	descriptorSets.resize(1);

	writer.writeBuffer(0, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, mean_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, var_buffer.X->getDescriptorBufferInfo());
	writer.writeBuffer(3, parameters.X->getDescriptorBufferInfo());
	writer.writeBuffer(4, results_buffer.X->getDescriptorBufferInfo());

	writer.build(descriptorSets[0]);
}

void Activation_Module::createDescriptorSetLayout()
{
	bindings.resize(5);
	bindings[0] = input_tensor.X->getDescriptorSetLayout(0);
	bindings[1] = mean_buffer.X->getDescriptorSetLayout(1);
	bindings[2] = var_buffer.X->getDescriptorSetLayout(2);
	bindings[3] = parameters.X->getDescriptorSetLayout(3);
	bindings[4] = results_buffer.X->getDescriptorSetLayout(4);

	descriptorSetLayout = new MyDescriptorSetLayout(m_device, bindings);
}

void Activation_Module::execute()
{
	START_TIMER()
	createDescriptorSets();

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		&descriptorSets[0], 0, 0);

	uint32_t n_threads = input_dimensions.B * input_dimensions.C * input_dimensions.H * input_dimensions.W;

	uint32_t n_WorkGroups_x = n_threads / 256;
	uint32_t n_WorkGroups_y = 1;
	uint32_t n_WorkGroups_z = 1;

	log() << "(" << n_WorkGroups_x << " / " << n_WorkGroups_y << ")\n";

	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, n_WorkGroups_y, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());

	PRINT_TIMER(Normalize and Activate)
}

void Activation_Module::cleanup()
{
	descriptorSetLayout->cleanup();

	pipeline->cleanup();

	vkDestroyPipelineLayout(m_device->getDevice(), pipelineLayout, nullptr);
}


