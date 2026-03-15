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
    std::cout << "---------time (" <<#text<< "): " << passedTime << "\n";

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
	REFLECT_STRUCT_MEMBER(output_tensor)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
	REFLECT_STRUCT_MEMBER(scratchpad)
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
	REFLECT_STRUCT_MEMBER(pass_output)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
	REFLECT_STRUCT_MEMBER(input_tensor)
	REFLECT_STRUCT_MEMBER(weights)
	REFLECT_STRUCT_MEMBER(scratchpad)
REFLECT_VKMOD_END()

void Convolution_Module::run()
{
	//createImages();
	createDescriptorSetLayout();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	createComputePipeline("shaders/conv.spv", push_constant);
	//createComputePipeline("shaders/conv.spv");

	pass_output.ready = true;

	execute();
	read_results();
	cleanup();
}

void Convolution_Module::createDescriptorSets()
{
	MyDescriptorWriter writer(*descriptorSetLayout, *m_DescriptorPool);

	descriptorSets.resize(1);

	writer.writeBuffer(0, weights.X->getDescriptorBufferInfo());
	writer.writeBuffer(1, input_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(2, pass_output.X->getDescriptorBufferInfo());
	writer.writeBuffer(3, scratchpad.X->getDescriptorBufferInfo());
	
	writer.build(descriptorSets[0]);
}

void Convolution_Module::createDescriptorSetLayout()
{
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	bindings.resize(4);

	bindings[0] = weights.X->getDescriptorSetLayout(0);
	bindings[1] = input_tensor.X->getDescriptorSetLayout(1);
	bindings[2] = pass_output.X->getDescriptorSetLayout(2);
	bindings[3] = scratchpad.X->getDescriptorSetLayout(3);

	descriptorSetLayout = new MyDescriptorSetLayout(m_device, bindings);
}

void Convolution_Module::setDimensions()
{
	pass_output.dimensions = output_dimensions;
	weights.dimensions = TensorDimension{ input_dimensions.C, output_dimensions.C ,4 ,4 };
}

void Convolution_Module::createImages()
{	
	/*
	int n_indices = output_dimensions.B * output_dimensions.C * output_dimensions.H * output_dimensions.W;
	VkDeviceSize bufferSize = sizeof(float) * n_indices;
	{
		

		output_tensor.X = vulkan->create_buffer(bufferSize,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			&output_tensor);

		output_tensor.X->range = bufferSize;
	}*/

	//=============================================


	{
		int n_indices = weight_dimensions.B * weight_dimensions.C * weight_dimensions.H * weight_dimensions.W;
		VkDeviceSize bufferSize = sizeof(float) * n_indices;

		MyBufferObject stagingBuffer(m_device, sizeof(float), n_indices, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);

		f32* data = new f32[n_indices];

		int outer_stride = weight_dimensions.C * weight_dimensions.H * weight_dimensions.W;
		for (int n = 0; n < weight_dimensions.B; n++)
		{
			for (int k = 0; k < weight_dimensions.C; k++)
				for (int i = 0; i < weight_dimensions.H; i++)
					for (int j = 0; j < weight_dimensions.W; j++)
					
					{
						f32 f;
						if (k == 5)
							f = random_f32();
						else
							f = 0;

						data[(outer_stride * n) +
							(weight_dimensions.W * i) +
							(weight_dimensions.H * weight_dimensions.W * k) + j] = 1.0;
					}
		}


		stagingBuffer.writeToBuffer((void*)data);

		m_device->copyBuffer(stagingBuffer.getBuffer(), weights.X->Buffer, bufferSize);

		delete[] data;
	}

	{
		int n_indices = input_dimensions.B * input_dimensions.C * input_dimensions.H * input_dimensions.W;
		VkDeviceSize bufferSize = sizeof(float) * n_indices;

		MyBufferObject stagingBuffer(m_device, sizeof(float), n_indices, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);

		f32* data = new f32[n_indices];

		int outer_stride = input_dimensions.C * input_dimensions.H * input_dimensions.W;
		for (int n = 0; n < input_dimensions.B; n++)
		{
			for (int k = 0; k < input_dimensions.C; k++)
				for (int i = 0; i < input_dimensions.H; i++)
					for (int j = 0; j < input_dimensions.W; j++)
					{
						f32 f;

						if (k == 5)
							f = i;
						else
							f = 0;

						data[(outer_stride * n) +
							(input_dimensions.W * i) +
							(input_dimensions.H * input_dimensions.W * k) + j] = random_f32();
					}
		}


		stagingBuffer.writeToBuffer((void*)data);

		m_device->copyBuffer(stagingBuffer.getBuffer(), input_tensor.X->Buffer, bufferSize);

		delete[] data;
	}
}

void Convolution_Module::execute()
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

	uint32_t n_WorkGroups_x = output_dimensions.W; //use output image size. Each WG writes to one output pixel.
	uint32_t n_WorkGroups_y = output_dimensions.H;
	uint32_t n_WorkGroups_z = output_dimensions.B; //batch size (n_images)

	std::cout << "executing compute shader (" << n_WorkGroups_z << " x " << n_WorkGroups_x << " x " << n_WorkGroups_y << ")\n";


	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, n_WorkGroups_y, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());
	PRINT_TIMER(Convolution);
}

void Convolution_Module::read_results()
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

	cout << "\n";

	for (int i = 0; i < 10; i++)
	{
		//cout PRINTV(hit_results[i].V) << "\n";
	}

	delete[] hit_results;

}

void Convolution_Module::cleanup()
{
	descriptorSetLayout->cleanup();

	pipeline->cleanup();

	vkDestroyPipelineLayout(m_device->getDevice(), pipelineLayout, nullptr);
}


//============================================================
// Normalization Module
//

REFLECT_VKMOD_BEGIN(Normalization_Module)
	ALIAS("Normalization Layer")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_STRUCT_MEMBER(input_tensor)
	REFLECT_STRUCT_MEMBER(pass_output)
	REFLECT_STRUCT_MEMBER(mean_buffer)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
	REFLECT_STRUCT_MEMBER(var_buffer)
		REFLECT_VKMOD_MEMBER_CREATE_MEMORY()
	REFLECT_STRUCT_MEMBER(scratchpad)
	REFLECT_STRUCT_MEMBER_FORWARD(input_tensor, pass_output)
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
	read_results();
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

	std::cout << "executing compute shader (" << n_WorkGroups_x << " / " << n_WorkGroups_y << ")\n";


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

	cout << "\n";

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
	REFLECT_STRUCT_MEMBER(input_tensor)
	REFLECT_STRUCT_MEMBER(pass_output)
	REFLECT_STRUCT_MEMBER(mean_buffer)
	REFLECT_STRUCT_MEMBER(var_buffer)
	REFLECT_STRUCT_MEMBER(parameters)
	REFLECT_STRUCT_MEMBER(results_buffer)
	REFLECT_STRUCT_MEMBER_FORWARD(input_tensor, pass_output)
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

	std::cout << "executing compute shader (" << n_WorkGroups_x << " / " << n_WorkGroups_y << ")\n";

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


