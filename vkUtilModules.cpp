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
//#include "vkAreaLightModule.h"
//#include "vkBouncedLightModule.h"

using namespace irr;
using namespace core;
using namespace std;

extern IrrlichtDevice* device;

//==========================================
// Create Lightmap Images
//

REFLECT_VKMOD_BEGIN(Create_Tensor_Module)
	ALIAS("Create Lightmap Images")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_STRUCT_MEMBER(output_tensor)
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

void Create_Tensor_Module::createImages(bool random_data)
{
	int n_indices = dimensions.B * dimensions.C * dimensions.H * dimensions.W;
	VkDeviceSize bufferSize = sizeof(float) * n_indices;

	output_tensor.X = vulkan->create_buffer(bufferSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		&output_tensor);

	if (random_data)
	{
		MyBufferObject stagingBuffer(m_device, sizeof(float), n_indices, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);

		f32* data = new f32[n_indices];

		int stride = dimensions.H * dimensions.W;
		for (int i = 0; i < dimensions.B * dimensions.C; i++)
		{
			for (int j = 0; j < stride; j++)
				data[stride * i + j] = i;
		}

		stagingBuffer.writeToBuffer((void*)data);

		m_device->copyBuffer(stagingBuffer.getBuffer(), output_tensor.X->Buffer, bufferSize);

		delete[] data;
	}

	//////////////////////

	VkDeviceSize sc_bufferSize = sizeof(aligned_vec3) * 512;

	scratchpad.X = vulkan->create_buffer(sc_bufferSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &scratchpad);

	scratchpad.X->range = sc_bufferSize;
	scratchpad.ready = true;
}

//=======================================================
// Convolution Module
//

REFLECT_VKMOD_BEGIN(Dummy_Consumer)
	INHERIT_FROM(Vulkan_Module)
	REFLECT_STRUCT_MEMBER(input_tensor)
REFLECT_VKMOD_END()

REFLECT_VKMOD_BEGIN(Convolution_Module)
	ALIAS("Convolution Layer")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_STRUCT_MEMBER(output_tensor)
	REFLECT_STRUCT_MEMBER(input_tensor)
	REFLECT_STRUCT_MEMBER(weights)
	REFLECT_STRUCT_MEMBER(scratchpad)
REFLECT_VKMOD_END()

void Convolution_Module::run()
{
	createImages();
	createDescriptorSetLayout();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(pushconstant_struct);
	push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	createComputePipeline("shaders/conv.spv", push_constant);
	//createComputePipeline("shaders/conv.spv");
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
	writer.writeBuffer(2, output_tensor.X->getDescriptorBufferInfo());
	writer.writeBuffer(3, scratchpad.X->getDescriptorBufferInfo());
	
	writer.build(descriptorSets[0]);
}

void Convolution_Module::createDescriptorSetLayout()
{
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	bindings.resize(4);

	bindings[0] = weights.X->getDescriptorSetLayout(0);
	bindings[1] = input_tensor.X->getDescriptorSetLayout(1);
	bindings[2] = output_tensor.X->getDescriptorSetLayout(2);
	bindings[3] = scratchpad.X->getDescriptorSetLayout(3);

	descriptorSetLayout = new MyDescriptorSetLayout(m_device, bindings);
}

void Convolution_Module::createImages()
{
	int n_indices = output_dimensions.B * output_dimensions.C * output_dimensions.H * output_dimensions.W;
	VkDeviceSize bufferSize = sizeof(float) * n_indices;

	output_tensor.X = vulkan->create_buffer(bufferSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		&output_tensor);
}

void Convolution_Module::execute()
{
	createDescriptorSets();

	VkCommandBuffer commandBuffer = m_device->beginSingleTimeCommands();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		pipeline->getPipeline());

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		&descriptorSets[0], 0, 0);

	uint32_t work_length = 1;
	uint32_t work_height = 1;

	uint32_t n_WorkGroups_x = 1;
	uint32_t n_WorkGroups_y = 1;

	std::cout << "executing compute shader (" << n_WorkGroups_x << " / " << n_WorkGroups_y << ")\n";

	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushconstant_struct), &pushconstants);

	vkCmdDispatch(commandBuffer, n_WorkGroups_x, n_WorkGroups_y, 1);

	m_device->endSingleTimeCommands(commandBuffer);

	m_DescriptorPool->freeDescriptorsSets(descriptorSets);

	vkDeviceWaitIdle(m_device->getDevice());
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
		cout << hit_results[i].V.X << " ";
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

/*
REFLECT_VKMOD_BEGIN(MultiImage_Copy_Module)
	ALIAS("Copy Images")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_STRUCT_MEMBER(images_in)
	REFLECT_STRUCT_MEMBER(images_out)
REFLECT_VKMOD_END()

void MultiImage_Copy_Module::run()
{
	for (int i = 0; i < configuration->lightmap_dimensions.size(); i++)
	{
		VkDeviceSize width = configuration->lightmap_dimensions[i].Width;
		VkDeviceSize height = configuration->lightmap_dimensions[i].Height;

		VkImage img;
		VkDeviceMemory imgMemory;
		VkImageView imgView;

		m_device->createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, img,
			imgMemory);

		imgView = m_device->createImageView(img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

		m_device->transitionImageLayout(img, VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		m_device->transitionImageLayout(images_in.X->Images[i].Image, VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

		m_device->copyImage(img, images_in.X->Images[i].Image, width, height);

		m_device->transitionImageLayout(img, VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

	}

	images_out.ready = true;
}
*/

//==========================================
// Create Texture Images
//


//==========================================
// Load Textures
//
/*
void Load_Textures_Module::initialize(Vulkan_App* vulkan)
{
	Vulkan_Module::initialize(vulkan);
	configuration = vulkan->configuration;
	driver = vulkan->driver;
}

void Load_Textures_Module::run()
{
	createImages();
}

void Load_Textures_Module::createImages()
{
	for (int i = 0; i < configuration->lightmap_dimensions.size(); i++)
	{
		VkDeviceSize width = configuration->lightmap_dimensions[i].Width;
		VkDeviceSize height = configuration->lightmap_dimensions[i].Height;
		VkDeviceSize imgSize = width * height * 4;

		VkImage img;
		VkDeviceMemory imgMemory;
		VkImageView imgView;

		MyBufferObject stagingBuffer(m_device, imgSize, 1, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);

		irr::video::IImage* pImage = driver->createImage(textures[i], core::vector2di(0, 0), configuration->lightmap_dimensions[i]);
		pImage->flip(true, false);

		irr::u8* imgDataPtr = (irr::u8*)pImage->lock();

		stagingBuffer.writeToBuffer(imgDataPtr);

		pImage->unlock();
		pImage->drop();
		m_device->createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, img,
			imgMemory);

		m_device->transitionImageLayout(img, VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		//VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL

		imgView = m_device->createImageView(img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

		lightmapImages.push_back(img);
		lightmapsMemory.push_back(imgMemory);
		lightmapImageViews.push_back(imgView);

		m_device->copyBufferToImage(stagingBuffer.getBuffer(), lightmapImages[i], static_cast<uint32_t>(width), static_cast<uint32_t>(height));

		m_device->transitionImageLayout(img, VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	}
}



//==========================================
// Download Textures
//

REFLECT_VKMOD_BEGIN(Download_Textures_Module)
	ALIAS("Download Textures")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_STRUCT_MEMBER(images_in)
REFLECT_VKMOD_END()

void Download_Textures_Module::run()
{
	//if (!load_resources())
	//	return;

	for (int i = 0; i < configuration->lightmap_dimensions.size(); i++)
	{
		VkDeviceSize width = configuration->lightmap_dimensions[i].Width;
		VkDeviceSize height = configuration->lightmap_dimensions[i].Height;
		VkDeviceSize imgSize = width * height * 4;

		MyBufferObject stagingBuffer(m_device, imgSize, 1, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);

		m_device->transitionImageLayout(images_in.X->Images[i].Image, VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

		m_device->copyImageToBuffer(stagingBuffer.getBuffer(), images_in.X->Images[i].Image, static_cast<uint32_t>(width), static_cast<uint32_t>(height));

		irr::video::IImage* pImage = driver->createImage(irr::video::ECF_A8R8G8B8, irr::core::dimension2du(width, height));

		irr::u8* imgDataPtr = (irr::u8*)pImage->lock();

		stagingBuffer.readFromBuffer(imgDataPtr);

		if (bFlip)
			pImage->flip(true, false);

		irr::video::ITexture* tex = driver->addTexture(irr::io::path("image name"), pImage);
		textures.push_back(tex);

		pImage->unlock();
		pImage->drop();

		m_device->transitionImageLayout(images_in.X->Images[i].Image, VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}

	//for (auto& img : images_in.X->Images)
	//	img.destroy(m_device->getDevice());
}

//==========================================
// Download Textures
//

REFLECT_VKMOD_BEGIN(Download_TextureArray_Module)
	ALIAS("Download Textures [Array]")
	INHERIT_FROM(Vulkan_Module)
	REFLECT_STRUCT_MEMBER(images_in)
REFLECT_VKMOD_END()

void Download_TextureArray_Module::run()
{
	
	m_device->transitionImageArrayLayout(images_in.X->n_images,images_in.X->Image, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

	for (int i = 0; i < configuration->lightmap_dimensions.size(); i++)
	{
		VkDeviceSize width = configuration->lightmap_dimensions[i].Width;
		VkDeviceSize height = configuration->lightmap_dimensions[i].Height;
		VkDeviceSize imgSize = width * height * 4;

		MyBufferObject stagingBuffer(m_device, imgSize, 1, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);

		m_device->copyImageLayerToBuffer(i,stagingBuffer.getBuffer(), images_in.X->Image, static_cast<uint32_t>(width), static_cast<uint32_t>(height));

		irr::video::IImage* pImage = driver->createImage(irr::video::ECF_A8R8G8B8, irr::core::dimension2du(width, height));

		irr::u8* imgDataPtr = (irr::u8*)pImage->lock();

		stagingBuffer.readFromBuffer(imgDataPtr);

		if (bFlip)
			pImage->flip(true, false);

		irr::video::ITexture* tex = driver->addTexture(irr::io::path("image name"), pImage);
		textures.push_back(tex);

		pImage->unlock();
		pImage->drop();

		//m_device->transitionImageLayout(images_in.X->Image, VK_FORMAT_R8G8B8A8_UNORM,
		//	VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}
	
	//images_in.X->destroy(m_device->getDevice());
}


void MultiImage_Copy_Module::initialize(Vulkan_App* vulkan)
{
	Vulkan_Module::initialize(vulkan);
}

void Download_Textures_Module::initialize(Vulkan_App* vulkan)
{
	Vulkan_Module::initialize(vulkan);
	driver = vulkan->driver;
}

void Download_TextureArray_Module::initialize(Vulkan_App* vulkan)
{
	Vulkan_Module::initialize(vulkan);
	driver = vulkan->driver;
}
*/


