#include <irrlicht.h>
//#include "LightMaps.h"
#include "vkModules.h"
#include "Reflection.h"
//#include "vkUtilModules.h"
//#include "csg_classes.h"
//#include "utils.h"
//#include "geometry_scene.h"
//#include "soa.h"
//#include "vkSunlightModule.h"
//#include "vkAreaLightModule.h"
//#include "my_reflected_nodes.h"
#include <vulkan/vulkan.h>
#include "reflect_custom_types.h"

using namespace irr;
using namespace core;
using namespace std;

extern IrrlichtDevice* device;
extern VkDevice vk_device = NULL;

#define PRINTV(x) << x.X <<","<<x.Y<<","<<x.Z<<" "

u64 random_number()
{
	static random_device dev;
	static mt19937 rng(dev());

	std::uniform_int_distribution<u64> dist(0, std::numeric_limits<u64>::max());

	return dist(rng);
}

float random_f32()
{
	static random_device dev;
	static mt19937 rng(dev());

	std::uniform_real_distribution<float> dist(0, 1.0);

	return dist(rng);
}

static std::vector<Vulkan_Module*>* all_vulkan_modules = NULL;

reflect::Vulkan_Reflection_Factory Vulkan_Module::factory{};

vector<Vulkan_Module*>* get_all_vk_modules()
{
	return  all_vulkan_modules;
}

/*
REFLECT_CUSTOM_STRUCT_BEGIN_TEMPLATE(int, reflect::input)
	REFLECT_STRUCT_MEMBER(my_uid)
	REFLECT_STRUCT_MEMBER(input_uids)
REFLECT_STRUCT_END()*/

REFLECT_STRUCT3_BEGIN(vkBufferResource)
	REFLECT_STRUCT_MEMBER(Buffer)
	REFLECT_STRUCT_MEMBER(BufferMemory)
REFLECT_STRUCT_END()

REFLECT_STRUCT3_BEGIN(vkMultiImageResource)
	REFLECT_STRUCT_MEMBER(Images)
REFLECT_STRUCT_END()

REFLECT_STRUCT3_BEGIN(vkImageResource)
	REFLECT_STRUCT_MEMBER(Image)
	REFLECT_STRUCT_MEMBER(ImageMemory)
	REFLECT_STRUCT_MEMBER(ImageView)
REFLECT_STRUCT_END()

REFLECT_STRUCT3_BEGIN(vkImageSubresource)
	REFLECT_STRUCT_MEMBER(Image)
	REFLECT_STRUCT_MEMBER(ImageMemory)
	REFLECT_STRUCT_MEMBER(ImageView)
REFLECT_STRUCT_END()

REFLECT_CUSTOM_STRUCT_BEGIN_TEMPLATE(vkMultiImageResource, reflect::input)
	INHERIT_FROM(reflect::input_type)
	REFLECT_STRUCT_MEMBER(input_uid)
	REFLECT_STRUCT_MEMBER(input_member)
REFLECT_STRUCT_END()

REFLECT_CUSTOM_STRUCT_BEGIN_TEMPLATE(vkMultiImageResource, reflect::output)
	INHERIT_FROM(reflect::output_type)
	REFLECT_STRUCT_MEMBER(output_uids)
	REFLECT_STRUCT_MEMBER(output_member);
REFLECT_STRUCT_END()

REFLECT_CUSTOM_STRUCT_BEGIN_TEMPLATE(vkBufferResource, reflect::output)
	INHERIT_FROM(reflect::output_type)
	REFLECT_STRUCT_MEMBER(output_uids)
	REFLECT_STRUCT_MEMBER(output_member);
REFLECT_STRUCT_END()

REFLECT_CUSTOM_STRUCT_BEGIN_TEMPLATE(vkBufferResource, reflect::input)
	INHERIT_FROM(reflect::input_type)
	REFLECT_STRUCT_MEMBER(input_uid)
	REFLECT_STRUCT_MEMBER(input_member)
REFLECT_STRUCT_END()

REFLECT_STRUCT3_BEGIN(reflect::input_type)
REFLECT_STRUCT_END()

REFLECT_STRUCT3_BEGIN(reflect::output_type)
REFLECT_STRUCT_END()

REFLECT_STRUCT3_BEGIN(reflect::parameter_type)
REFLECT_STRUCT_END()

REFLECT_CUSTOM_STRUCT_BEGIN_TEMPLATE(vkBufferResource, reflect::parameter)
	INHERIT_FROM(reflect::parameter_type)
REFLECT_STRUCT_END()

REFLECT_STRUCT_BEGIN(reflect::vector2i)
	REFLECT_STRUCT_MEMBER(X)
	REFLECT_STRUCT_MEMBER(Y)
REFLECT_STRUCT_END()

namespace reflect
{
	template <>
	TypeDescriptor* getPrimitiveDescriptor<VkImage>() {
		static TypeDescriptor_U64 typeDesc;
		return &typeDesc;
	}
	template <>
	TypeDescriptor* getPrimitiveDescriptor<VkImageView>() {
		static TypeDescriptor_U64 typeDesc;
		return &typeDesc;
	}
	template <>
	TypeDescriptor* getPrimitiveDescriptor<VkDeviceMemory>() {
		static TypeDescriptor_U64 typeDesc;
		return &typeDesc;
	}
	template <>
	TypeDescriptor* getPrimitiveDescriptor<VkBuffer>() {
		static TypeDescriptor_U64 typeDesc;
		return &typeDesc;
	}
	/*
	void connect(output<vkTensorResource>* in, input<vkMultiImageResource>* out)
	{
		ImageArray_To_MultiImage_Module* link_module = (ImageArray_To_MultiImage_Module* )in->owner->vulkan->create_module<ImageArray_To_MultiImage_Module>();
		
		connect(in, &link_module->images_in);
		connect(&link_module->images_out,out);
	}

	void connect(output<vkMultiImageResource>* in, input<vkTensorResource>* out)
	{
		MultiImage_To_ImageArray_Module* link_module = (MultiImage_To_ImageArray_Module*)in->owner->vulkan->create_module<MultiImage_To_ImageArray_Module>();
	
		connect(in, &link_module->images_in);
		connect(&link_module->images_out, out);
	}*/

}

Vulkan_App::Vulkan_App(video::IVideoDriver* driver)
: driver{ driver }
{
	initVulkan();

	all_vulkan_modules = &all_modules;
}

void Vulkan_App::initVulkan()
{
	m_device = new MyDevice();

	vk_device = m_device->getDevice();

	createDescriptorPool();
	createCommandBuffers();

}

void Vulkan_App::cleanup()
{
	for (Vulkan_Module* vk : all_modules)
	{
		reflect::TypeDescriptor_Struct* tD = vk->GetDynamicReflection();

		if (vk->my_status == VK_MODULE_NOT_RAN)
		{
			vk->log() << tD->name << " did not run\n";
		}
	}
	bool UnusedResources = false;
	for (vkMemoryResource* res : resources)
	{
		if (res->status != RESOURCE_DESTROYED)
		{
			if (!UnusedResources)
			{
				UnusedResources = true;
				cout << "Resources not consumed:\n";
			}/*
			reflect::TypeDescriptor_Struct* res_tD = res->GetDynamicReflection();
			reflect::TypeDescriptor_Struct* res_owner_tD = res->owner->owner->GetDynamicReflection();
			cout << "  " << res_owner_tD->name << "::" << res_tD->name << "\n";

			for (reflect::input_type* in : res->consumers)
			{
				reflect::TypeDescriptor_Struct* tD = in->owner->GetDynamicReflection();
				cout << "    -> " << tD->name << "\n";
			}*/
			res->status = RESOURCE_DESTROYED;
			res->destroy(m_device->getDevice());
		}
	}
	if (!UnusedResources)
	{
		cout << "All resources consumed\n";
	}
	cout << "Finished running workflow\n";

	m_DescriptorPool->cleanup();
	m_device->cleanup();

	for (Vulkan_Module* module : all_modules)
	{
		//delete module;
	}
}

Vulkan_Module::Vulkan_Module()
	: vulkan{ vulkan }
{
	m_uid = random_number();
}

void Vulkan_Module::initialize(Vulkan_App* vulkan)
{
	this->vulkan = vulkan;
	vulkan->all_modules.push_back(this);
	m_device = vulkan->m_device;
	m_DescriptorPool = vulkan->m_DescriptorPool;

	reflect::TypeDescriptor_Struct* tD = GetDynamicReflection();

	
	for (reflect::Member& m : tD->members)
	{
		reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;

		if (m_tD->inherited_type == &reflect::output_type::Reflection)
		{
			reflect::output_type* out = (reflect::output_type*)m.get(this);
			out->vulkan = vulkan;
		}
		else if (m_tD->inherited_type == &reflect::parameter_type::Reflection)
		{
			reflect::parameter_type* p = (reflect::parameter_type*)m.get(this);
			p->vulkan = vulkan;
		}
	}
}

bool Vulkan_Module::all_resources_ready()
{
	reflect::TypeDescriptor_Struct* tD = GetDynamicReflection();

	for (reflect::Member& m : tD->members)
	{
		reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;

		if (m_tD->inherited_type == &reflect::input_type::Reflection)
		{
			reflect::input_type* in = (reflect::input_type*)m.get(this);
			if (in->ready == false)
				return false;
		}
	}
	return true;
}

bool Vulkan_Module::signaled()
{
	if (my_status != VK_MODULE_NOT_RAN)
		return false;

	if(all_resources_ready())
	{
		reflect::TypeDescriptor_Struct* tD = GetDynamicReflection();

		my_status = VK_MODULE_RAN;

		log() << tD->name << ": running \n";

		for (reflect::Member& m : tD->members)
		{
			reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;

			if (m_tD->inherited_type == &reflect::input_type::Reflection)
			{
				reflect::input_type* in = (reflect::input_type*)m.get(this);

			}
			else if (m_tD->inherited_type == &reflect::output_type::Reflection)
			{
				reflect::output_type* out = (reflect::output_type*)m.get(this);
			
				if (m.flags & REFLECT_VKMOD_MEMBER_FLAG_CREATE_MEMORY)
				{
					out->make_buffer();
				}
			}
			else if (m_tD->inherited_type == &reflect::parameter_type::Reflection)
			{
				reflect::parameter_type* p = (reflect::parameter_type*)m.get(this);

				p->make_buffer();
			}
		}

		run();

		for (reflect::Member& m : tD->members)
		{
			reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;

			if (m_tD->inherited_type == &reflect::input_type::Reflection)
			{
				reflect::input_type* in = (reflect::input_type*)m.get(this);

				//if(in->ready)
				//	in->src_output->consume(in);
			}
			else if (m_tD->inherited_type == &reflect::output_type::Reflection)
			{
				reflect::output_type* out = (reflect::output_type*)m.get(this);
				if (out->ready)
				{
					out->signal();
				}
			}
		}
		return true;
	}
	return false;
}

void Vulkan_Module::run_and_push()
{
	if (my_status != VK_MODULE_NOT_RAN)
		return;

	if (all_resources_ready())
	{
		reflect::TypeDescriptor_Struct* tD = GetDynamicReflection();

		my_status = VK_MODULE_RAN;

		log() << tD->name << ": running \n";

		run();

		for (reflect::Member& m : tD->members)
		{
			reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;

			if (m_tD->inherited_type == &reflect::output_type::Reflection)
			{
				reflect::output_type* out = (reflect::output_type*)m.get(this);
				if (out->ready)
				{
					out->push();
				}
			}
		}
	}
}

void Vulkan_Module::set_ptrs()
{
	reflect::TypeDescriptor_Struct* tD = GetDynamicReflection();

	reflect::TypeDescriptor* input_tD = reflect::TypeResolver<reflect::input_type>::get();
	reflect::TypeDescriptor* output_tD = reflect::TypeResolver<reflect::output_type>::get();

	for (reflect::Member& m : tD->members)
	{
		reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;

		if(m_tD->inherited_type == input_tD)
		{
			reflect::input_type* in = (reflect::input_type*)m.get(this);
			in->owner = this;

			if (m.forward_output != 0xFF)
			{
				reflect::Member& forward_m = tD->members[m.forward_output];
				in->forward_output = (reflect::output_type*)forward_m.get(this);
			}
		}
		else if (m_tD->inherited_type == output_tD)
		{
			reflect::output_type* out = (reflect::output_type*)m.get(this);
			out->owner = this;
		}
	}
}

reflect::input_type* Vulkan_Module::get_input_by_name(std::string str)
{
	reflect::TypeDescriptor_Struct* tD = GetDynamicReflection();

	//tD->dump((void*)this, 0);

	reflect::TypeDescriptor* input_tD = reflect::TypeResolver<reflect::input_type>::get();
	reflect::TypeDescriptor* output_tD = reflect::TypeResolver<reflect::output_type>::get();
	
	for (reflect::Member& m : tD->members)
	{
		reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;

		if (strcmp(m.name, str.c_str()) == 0)
		{
			return (reflect::input_type*)m.get(this);
		}
		/*
		if (m_tD->inherited_type == input_tD)
		{
			reflect::input_type* in = (reflect::input_type*)m.get(this);
			in->owner = this;

			if (m.forward_output != 0xFF)
			{
				reflect::Member& forward_m = tD->members[m.forward_output];
				in->forward_output = (reflect::output_type*)forward_m.get(this);
			}
		}
		else if (m_tD->inherited_type == output_tD)
		{
			reflect::output_type* out = (reflect::output_type*)m.get(this);
			out->owner = this;
		}*/
	}
}

Vulkan_Module* get_module_by_uid(std::vector<Vulkan_Module*>* modules, u64 uid)
{
	for (Vulkan_Module* m : *modules)
	{
		if (m->m_uid == uid)
			return m;
	}
	return NULL;
}

void Vulkan_Module::createComputePipeline(const char* shader_path)
{
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType =
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &(descriptorSetLayout->getDescriptorSetLayout());

	if (vkCreatePipelineLayout(m_device->getDevice(), &pipelineLayoutInfo, nullptr,
		&pipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute pipeline layout!");
	}

	pipeline = new ComputePipeline(m_device, shader_path, pipelineLayout);
}

void Vulkan_Module::createComputePipeline(const char* shader_path, VkPushConstantRange pushconstant)
{
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType =
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &(descriptorSetLayout->getDescriptorSetLayout());

	pipelineLayoutInfo.pPushConstantRanges = &pushconstant;
	pipelineLayoutInfo.pushConstantRangeCount = 1;

	if (vkCreatePipelineLayout(m_device->getDevice(), &pipelineLayoutInfo, nullptr,
		&pipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute pipeline layout!");
	}

	pipeline = new ComputePipeline(m_device, shader_path, pipelineLayout);
}


vkMultiImageResource* Vulkan_App::create_multiImage(int n_layers, int width, int height, VkImageUsageFlags flags, reflect::output_type* output_binding)
{
	vkMultiImageResource* img = new vkMultiImageResource(output_binding);

	img->Images.resize(n_layers);

	for (int i = 0; i < n_layers; i++)
	{
		m_device->createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_TILING_OPTIMAL,
			flags,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, img->Images[i].Image,
			img->Images[i].ImageMemory);

		img->Images[i].ImageView = m_device->createImageView(img->Images[i].Image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
	}

	img->status = RESOURCE_VALID;

	resources.push_back(img);

	return img;
}

vkImageResource* Vulkan_App::create_image(int width, int height, VkImageUsageFlags flags, reflect::output_type* output_binding)
{
	vkImageResource* img = new vkImageResource(output_binding);

	m_device->createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_TILING_OPTIMAL,
		flags,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, img->Image,
		img->ImageMemory);

	img->status = RESOURCE_VALID;

	resources.push_back(img);

	return img;
}

vkBufferResource* Vulkan_App::create_buffer(VkDeviceSize bufferSize, VkBufferUsageFlags flags)
{
	vkBufferResource* buffer = new vkBufferResource();
	/* VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT*/

	m_device->createBuffer(bufferSize, flags,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer->Buffer,
		buffer->BufferMemory);

	buffer->status = RESOURCE_VALID;
	buffer->range = bufferSize;
	buffer->used = bufferSize;

	resources.push_back(buffer);

	return buffer;
}

void Vulkan_App::createCommandBuffers() {
	commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = m_device->getCommandPool();
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

	if (vkAllocateCommandBuffers(m_device->getDevice(), &allocInfo, commandBuffers.data()) !=
		VK_SUCCESS) {
		throw std::runtime_error("failed to allocate command buffers!");
	}
}

void Vulkan_App::createDescriptorPool() {

	std::vector<VkDescriptorPoolSize> poolSizes{};
	poolSizes.resize(4);

	poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[0].descriptorCount = 1;

	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[1].descriptorCount = 6;

	poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[2].descriptorCount = 2;

	poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[3].descriptorCount = 1;

	m_DescriptorPool = new MyDescriptorPool(m_device, 6, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, poolSizes);
}


void Vulkan_App::status()
{
	for (Vulkan_Module* vk : all_modules)
	{
		if (vk->my_status == VK_MODULE_NOT_RAN)
		{
			vk->log() << vk->GetDynamicReflection()->name << " did not run\n";

			reflect::TypeDescriptor_Struct* tD = vk->GetDynamicReflection();

			for (reflect::Member& m : tD->members)
			{
				reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;

				if (m_tD->inherited_type == &reflect::input_type::Reflection)
				{
					reflect::input_type* in = (reflect::input_type*)m.get(vk);
					if (!in->ready)
					{
						vk->log() << "  " << m.name << " is not ready\n";
					}
				}
			}
		}
	}
}

void Vulkan_App::pre_run_check()
{
	reflect::TypeDescriptor* input_tD = reflect::TypeResolver<reflect::input_type>::get();
	reflect::TypeDescriptor* output_tD = reflect::TypeResolver<reflect::output_type>::get();

	u32 total_mem = 0;
	for (Vulkan_Module* vk : all_modules)
	{
		reflect::TypeDescriptor_Struct* tD = vk->GetDynamicReflection();

		if(vk->is_submodule == false)
			vk->setDimensions(); //propagate dimensions to inputs/outputs and push constants

		for (reflect::Member& m : tD->members)
		{
			reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;
			if (m_tD->inherited_type == input_tD)
			{
				reflect::input_type* in = (reflect::input_type*)m.get(vk);
				if (in->status == reflect::INPUT_NOT_CONNECTED)
				{
					vk->log() << tD->name << " disabled\n";
					vk->enabled = false;
				}
			}
			else if (m_tD->inherited_type == output_tD)
			{
				reflect::output_type* out = (reflect::output_type*)m.get(vk);
				if (m.flags & REFLECT_VKMOD_MEMBER_FLAG_CREATE_MEMORY)
				{
					vk->memory_needed = out->dimensions.size();
					vk->log() << tD->alias << " memory: " << (vk->memory_needed * 4) / 1000 << "k \n";
					total_mem += (vk->memory_needed * 4);
				}
			}
		}
	}
	cout << "--------------\n";
	cout << "total memory use: " << total_mem / (1024 * 1024) << "Mb\n";
	cout << "GPU RAM size: " << m_device->getDeviceRAMSize() << "Mb \n";
	cout << "Usage: " << f32(total_mem / (1024 * 1024)) / f32(m_device->getDeviceRAMSize()) << "\n\n";

}

void Vulkan_App::run_workflow()
{
	

	//geo_module->run_and_push();

	bool still_running = true;
	while (still_running)
	{
		still_running = false;
		for (Vulkan_Module* vk : all_modules)
		{
			if (vk->enabled && vk->is_submodule == false && vk->my_status == VK_MODULE_NOT_RAN)
			{
				if(vk->signaled() == true)
					still_running = true;
			}
		}
	}

	
}

void vkImageSubresource::destroy(VkDevice device)
{
	vkDestroyImageView(device, ImageView, nullptr);
	vkDestroyImage(device, Image, nullptr);
	vkFreeMemory(device, ImageMemory, nullptr);
}

void vkImageResource::destroy(VkDevice device)
{
	vkDestroyImageView(device, ImageView, nullptr);
	vkDestroyImage(device, Image, nullptr);
	vkFreeMemory(device, ImageMemory, nullptr);
}

void vkImageResource::create_and_load_texture(MyDevice* device, video::IVideoDriver* driver, video::ITexture* tex)
{
	VkDeviceSize width = tex->getOriginalSize().Width;
	VkDeviceSize height = tex->getOriginalSize().Height;
	VkDeviceSize imgSize = width * height * 4;

	MyBufferObject stagingBuffer(device, imgSize, 1, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);

	irr::video::IImage* pImage = driver->createImage(tex, core::vector2di(0, 0), tex->getOriginalSize());
	pImage->flip(true, false);

	irr::u8* imgDataPtr = (irr::u8*)pImage->getData();

	stagingBuffer.writeToBuffer(imgDataPtr);

	//pImage->unlock();
	pImage->drop();

	device->createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Image,
		ImageMemory);

	device->transitionImageLayout(Image, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	ImageView = device->createImageView(Image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

	device->copyBufferToImage(stagingBuffer.getBuffer(), Image, static_cast<uint32_t>(width), static_cast<uint32_t>(height));

	device->transitionImageLayout(Image, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
}


void vkImageSubresource::create_and_load_texture(MyDevice* device, video::IVideoDriver* driver, video::ITexture* tex)
{
	VkDeviceSize width = tex->getOriginalSize().Width;
	VkDeviceSize height = tex->getOriginalSize().Height;
	VkDeviceSize imgSize = width * height * 4;

	MyBufferObject stagingBuffer(device, imgSize, 1, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);

	irr::video::IImage* pImage = driver->createImage(tex, core::vector2di(0, 0), tex->getOriginalSize());
	pImage->flip(true, false);

	irr::u8* imgDataPtr = (irr::u8*)pImage->getData();

	stagingBuffer.writeToBuffer(imgDataPtr);

	//pImage->unlock();
	pImage->drop();

	device->createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Image,
		ImageMemory);

	device->transitionImageLayout(Image, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	ImageView = device->createImageView(Image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

	device->copyBufferToImage(stagingBuffer.getBuffer(), Image, static_cast<uint32_t>(width), static_cast<uint32_t>(height));

	device->transitionImageLayout(Image, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
}

VkDescriptorSetLayoutBinding vkMultiImageResource::getDescriptorSetLayout(u32 binding_no)
{
	VkDescriptorSetLayoutBinding imageStorageBinding{};
	imageStorageBinding.binding = binding_no;
	imageStorageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	imageStorageBinding.descriptorCount = 1;
	imageStorageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	imageStorageBinding.pImmutableSamplers = nullptr; // Optional

	return imageStorageBinding;
}

VkDescriptorImageInfo vkMultiImageResource::getDescriptorBufferInfo(u32 image_no)
{
	VkDescriptorImageInfo imageStorageInfo{};
	imageStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	imageStorageInfo.imageView = Images[image_no].ImageView;

	return imageStorageInfo;
}

void vkMultiImageResource::destroy(VkDevice device)
{
	for (auto& img : Images)
		img.destroy(device);
}

VkDescriptorSetLayoutBinding vkBufferResource::getDescriptorSetLayout(u32 binding_no)
{
	VkDescriptorSetLayoutBinding Binding{};
	Binding.binding = binding_no;
	Binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	Binding.descriptorCount = 1;
	Binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	Binding.pImmutableSamplers = nullptr; // Optional

	return Binding;
}

VkDescriptorBufferInfo* vkBufferResource::getDescriptorBufferInfo()
{
	//VkDescriptorBufferInfo BufferInfo{};
	BufferInfo.buffer = Buffer;
	BufferInfo.offset = 0;
	BufferInfo.range = used;

	return &BufferInfo;
}

void vkBufferResource::destroy(VkDevice device)
{
	vkDestroyBuffer(device, Buffer, nullptr);
	vkFreeMemory(device, BufferMemory, nullptr);
}

VkDescriptorSetLayoutBinding vkUniformBufferResource::getDescriptorSetLayout(u32 binding_no)
{
	VkDescriptorSetLayoutBinding Binding{};
	Binding.binding = binding_no;
	Binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	Binding.descriptorCount = 1;
	Binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	Binding.pImmutableSamplers = nullptr; // Optional

	return Binding;
}

VkDescriptorBufferInfo vkUniformBufferResource::getDescriptorBufferInfo()
{
	VkDescriptorBufferInfo BufferInfo{};
	BufferInfo.buffer = Buffer;
	BufferInfo.offset = 0;
	BufferInfo.range = range;

	return BufferInfo;
}

void vkUniformBufferResource::destroy(VkDevice device)
{
	vkDestroyBuffer(device, Buffer, nullptr);
	vkFreeMemory(device, BufferMemory, nullptr);
}

void vkUniformBufferResource::writeToBuffer(VkDevice device, void* data, VkDeviceSize, VkDeviceSize offset)
{
	void* mapped_data = nullptr;

	vkMapMemory(device, BufferMemory, 0, range, 0, &mapped_data);

	memcpy(mapped_data, data, range);

	vkUnmapMemory(device, BufferMemory);
}

namespace reflect
{
	template <>
	void output<vkBufferResource>::make_buffer()
	{
		int n_indices = dimensions.B * dimensions.C * dimensions.H * dimensions.W;
		VkDeviceSize bufferSize = sizeof(float) * n_indices;

		X = vulkan->create_buffer(bufferSize,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	}

	template <>
	void parameter<vkBufferResource>::make_buffer()
	{
		int n_indices = dimensions.B * dimensions.C * dimensions.H * dimensions.W;
		VkDeviceSize bufferSize = sizeof(float) * n_indices;

		X = vulkan->create_buffer(bufferSize,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	}
}

vkMemoryResource::vkMemoryResource(/*reflect::output_type* out*/)
{
	//owner = out;
	//find_consumers(out);
}
/*
void vkMemoryResource::find_consumers(reflect::output_type* out)
{
	if (out == NULL)
		return;

	for (reflect::input_type* dest : out->dest_inputs)
	{
		if (dest->owner->enabled == false)
			continue;

		consumers.push_back(dest);

		if (dest->forward_output != NULL)
		{
			find_consumers(dest->forward_output);
		}
	}
}*/
/*
void vkMemoryResource::consume(reflect::input_type* in)
{
	vector<reflect::input_type*> new_consumers;
	for (reflect::input_type* it : consumers)
	{
		if (it != in)
			new_consumers.push_back(it);
	}

	if (new_consumers.size() > 0)
	{
		consumers = new_consumers;
	}
	else
	{
		consumers.clear();
		//reflect::TypeDescriptor_Struct* owner_tD = (reflect::TypeDescriptor_Struct*)owner->owner->GetDynamicReflection();
		//reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)GetDynamicReflection();
		//cout << "  " << owner_tD->name << "::" << m_tD->name << "  destroyed\n";
		status = RESOURCE_DESTROYED;
		destroy(vk_device);
	}
}*/

namespace reflect
{
	template <>
	TypeDescriptor* getPrimitiveDescriptor<VkMod_Reference>() {
		static TypeDescriptor_VkMod_Reference typeDesc;
		return &typeDesc;
	}
}


namespace reflect
{
	void TypeDescriptor_VkWorkflow_Struct::serialize(std::ofstream& f, const void* obj)
	{
		TypeDescriptor_Struct::serialize(f, obj);
	}

	void TypeDescriptor_VkWorkflow_Struct::deserialize(std::ifstream& f, void* obj)
	{
		TypeDescriptor_Struct::deserialize(f, obj);

		Vulkan_Workflow* workflow = (Vulkan_Workflow*)obj;

		//Set input/output pointers
		for (VkMod_Reference& m : workflow->Modules)
		{
			Vulkan_Module* mod = m.X;

			reflect::TypeDescriptor_Struct* tD = mod->GetDynamicReflection();

			//tD->dump((void*)mod,0);

			for (reflect::Member& m : tD->members)
			{
				reflect::TypeDescriptor_Struct* m_tD = (reflect::TypeDescriptor_Struct*)m.type;

				if (m_tD->inherited_type == &reflect::output_type::Reflection)
				{
					reflect::output_type* out = (reflect::output_type*)m.get(mod);
					
					for (int i=0; i<out->output_uids.size() ; i++)
					{
						Vulkan_Module* mod2 = workflow->get_module_by_uid(out->output_uids[i]);
						if (mod2)
						{
							//reflect::input_type* in = (reflect::input_type *)mod2->get_inout_by_name(out->output_member[i]);
							//reflect::inout_gui_type* in_ = mod2->get_input_by_name(out->output_member[i]);
							reflect::input_type* in = mod2->get_input_by_name(out->output_member[i]);
							if (in)
							{
								out->dest_inputs.push_back(in);
								in->src_output = out;
							}
							else
							{
								cout << "could not find " << out->output_member[i] << "\n";
							}
						}
						else
						{
							cout << "could not find " << out->output_uids[i] << "\n";
						}
					}
				}
			}
		}
	}
}


//============================================================================
//  Add type to factory at initialization


namespace reflect
{
	TypeDescriptor_VkMod_Struct::TypeDescriptor_VkMod_Struct(void (*init)(TypeDescriptor_Struct*)) : reflect::TypeDescriptor_Struct(init)
	{
		Vulkan_Module::factory.addType(this);
	}
}

REFLECT_VKMOD_BEGIN(Vulkan_Module)
	REFLECT_STRUCT_MEMBER(pos);
	REFLECT_STRUCT_MEMBER(m_uid);
REFLECT_STRUCT3_END()

REFLECT_VK_WORKFLOW_BEGIN(Vulkan_Workflow)
	REFLECT_STRUCT_MEMBER(Modules)
REFLECT_STRUCT_END()