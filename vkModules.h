#pragma once

#ifndef _LIGHTMAP_MODULES_H_
#define _LIGHTMAP_MODULES_H_

#include <irrlicht.h>
#include <vector>
//#include "BufferManager.h"
#include <iterator>
#include "vkDevice.h"
//#include "vkModel.h"
//#include "vk_BVH.h"
#include <vulkan/vulkan.h>
#include "vkDescriptors.h"
#include "vkBufferObject.h"
#include "vkComputePipeline.h"
#include "reflect_custom_types.h"
#include <fstream>
#include <random>
#include "tkt_set.h"
#include "tkt_reflection_factory.h"
#include "global.h"



class MyDescriptorPool;
class Vulkan_Module;

class Module_GUI_Element;
class System_GUI_Element;
class Light_System_Base;
class CachedText;
class AGG_TT_Font;

using namespace std;




namespace reflect 
{

	struct input_type;
	struct output_type;

	struct vector2i
	{
		int X;
		int Y;

		void operator=(core::vector2d<int> v) {
			X = v.X;
			Y = v.Y;
		}

		operator core::vector2di() const {
			core::vector2di ret;
			ret.X = X;
			ret.Y = Y;
			return ret;
		}

		REFLECT()
	};

	struct TypeDescriptor_VkWorkflow_Struct : TypeDescriptor_Struct
	{
		bool placeable;
		//Reflected_SceneNode* (*create_func)(USceneNode* parent, geometry_scene*, irr::scene::ISceneManager* smgr, int id, const irr::core::vector3df& pos);
		TypeDescriptor_VkWorkflow_Struct(void (*init)(TypeDescriptor_Struct*))
			: TypeDescriptor_Struct(init)
		{}

		virtual void serialize(std::ofstream& f, const void* obj) override;
		// {
		//		TypeDescriptor_Struct::serialize(f, obj);
		//}

		virtual void deserialize(std::ifstream& f, void* obj) override;
		//{
		//	TypeDescriptor_Struct::deserialize(f, obj);
		//}


	};

	struct TypeDescriptor_VkMod_Struct : TypeDescriptor_Struct
	{
		bool placeable;
		//Reflected_SceneNode* (*create_func)(USceneNode* parent, geometry_scene*, irr::scene::ISceneManager* smgr, int id, const irr::core::vector3df& pos);
		TypeDescriptor_VkMod_Struct(void (*init)(TypeDescriptor_Struct*));
		Vulkan_Module* (*getNew)();
		int flags;

		gui::IGUIElement* get_gui_element(System_GUI_Element* system, Light_System_Base* base, gui::IGUIEnvironment* environment, s32 id, core::rect<s32> rectangle, void* obj);
		int addComponents(Light_System_Base*, gui::IGUIEnvironment*, gui::IGUIElement*, CachedText&, vector2di, void* obj);
		int calculate_height();
		input_type* get_input0(void* obj);
		output_type* get_output0(void* obj);
	};

	using TypeDescriptor_VkMod_Reference = TypeDescriptor_Typeless_Reference<Vulkan_Module>;

#define REFLECT_VKMOD() \
        virtual reflect::TypeDescriptor_Struct* GetDynamicReflection(); \
        friend struct reflect::DefaultResolver; \
        static reflect::TypeDescriptor_VkMod_Struct Reflection; \
        static void initReflection(reflect::TypeDescriptor_Struct*); \
		static Vulkan_Module* getNew();

#define REFLECT_VKMOD_BEGIN(type) \
        reflect::TypeDescriptor_VkMod_Struct type::Reflection{type::initReflection}; \
		Vulkan_Module* type::getNew(){ \
            return new type();\
            }\
        reflect::TypeDescriptor_Struct* type::GetDynamicReflection() {\
            return &type::Reflection;\
            }\
        void type::initReflection(reflect::TypeDescriptor_Struct* typeDesc) { \
			bool backward_pass = false; \
            using T = type; \
            typeDesc->name = #type; \
            typeDesc->size = sizeof(T); \
            typeDesc->inherited_type = NULL; \
            typeDesc->name_func = NULL; \
            typeDesc->alias = typeDesc->name;\
			((reflect::TypeDescriptor_VkMod_Struct*)typeDesc)->getNew = type::getNew;

#define REFLECT_VKMOD_FLAG_TRANSITION_MODULE	2

#define REFLECT_VKMOD_MEMBER_FLAG_CREATE_MEMORY	2
#define REFLECT_VKMOD_MEMBER_FLAG_BACKWARD_PASS	4

#define REFLECT_VKMOD_MEMBER(name) \
        typeDesc->members.push_back(reflect::Member{#name, offsetof(T, name), reflect::TypeResolver<decltype(T::name)>::get(),0xFF}); \
		if(backward_pass) typeDesc->members[typeDesc->members.size()-1].flags |= REFLECT_VKMOD_MEMBER_FLAG_BACKWARD_PASS;

#define REFLECT_VKMOD_FLAG(f) \
			((reflect::TypeDescriptor_VkMod_Struct*)typeDesc)->flags |= f;

#define REFLECT_VKMOD_MEMBER_CREATE_MEMORY() \
			typeDesc->members[typeDesc->members.size()-1].flags |= REFLECT_VKMOD_MEMBER_FLAG_CREATE_MEMORY;

#define INHERIT_FROM(name) \
            typeDesc->inherited_type = (reflect::TypeDescriptor_Struct*)reflect::TypeResolver<name>::get();

#define ALIAS(name) \
            typeDesc->alias = name;

#define REFLECT3() \
        virtual reflect::TypeDescriptor_Struct* GetDynamicReflection(); \
        friend struct reflect::DefaultResolver; \
        static reflect::TypeDescriptor_Struct Reflection; \
        static void initReflection(reflect::TypeDescriptor_Struct*); \

#define REFLECT_STRUCT3_BEGIN(type) \
        reflect::TypeDescriptor_Struct type::Reflection{type::initReflection}; \
        reflect::TypeDescriptor_Struct* type::GetDynamicReflection() {\
            return &type::Reflection;\
            }\
        void type::initReflection(reflect::TypeDescriptor_Struct* typeDesc) { \
            using T = type; \
            typeDesc->name = #type; \
            typeDesc->size = sizeof(T); \
            typeDesc->inherited_type = NULL; \
            typeDesc->name_func = NULL; \
            typeDesc->alias = typeDesc->name;\

#define REFLECT_VKMOD_MEMBER_OUTPUT_IN_PLACE(name0,name1) \
		int a = offsetof(T,name0);int b = offsetof(T,name1);\
        for(int i=0;i<typeDesc->members.size();i++) {\
			if(strcmp(typeDesc->members[i].name,#name0)==0) {\
				for(int j=0;j<typeDesc->members.size();j++) {\
					if(strcmp(typeDesc->members[j].name,#name1)==0) { \
						typeDesc->members[i].in_place_output = j; \
						std::cout << #name0 <<" FWD DST = "<<j<<", "<<typeDesc->members[j].name<<"\n";\
					}\
				}\
			}\
		}

#define REFLECT_VKMOD_FORWARD_PASS() backward_pass = false;

#define REFLECT_VKMOD_BACKWARD_PASS() backward_pass = true;

#define REFLECT_STRUCT3_END() \
		}

#define REFLECT_VKMOD_END() \
		}

#define REFLECT_VK_WORKFLOW() \
        virtual reflect::TypeDescriptor_Struct* GetDynamicReflection(); \
        friend struct reflect::DefaultResolver; \
        static reflect::TypeDescriptor_VkWorkflow_Struct Reflection; \
        static void initReflection(reflect::TypeDescriptor_Struct*); 

#define REFLECT_VK_WORKFLOW_BEGIN(type) \
        reflect::TypeDescriptor_VkWorkflow_Struct type::Reflection{type::initReflection}; \
        reflect::TypeDescriptor_Struct* type::GetDynamicReflection() {\
            return &type::Reflection;\
            }\
        void type::initReflection(reflect::TypeDescriptor_Struct* typeDesc) { \
            using T = type; \
            typeDesc->name = #type; \
            typeDesc->size = sizeof(T); \
            typeDesc->inherited_type = NULL; \
            typeDesc->name_func = NULL; \
            typeDesc->alias = typeDesc->name;

	//========================================================================================
	//  FACTORY CLASS
	//
	//

	using Vulkan_Reflection_Factory = ::tkt_reflection_factory<TypeDescriptor_VkMod_Struct>;
}

enum {
	RESOURCE_UNK,
	RESOURCE_VALID,
	RESOURCE_DESTROYED
};

struct vkMemoryResource
{
	vkMemoryResource(/*reflect::output_type**/);
	//void find_consumers(reflect::output_type*);
	//void consume(reflect::input_type*);

	//u32 reference_count = 0;
	//std::vector<reflect::input_type*> consumers;
	virtual void destroy(VkDevice) = 0;
	//reflect::output_type* owner = NULL;
	int status = RESOURCE_UNK;

	virtual bool equal_dimensions(vkMemoryResource* other) {
		return true;
	}

	virtual reflect::TypeDescriptor_Struct* GetDynamicReflection() = 0;
};

struct vkRaytraceResource
{
	
};

struct vkUniformBufferResource //: public vkMemoryResource
{

	VkBuffer Buffer;
	VkDeviceMemory BufferMemory;
	u32 range = 0;

	VkDescriptorSetLayoutBinding getDescriptorSetLayout(u32);
	VkDescriptorBufferInfo getDescriptorBufferInfo();

	void destroy(VkDevice);
	void writeToBuffer(VkDevice device, void* data, VkDeviceSize = VK_WHOLE_SIZE, VkDeviceSize offset = 0);

};

struct vkBufferResource : public vkMemoryResource
{
	VkBuffer Buffer;
	VkDeviceMemory BufferMemory;
	u32 range = 0;
	u32 used = 0;
	VkDescriptorBufferInfo BufferInfo;

	vkBufferResource() : vkMemoryResource() {}
	VkDescriptorSetLayoutBinding getDescriptorSetLayout(u32);
	VkDescriptorBufferInfo* getDescriptorBufferInfo();

	void destroy(VkDevice);

	REFLECT3()
};

struct vkImageResource : public vkMemoryResource
{
	VkImage Image;
	VkDeviceMemory ImageMemory;
	VkImageView ImageView;

	vkImageResource(reflect::output_type* out) : vkMemoryResource() {}
	void destroy(VkDevice);
	void create_and_load_texture(MyDevice* device, video::IVideoDriver* driver, video::ITexture* tex);

	REFLECT3()
};

struct vkImageSubresource
{
	VkImage Image;
	VkDeviceMemory ImageMemory;
	VkImageView ImageView;

	void destroy(VkDevice);
	void create_and_load_texture(MyDevice* device, video::IVideoDriver* driver, video::ITexture* tex);

	REFLECT3()
};

struct vkMultiImageResource : public vkMemoryResource
{
	std::vector<vkImageSubresource> Images;

	vkMultiImageResource(reflect::output_type* out) : vkMemoryResource() {}

	VkDescriptorSetLayoutBinding getDescriptorSetLayout(u32);
	VkDescriptorImageInfo getDescriptorBufferInfo(u32);

	void destroy(VkDevice);

	REFLECT3()
};

class Vulkan_Module;
class Vulkan_App;
std::vector<Vulkan_Module*>* get_all_vk_modules();

Vulkan_Module* get_module_by_uid(std::vector<Vulkan_Module*>*, u64 uid);

class Hookup_GUI_Element;

namespace reflect
{

	template <typename TY> struct input;
	template <typename TY> struct output;

	template <typename TY>
	void connect(input<TY>* in, output<TY>* out);

	struct inout_gui_type
	{
		Vulkan_App* vulkan = NULL;
		Hookup_GUI_Element* m_gui_element = NULL;
	};

	struct output_type;

	enum {
		INPUT_NOT_CONNECTED,
		INPUT_WAITING,
		INPUT_FINISHED
	};

	struct input_type : public inout_gui_type
	{
		//u64 my_uid = 0;
		
		u64 input_uid = 0;
		std::string input_member;

		//not reflected
		//

		Vulkan_Module* owner = NULL;
		output_type* src_output = NULL;		
		output_type* in_place_output = NULL;	
		bool ready = false;				
		int status = INPUT_NOT_CONNECTED;

		REFLECT3()
	};

	struct output_type : public inout_gui_type
	{
		//u64 my_uid = 0;
		
		std::vector<u64> output_uids;
		std::vector<std::string> output_member;

		//not reflected
		//
		virtual void make_buffer() = 0;
		Vulkan_Module* owner = NULL;
		std::vector<input_type*> dest_inputs;	
		TensorDimension dimensions; //not reflected
		bool ready = false;			

		virtual void signal() = 0;
		virtual void push() = 0;

		REFLECT3()
	};

	template <typename TY>
	struct input : public input_type
	{
		struct attributes
		{
		};

		input() {}

		u64 old_uid;	//not reflected
		TY* X = NULL;			//not reflected

		//virtual bool load() override;

		virtual reflect::TypeDescriptor_Struct* GetDynamicReflection() { return &input<TY>::Reflection; }
		REFLECT_CUSTOM_STRUCT()
	};

	template <typename TY>
	struct output : public output_type
	{
		struct attributes
		{
		};

		output() {}

		virtual void make_buffer();

		bool equal_dimensions(input<TY>* input_obj) {
			return X->equal_dimensions(input_obj->X);
		}

		std::vector<u64> old_uids;	//not reflected
		TY* X = NULL;						

		virtual void signal() override;
		virtual void push() override;

		virtual reflect::TypeDescriptor_Struct* GetDynamicReflection() { return &output<TY>::Reflection; }
		REFLECT_CUSTOM_STRUCT()
	};

	struct parameter_type : public inout_gui_type
	{
		virtual void make_buffer() = 0;
		Vulkan_Module* owner = NULL;
		TensorDimension dimensions; //not reflected
		bool ready = false;

		REFLECT3()
	};

	template <typename TY>
	struct parameter : public parameter_type
	{
		parameter(std::vector<f32>&) : map{ map } {}
		struct attributes
		{
		};

		TY* X = NULL;
		std::vector<f32>& map = NULL;

		virtual void make_buffer();

		REFLECT_CUSTOM_STRUCT()
	};

	//void connect(output<vkImageArrayResource>* in, input<vkMultiImageResource>* out);
	//void connect(output<vkMultiImageResource>* in, input<vkImageArrayResource>* out);

	template <typename TY>
	void connect( output<TY>* out, input<TY>* in)
	{
		//in->input_uid = out->my_uid;
		in->input_uid = out->owner->m_uid;
		//out->output_uids.push_back(in->my_uid);
		out->output_uids.push_back(in->owner->m_uid);

		Vulkan_Module* out_module = out->owner;
		Vulkan_Module* in_module = in->owner;

		if (!out_module || !in_module)
			return;

		//if (!out->equal_dimensions(in))
		//{
		//	cout << "Connection Failed---\n";
		//	return;
		//}

		reflect::TypeDescriptor_Struct* out_tD = out_module->GetDynamicReflection();
		for (int i=0;i< out_tD->members.size();i++)
		{
			reflect::Member& m = out_tD->members[i];

			if ((char*)out_module + m.offset == (char*)out)
			{
				in->input_member = m.name;
				in->src_output = (reflect::output_type*)m.get(out_module);
				in->status = reflect::INPUT_WAITING;
			}
		}

		reflect::TypeDescriptor_Struct* in_tD = in_module->GetDynamicReflection();
		for (int i = 0; i < in_tD->members.size(); i++)
		{
			reflect::Member& m = in_tD->members[i];

			if ((char*)in_module + m.offset == (char*)in)
			{
				out->output_member.push_back(m.name);
				out->dest_inputs.push_back((reflect::input_type*)m.get(in_module));
			}
		}

	}

	template <typename TY>
	void disconnect(input<TY>* in, output<TY>* out)
	{

	}
	/*
	template <typename TY>
	void output<TY>::consume(input_type* in)
	{
		X->consume(in);
	}

	*/

	template <typename TY>
	void output<TY>::push()
	{
		std::vector<Vulkan_Module*>* all_vulkan_modules = get_all_vk_modules();

		for (int i = 0; i < this->output_uids.size(); i++)
		{
			Vulkan_Module* module = get_module_by_uid(all_vulkan_modules, this->output_uids[i]);

			if (!module)
				return;

			TypeDescriptor_Struct* in_td = module->GetDynamicReflection();

			if (!in_td)
				return;

			input<TY>* in = (input<TY>*)this->dest_inputs[i];
			in->X = this->X;
			in->ready = true;
		}
	}

	template <typename TY>
	void output<TY>::signal()
	{
		std::vector<Vulkan_Module*>* all_vulkan_modules = get_all_vk_modules();

		for (int i = 0; i < this->output_uids.size(); i++)
		{
			Vulkan_Module* module = get_module_by_uid(all_vulkan_modules, this->output_uids[i]);

			if (!module)
				return;

			TypeDescriptor_Struct* in_td = module->GetDynamicReflection();

			if (!in_td)
				return;

			input<TY>* in = (input<TY>*)this->dest_inputs[i];
			in->X = this->X;
			in->ready = true;

			//module->signaled();
		}
	}
}


class Vulkan_App;

enum
{
	VK_MODULE_NOT_RAN = 0,
	VK_MODULE_RAN
};

struct VkMod_Reference
{
	Vulkan_Module* X;
};

struct ModuleLogProxy
{
	bool colored;
	~ModuleLogProxy() { if (colored) std::cout << "\033[0m"; }

	template<typename T>
	ModuleLogProxy& operator<<(const T& val) { std::cout << val; return *this; }

	// support std::endl and other manipulators
	ModuleLogProxy& operator<<(std::ostream& (*manip)(std::ostream&)) { manip(std::cout); return *this; }
};

class Vulkan_Module
{
public:

	Vulkan_Module();

	ModuleLogProxy log() const
	{
		for (u32 i = 0; i < depth; i++) std::cout << "   ";
		bool colored = depth > 0;
		if (colored) std::cout << "\033[90m";
		return ModuleLogProxy{ colored };
	}

	virtual void initialize(Vulkan_App* vulkan);
	virtual void setDimensions() {};

	void createComputePipeline(const char* shader_path);
	void createComputePipeline(const char* shader_path, VkPushConstantRange);

	virtual void build_workflow(std::vector<Vulkan_Module*>&) {}
	virtual void run() {}
	virtual void forward() {}
	virtual void backward() {}
	//bool load_resources();
	bool ready_forward();
	bool ready_backward();
	bool signaled();
	void run_and_push();
	void set_ptrs();

	reflect::input_type* get_input_by_name(std::string);

	Vulkan_App* vulkan = NULL;
	MyDevice* m_device = NULL;
	MyDescriptorPool* m_DescriptorPool = NULL;

	VkPipelineLayout pipelineLayout;
	std::vector<VkDescriptorSet> descriptorSets;
	MyDescriptorSetLayout* descriptorSetLayout = NULL;
	ComputePipeline* pipeline = NULL;

	u64 m_uid;
	u32 memory_needed = 0;
	int my_status = VK_MODULE_NOT_RAN;
	bool enabled = true;
	bool is_submodule = false;
	u32 depth = 0;

	static reflect::Vulkan_Reflection_Factory factory;

	reflect::vector2i pos{ 0,0 };

	REFLECT_VKMOD()
};

class Geometry_Module;

class Vulkan_App
{
public:

	Vulkan_App(video::IVideoDriver* driver);

	void initVulkan();
	
	void status();
	void pre_run_check();
	void run_workflow();
	void cleanup();

	void createDescriptorPool();
	void createCommandBuffers();

	vkMultiImageResource* create_multiImage(int n_layers, int width, int height, VkImageUsageFlags flags, reflect::output_type*);
	vkImageResource* create_image(int width, int height, VkImageUsageFlags flags, reflect::output_type*);
	vkBufferResource* create_buffer(VkDeviceSize bufferSize, VkBufferUsageFlags flags);

	template<typename T>
	Vulkan_Module* create_module();


	std::vector<VkCommandBuffer> commandBuffers;
	MyDescriptorPool* m_DescriptorPool = NULL;
	MyDevice* m_device = NULL;
	std::vector<Vulkan_Module*> all_modules;
	std::vector<vkMemoryResource*> resources;

	video::IVideoDriver* driver;
};

template<typename T>
Vulkan_Module* Vulkan_App::create_module()
{
	reflect::TypeDescriptor_Struct* descriptor = &T::Reflection;

	void* mem = malloc(descriptor->size);
	T* module = (T*)mem;

	new(module) T();
	module->initialize(this);

	return module;
}

class Vulkan_Workflow : public tkt_set<Vulkan_Module>
{
public:
	~Vulkan_Workflow();

	std::vector<VkMod_Reference> Modules;

	void make_default_workflow();
	void initialize_and_run(Vulkan_App* vulkan);


	void save();
	void load();

	Vulkan_Module* get_module_by_uid(u64 uid)
	{
		for (VkMod_Reference& m : Modules)
		{
			if (m.X->m_uid == uid)
				return m.X;
		}
		return NULL;
	}

	Vulkan_Module* backward_pass_head = NULL;

	REFLECT_VK_WORKFLOW()
};


#endif