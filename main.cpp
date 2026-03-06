#include <iostream>
#include <vector>
#include <chrono>

#include <Windows.h>
#include <WinUser.h>
#include <WinDef.h>

#include <irrlicht.h>

#include "tkt_set.h"
#include "edit_env.h"
#include "fonts.h"
#include "CameraPanel.h"


struct aligned_float {
	alignas(16) float x;
};

struct Convolution_Layer
{
	struct info_struct
	{
		int k = 4;
		int stride = 2;
		int padding = 1;
		int channels_out = 64;
		int channels_in = 64;
		float bias = 0;
	};

	std::vector<aligned_float> weights;
	std::vector<aligned_float> activations;
	std::vector<aligned_float> outputs;

	Convolution_Layer()
	{
		weights.resize(info.k * info.k * info.channels_out * info.channels_in);
		activations.resize(info.k * info.k * info.channels_in);
		outputs.resize(info.k * info.k * info.channels_out);
	}

	info_struct info;
};

using namespace irr;
using namespace std;

extern IrrlichtDevice* device = 0;
extern GUI_layout* gui_layout = NULL;
extern float g_time = 0;

void initialize_camera_quad(scene::ISceneManager* smgr, GUI_layout* gui_layout, RenderList* renderList)
{
	CameraQuad* cameraQuad = gui_layout->getCameraQuad();

	cameraQuad->setRenderList(renderList);
	cameraQuad->initialize(smgr);
	//cameraQuad->setGridSnap(8);
	//cameraQuad->setRotateSnap(7.5);
}

void initialize_tools(gui::IGUIEnvironment* gui, multi_tool_panel* tool_panel)
{

}


int main()
{
	video::E_DRIVER_TYPE driverType = video::EDT_OPENGL;

	if (driverType == video::EDT_COUNT)
		return 1;

	MyEventReceiver receiver;

	device = createDevice(driverType, core::dimension2d<u32>(1500, 850), 16, false, false, false, &receiver);

	if (device == 0)
		return 1; // could not create selected driver.

	video::IVideoDriver* driver = device->getVideoDriver();
	scene::ISceneManager* smgr = device->getSceneManager();
	gui::IGUIEnvironment* gui = device->getGUIEnvironment();

	//==============================================================
	// Adjust for DPI
	//
	
	bool dpiAware = true;
	if (dpiAware)
	{
		DPI_AWARENESS_CONTEXT dpi_context = DPI_AWARENESS_CONTEXT_SYSTEM_AWARE;
		SetProcessDpiAwarenessContext(dpi_context);

		video::SExposedVideoData videodata = driver->getExposedVideoData();
		RECT window_rect;
		GetClientRect((HWND)videodata.OpenGLWin32.HWnd, &window_rect);
		u32 dpi = GetDpiForWindow((HWND)videodata.OpenGLWin32.HWnd);

		cout << "dpi: " << dpi << "\n";
	}

	//==============================================================
	//Choose a font and render it
	//

	AGG_TT_Font_Renderer font_render;
	AGG_TT_Font* agg_font = font_render.Render_Font(gui, "Calibri", 22, 1.0, -1, 2, core::dimension2du{ 256,256 });

	//==============================================================
	//Initialize GUI elements
	//
	gui_layout = new GUI_layout(driver, gui);
	gui_layout->initialize(core::rect<s32>(core::position2d<s32>(0, 0), core::dimension2d<u32>(1200, 680)), agg_font);
	//gui_layout->initialize(core::rect<s32>(core::position2d<s32>(0, 0), core::dimension2d<u32>(1200, 680)), NULL);

	multi_tool_panel* tool_panel = gui_layout->getToolPanel();
	initialize_tools(gui, tool_panel);

	RenderList* renderList = new RenderList(driver);
	initialize_camera_quad(smgr, gui_layout, renderList);
	gui_layout->getCameraQuad()->SetFullscreen(true);

	//==============================================================
	//
	//

	device->getCursorControl()->setVisible(true);

	int lastFPS = -1;

	core::rect<s32> windowsize = driver->getViewPort();
	core::rect<s32> windowsize_0 = windowsize;

	gui_layout->resize(windowsize);

	while (device->run())
		if (device->isWindowActive())
		{
			const u32 now = device->getTimer()->getTime();
			g_time = double(now) / 1000;

			core::rect<s32> windowsize = driver->getViewPort();
			if (windowsize != windowsize_0)
			{
				windowsize_0 = windowsize;
				gui_layout->resize(windowsize);
			}


			driver->beginScene(true, true, video::SColor(255, 32, 32, 32));

			renderList->renderAll();

			gui->drawAll();


			driver->endScene();
			/*
			int fps = driver->getFPS();

			if (lastFPS != fps)
			{
				core::stringw str = L"Das Irrlicht Engine  [";
				str += driver->getName();
				str += "] FPS:";
				str += fps;

				device->setWindowCaption(str.c_str());
				lastFPS = fps;
			}*/
		}
	return 0;
}