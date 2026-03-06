
#include <irrlicht.h>
#include <iostream>

#include "edit_env.h"
#include "CameraPanel.h"
//#include "CGUIWindow.h"
//#include "csg_classes.h"
//#include "utils.h"
#include "edit_classes.h"
//#include "create_primitives.h"
//#include "texture_picker.h"
#include "GUI_tools.h"
//#include "geo_scene_settings.h"
//#include "NodeClassesTool.h"
//#include "NodeInstancesTool.h"
//#include "node_properties.h"
//#include "file_open.h"
//#include "lightmaps_tool.h"
//#include "uv_tool.h"
//#include "vtoolbar.h"
#include "ex_gui_elements.h"
//#include "geometry_scene.h"
//#include "LightMaps.h"
//#include "material_groups.h"
//#include "RenderTargetsTool.h"
//#include "SceneTool.h"
//#include "LightSystemTool.h"

#include <sstream>

extern IrrlichtDevice* device;
using namespace irr;
using namespace core;
using namespace gui;
using namespace std;

bool MyEventReceiver::OnEvent(const SEvent& event)
{
    

    if(event.EventType == EET_GUI_EVENT)
    {
        if (!event.GUIEvent.Caller)
            return false;

        s32 id = event.GUIEvent.Caller->getID();
        gui::IGUIEnvironment* env = device->getGUIEnvironment();

        switch(event.GUIEvent.EventType)
        {
            case EGET_MENU_ITEM_SELECTED:
            {
                OnMenuItemSelected((gui::IGUIContextMenu*)event.GUIEvent.Caller);
            }
            break;
            case EGET_BUTTON_CLICKED:
            {

                break;
            }
           
        }

    }

    // Remember the mouse state
    if (event.EventType == irr::EET_MOUSE_INPUT_EVENT)
    {
        switch(event.MouseInput.Event)
        {
        case EMIE_LMOUSE_PRESSED_DOWN:
            MouseState.LeftButtonDown = true;
            break;

        case EMIE_LMOUSE_LEFT_UP:
            MouseState.LeftButtonDown = false;
            break;

        case EMIE_MOUSE_MOVED:
            MouseState.Position.X = event.MouseInput.X;
            MouseState.Position.Y = event.MouseInput.Y;
            break;

        case EMIE_MOUSE_WHEEL:
            if(event.MouseInput.Wheel > 0)
                MouseState.WheelPos++;
            else
                MouseState.WheelPos--;
            break;

        case EMIE_RMOUSE_LEFT_UP:
            MouseState.RightButtonDown = false;
            break;

        case EMIE_RMOUSE_PRESSED_DOWN:
            MouseState.RightButtonDown = true;
            break;

        default:
            // We won't use the wheel
            break;
        }
    }
    else if(event.EventType ==   irr::EET_KEY_INPUT_EVENT)
    {
            KeyIsDown[event.KeyInput.Key] = event.KeyInput.PressedDown;
    }
    else if(event.EventType == irr::EET_USER_EVENT)
    {
        for(auto it = receivers.begin(); it != receivers.end(); ++it)
            (*it)->OnEvent(event);

        for (auto it = safe_remove_receivers.begin(); it != safe_remove_receivers.end(); ++it)
            receivers.remove(*it);

        safe_remove_receivers.clear();
    }

    return false;
}


void OnMenuItemSelected(IGUIContextMenu* menu)
{
    s32 id = menu->getItemCommandId(menu->getSelectedItem());
    gui::IGUIEnvironment* env = device->getGUIEnvironment();
    gui::IGUIElement* root = env->getRootGUIElement();
    gui::IGUIElement* quad = (CameraQuad*)root->getElementFromId(GUI_ID_CAMERA_QUAD, true);

    //do stuff
}

bool MyEventReceiver::IsKeyDown(EKEY_CODE keyCode) const
{
    return KeyIsDown[keyCode];
}

const MyEventReceiver::SMouseState& MyEventReceiver::GetMouseState(void) const
{
    return MouseState;
}

MyEventReceiver::MyEventReceiver()
{
    for (u32 i = 0; i<KEY_KEY_CODES_COUNT; ++i)
        KeyIsDown[i] = false;
}

void MyEventReceiver::resizeView(core::dimension2du newsize)
{
    for (auto it = resize_receivers.begin(); it != resize_receivers.end(); ++it)
        (*it)->resizeView(newsize);

    view_size = newsize;
}
/*
bool hasModalDialogue()
{
    if(!device)
        return false;
    gui::IGUIEnvironment* env = device->getGUIEnvironment();
    gui::IGUIElement* focused = env->getFocus();
    while(focused)
    {
        if(focused->isVisible() && focused->hasType(gui::EGUIET_MODAL_SCREEN))
           return true;
        focused = focused->getParent();
    }
    return false;
}*/

//==================================================================================================
//
//

bool GetAnyPlaneClickVector(dimension2d<u32> screenSize, scene::ICameraSceneNode * camera, core::plane3df plane, int clickx, int clicky, vector3df &hit_vec)
{
    const scene::SViewFrustum* frustum = camera->getViewFrustum();
    const core::vector3df cameraPosition = camera->getAbsolutePosition();

    vector3df vNearLeftDown = frustum->getNearLeftDown();
    vector3df vNearRightDown = frustum->getNearRightDown();
    vector3df vNearLeftUp = frustum->getNearLeftUp();
    vector3df vNearRightUp = frustum->getNearRightUp();

    f32 t_X = (f32)clickx / screenSize.Width;
    f32 t_Y = 1.0 - (f32)clicky / screenSize.Height;

    //screen space: Y is horizontal axis
    vector3df X_vec = (vNearRightDown - vNearLeftDown) * t_X;
    vector3df Y_vec = (vNearLeftUp - vNearLeftDown) * t_Y;
    vector3df target = vNearLeftDown + Y_vec + X_vec;

    vector3df ray = target-cameraPosition;
    ray.normalize();

    //World Space
    core::vector3df hitvec;

    if(plane.getIntersectionWithLine(cameraPosition,ray,hitvec))
        {
         hit_vec = hitvec;
         return true;
        }

    return false;
}

GUI_layout::GUI_layout(video::IVideoDriver* driv, gui::IGUIEnvironment* gui)
{
    driver = driv;
    env = gui;
}

void GUI_layout::initialize(core::rect<s32> win_rect, gui::IGUIFont* font)
{
    core::rect<s32> main_panel_rect = win_rect;

    main_panel_rect.LowerRightCorner.X -= 300;
    main_panel = new gui::IGUIElement(gui::EGUIET_ELEMENT, env, env->getRootGUIElement(), -1, main_panel_rect);

    MySkin* skin = new MySkin(EGST_WINDOWS_CLASSIC, driver);
    if (font)
    {
        gui::IGUIFont* builtinfont = env->getBuiltInFont();
        gui::IGUIFontBitmap* bitfont = 0;
        if (builtinfont && builtinfont->getType() == EGFT_BITMAP)
            bitfont = (IGUIFontBitmap*)builtinfont;


        IGUISpriteBank* bank = 0;
        if (bitfont)
            bank = bitfont->getSpriteBank();

        skin->setSpriteBank(bank);

        skin->setSize(EGDS_CHECK_BOX_WIDTH, font->getDimension(L"A").Height);
        skin->setSize(EGDS_WINDOW_BUTTON_WIDTH, font->getDimension(L"A").Height);
        skin->setFont(font);
        skin->setFont(font, EGDF_MENU);
    }
    else
    {
        gui::IGUIFont* builtinfont = env->getBuiltInFont();
        gui::IGUIFontBitmap* bitfont = 0;
        if (builtinfont && builtinfont->getType() == EGFT_BITMAP)
            bitfont = (IGUIFontBitmap*)builtinfont;

        IGUISpriteBank* bank = 0;
        skin->setFont(builtinfont);

        if (bitfont)
            bank = bitfont->getSpriteBank();

        gui::IGUIFont* def_font = env->getFont("fonthaettenschweiler.bmp");
        if (def_font)
            skin->setFont(def_font);
    }

    env->setSkin(skin);
    skin->drop();

    skin->setColor(EGDC_BUTTON_TEXT, video::SColor(255, 236, 236, 236));
    skin->setColor(EGDC_3D_FACE, video::SColor(255, 16, 16, 16));
    skin->setColor(EGDC_EDITABLE, video::SColor(255, 24, 24, 24));
    skin->setColor(EGDC_FOCUSED_EDITABLE, video::SColor(255, 40, 50, 65));

    skin->setColor(EGDC_SCROLLBAR, video::SColor(255, 0, 0, 0));

    skin->setColor(EGDC_WINDOW_SYMBOL, video::SColor(255, 236, 236, 236)),
        skin->setColor(EGDC_3D_HIGH_LIGHT, video::SColor(255, 32, 32, 32));
    skin->setColor(EGDC_3D_LIGHT, video::SColor(255, 48, 48, 48));
    skin->setColor(EGDC_3D_SHADOW, video::SColor(255, 32, 32, 32));
    skin->setColor(EGDC_3D_DARK_SHADOW, video::SColor(255, 16, 16, 16));

    for (int i = 0; i < irr::gui::EGDC_COUNT; i++)
    {
        video::SColor col = skin->getColor((EGUI_DEFAULT_COLOR)i);
        col.setAlpha(255);
        skin->setColor((EGUI_DEFAULT_COLOR)i, col);
    }

    menu_layout();

    s32 menu_height = skin->getFont()->getDimension(L"A").Height + 5;

    core::rect<s32> quad_rect(core::position2d<s32>(32, menu_height), main_panel_rect.LowerRightCorner);

    cameraQuad = new CameraQuad(env, main_panel,GUI_ID_CAMERA_QUAD, quad_rect, driver);

    core::rect<s32> tool_panel_r(core::position2d<s32>(main_panel_rect.getWidth(), 0), win_rect.LowerRightCorner);

    tool_panel = new multi_tool_panel(env, env->getRootGUIElement(), -1,
        tool_panel_r);

}

void GUI_layout::resize(core::rect<s32> win_rect)
{
    cameraQuad->grab();

    if (main_panel)
        main_panel->remove();

    core::rect<s32> main_panel_rect = win_rect;

    main_panel_rect.LowerRightCorner.X -= 400;

    main_panel = new gui::IGUIElement(gui::EGUIET_ELEMENT, env, env->getRootGUIElement(), -1, main_panel_rect);

    menu_layout();

    s32 menu_height = env->getSkin()->getFont()->getDimension(L"A").Height + 5;

    core::rect<s32> quad_rect(core::position2d<s32>(32, menu_height), main_panel_rect.LowerRightCorner);

    main_panel->addChild(cameraQuad);
    cameraQuad->drop();

    cameraQuad->resize(quad_rect);

    core::rect<s32> tool_panel_r(core::position2d<s32>(main_panel_rect.getWidth(), 0), win_rect.LowerRightCorner);
    tool_panel->resize(tool_panel_r);
}

void GUI_layout::menu_layout()
{
    if (!main_panel || !env)
        return;

    gui::IGUIContextMenu* menu = env->addMenu(main_panel, GUI_ID_MAIN_MENU);
    gui::IGUIContextMenu* submenu;
    gui::IGUIContextMenu* submenu2;

    menu->addItem(L"File", -1, true, true);
    submenu = menu->getSubMenu(0);
    submenu->addItem(L"New Scene", GUI_ID_MENU_FILE_NEW, true, false);
    submenu->addItem(L"Open", GUI_ID_MENU_FILE_OPEN, true, false);
    submenu->addItem(L"Save", GUI_ID_MENU_FILE_SAVE, true, false);
    submenu->addItem(L"Save As", GUI_ID_MENU_FILE_SAVE_AS, true, false);

    menu->addItem(L"Editor", -1, true, true);
    submenu = menu->getSubMenu(1);


}
