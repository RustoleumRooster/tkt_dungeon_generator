
#include "CameraPanel.h"
#include <irrlicht.h>
#include <iostream>
#include "edit_env.h"
#include "CMeshBuffer.h"
//#include "utils.h"
//#include "node_properties.h"
//#include "geometry_scene.h"
//#include "CMeshSceneNode.h"


extern IrrlichtDevice* device;
using namespace irr;
using namespace core;
using namespace gui;

extern IEventReceiver* ContextMenuOwner;
extern TestPanel* Active_Camera_Window;

extern irr::video::ITexture* small_circle_tex_add_selected;
extern irr::video::ITexture* small_circle_tex_add_not_selected;
extern irr::video::ITexture* small_circle_tex_sub_selected;
extern irr::video::ITexture* small_circle_tex_sub_not_selected;
extern irr::video::ITexture* small_circle_tex_red_selected;
extern irr::video::ITexture* small_circle_tex_red_not_selected;

extern irr::video::ITexture* med_circle_tex_add_selected;
extern irr::video::ITexture* med_circle_tex_add_not_selected;
extern irr::video::ITexture* med_circle_tex_sub_selected;
extern irr::video::ITexture* med_circle_tex_sub_not_selected;
extern irr::video::ITexture* med_circle_tex_red_selected;
extern irr::video::ITexture* med_circle_tex_red_not_selected;


//=============================================================================================
// Camera Panel 2D
//
//

TestPanel_2D::TestPanel_2D(IGUIEnvironment* environment, video::IVideoDriver* driver, scene::ISceneManager* smgr, IGUIElement* parent, s32 id, core::rect<s32> rectangle)
: TestPanel(environment, driver, parent, id, rectangle)
{
    this->smgr = smgr;
	bShowGeometry=false;
	bShowBrushes=true;
}

void TestPanel_2D::setImage(video::ITexture* image)
{
    if (image == Texture)
		return;

	if (Texture)
		Texture->drop();

	Texture = image;

	if (Texture)
    {
		Texture->grab();
		viewSize=Texture->getOriginalSize()*4;
    }
}

void TestPanel_2D::resize(core::dimension2d<u32> new_size)
{
    viewSize = this->Texture->getOriginalSize() * 4;

    core::matrix4 M;
    M.buildProjectionMatrixOrthoLH(this->Texture->getOriginalSize().Width * 4, this->Texture->getOriginalSize().Height * 4, 0, 10000);
    getCamera()->setProjectionMatrix(M, true);
}

scene::ICameraSceneNode* TestPanel_2D::getCamera()
{
    if(this->camera == NULL)
    {
        if(this->smgr)
        {
            core::matrix4 M;
            M.buildProjectionMatrixOrthoLH(this->Texture->getOriginalSize().Width*4,this->Texture->getOriginalSize().Height*4,0,10000);
            this->camera = smgr->addCameraSceneNode(0,core::vector3df(0,1000,0),core::vector3df(0,0,0),-1,false);

            this->camera->setProjectionMatrix(M,true);
        }
    }
    return this->camera;
}

void TestPanel_2D::resetCamera()
{
    this->setAxis(m_axis);
    this->resize(core::dimension2du{});
}

f32 TestPanel_2D::getViewScaling()
{
    return ((f32)this->viewSize.Width / this->Texture->getOriginalSize().Width);
}

void TestPanel_2D::setAxis(int axis)
{
    if(axis==CAMERA_Y_AXIS)
    {
        this->getCamera()->setPosition(core::vector3df(0,5000,0));
        this->getCamera()->setTarget(core::vector3df(0,0,0));
        this->vHorizontal=core::vector3df(0,0,1);
        this->vVertical=core::vector3df(1,0,0);
        this->vAxis=core::vector3df(0,1,0);
    }
    else if(axis==CAMERA_X_AXIS)
    {
        this->getCamera()->setPosition(core::vector3df(5000,0,0));
        this->getCamera()->setTarget(core::vector3df(0,0,0));
        this->vHorizontal=core::vector3df(0,0,-1);
        this->vVertical=core::vector3df(0,1,0);
        this->vAxis=core::vector3df(1,0,0);
    }
    else if(axis==CAMERA_Z_AXIS)
    {
        this->getCamera()->setPosition(core::vector3df(0,0,5000));
        this->getCamera()->setTarget(core::vector3df(0,0,0));
        this->vHorizontal=core::vector3df(1,0,0);
        this->vVertical=core::vector3df(0,1,0);
        this->vAxis=core::vector3df(0,0,1);
    }
    this->m_axis=axis;
}


bool TestPanel_2D::OnEvent(const SEvent& event)
{
	{
		switch(event.EventType)

		{
		    case EET_GUI_EVENT:
                switch(event.GUIEvent.EventType)
                {
                    case EGET_MENU_ITEM_SELECTED:
                        {
                            this->OnMenuItemSelected((gui::IGUIContextMenu*)event.GUIEvent.Caller);
                            return true;
                        }
                        break;
                    case EGET_ELEMENT_HOVERED:
                        {

                        }
                        break;
                    case EGET_ELEMENT_LEFT:
                        {

                        }
                        break;
                    case EGET_ELEMENT_FOCUS_LOST:
                        {

                            this->bMouseDown=false;
                            this->rMouseDown=false;
                            this->bDragBrush=false;
                            this->bZoomCamera=false;
                            this->bDragCamera=false;
                        }
                        break;
/*
                    case EGET_ELEMENT_FOCUS_LOST:
                       // if (event.GUIEvent.Caller == this)// && !isMyChild(event.GUIEvent.Element) )
                        {
                           // std::cout<<"lost focus\n";

                            if (event.GUIEvent.Caller == this && isMyChild(event.GUIEvent.Element))
                            {
                                  //  Environment->setFocus(event.GUIEvent.Element);
                                  //  std::cout<<"me\n";
                                  //  return true;
                            }

                            //return false;
                        }
                        break;
                    case EGET_ELEMENT_FOCUSED:
                        if (event.GUIEvent.Caller == this )
                        {
                            //std::cout<<"focus\n";
                            //return true;
                            //return false;
                        }
                        break;
*/
                    default:
                        break;
                }
                break;
		    case EET_MOUSE_INPUT_EVENT:

                mousex = event.MouseInput.X;
                mousey = event.MouseInput.Y;
                bShiftDown = event.MouseInput.Shift;
                bCtrlDown = event.MouseInput.Control;

                switch(event.MouseInput.Event)
                {
                    case EMIE_LMOUSE_LEFT_UP:
                        {
                            if(bZoomCamera)
                            {
                                bZoomCamera=false;
                            }

                            if(clickx==mousex && clicky==mousey)
                                left_click(core::vector2di(event.MouseInput.X,event.MouseInput.Y));

                            bMouseDown=false;
                            bDragCamera=false;
                            bDragBrush=false;
                            bDragNode = false;
                            bDragVertex = false;

                        }
                        return true;
                    case EMIE_LMOUSE_PRESSED_DOWN:
                        {

                            clickx=mousex;
                            clicky=mousey;

                            //Initiate Drag
                            if(!bMouseDown && !rMouseDown)
                            {
                                bDragCamera = true;
                                startCameraPan(vector2di{ mousex,mousey });
                            }
                            else if(bMouseDown == false)
                            {
                                clickx=mousex;
                                clicky=mousey;
                                bMouseDown = true;
                            }
                        }
                        return true;
                    case EMIE_RMOUSE_PRESSED_DOWN:
                        {
                           
                            if(!rMouseDown)
                            {
                                clickx=mousex;
                                clicky=mousey;
                                rMouseDown=true;
                            }
                            
                        }
                        return true;
                    case EMIE_RMOUSE_LEFT_UP:
                        {
                            if (bDragNode)
                            {
                                bDragNode = false;
                                //geo_scene->selectionChanged();
                            }
                            if (bRotateNode)
                            {
                                bRotateNode = false;
                               // geo_scene->selectionChanged();
                            }

                            /* TODO
                            if(!AbsoluteClippingRect.isPointInside( core::position2d<s32>(event.MouseInput.X, event.MouseInput.Y ) ))
                            {
                                bZoomCamera=false;
                                bDragVertex=false;  
                               // Environment->removeFocus(this);
                            }*/

                            if (bDragBrush)
                            {
                                bDragBrush = false;
                            }
                            if(bDragVertex)
                            {
                                bDragVertex=false;
                            }
                           
                            if (clickx == mousex && clicky == mousey)
                            {
                                if (this->m_viewPanel)
                                {
                                    vector2d<s32> clickpos = m_viewPanel->getClickPos();
                                    right_click(clickpos);
                                }
                                
                            }

                            rMouseDown=false;
                        }
                        return true;
                    case EMIE_MOUSE_MOVED:

                        if( bMouseDown == true && rMouseDown == true && bZoomAllowed)
                        {
                            if(bZoomCamera==false)
                            {
                                vDragCameraInitialPosition = this->getCamera()->getAbsolutePosition();
                                vDragCameraInitialTarget = this->getCamera()->getTarget();
                                vDragCameraRay = this->getCamera()->getAbsolutePosition()-this->getCamera()->getTarget();
                                this->oldViewSize=this->viewSize;

                                bZoomCamera=true;
                                bDragCamera=false;
                            }
                            else
                            {

                                int ydif = mousey-clicky;
                                float zoom_f = 1.0-(0.01 * ydif);
                                zoom_f = fmax(zoom_f,0.01);

                                core::matrix4 M;
                                M.buildProjectionMatrixOrthoLH(oldViewSize.Width*(zoom_f),oldViewSize.Height*(zoom_f),10,10000);
                                this->camera->setProjectionMatrix(M,true);
                                this->viewSize.Width=oldViewSize.Width*(zoom_f);
                                this->viewSize.Height=oldViewSize.Height*(zoom_f);
                            }
                        }
                        else if (bMouseDown == true && rMouseDown == true && !bZoomAllowed)
                        {
                            bDragCamera = false;
                        }
                        else if(bDragCamera )
                        {
                            int ydif = mousey-clicky;
                            int xdif = mousex-clickx;
                            

                            vector2di move{ xdif,ydif };

                            cameraPan(move);

                        }
                        return true;
                default:
                    break;
			}
			break;
		default:
			break;
		}
		
	}
    //return IGUIElement::OnEvent(event);
    return false;
}

void TestPanel_2D::startCameraPan(vector2di)
{
    vDragCameraInitialPosition = this->getCamera()->getAbsolutePosition();
    vDragCameraInitialTarget = this->getCamera()->getTarget();

    bRotateCamera = false;
    bZoomCamera = false;

    bMouseDown = true;
}

void TestPanel_2D::cameraPan(vector2di d)
{
    core::vector3df vMove = d.X * this->vHorizontal + d.Y * this->vVertical;
    vMove *= ((f32)this->viewSize.Width / this->Texture->getOriginalSize().Width);
    this->getCamera()->setPosition(vDragCameraInitialPosition + vMove);
    this->getCamera()->setTarget(vDragCameraInitialTarget + vMove);
    this->getCamera()->updateAbsolutePosition();
}

void TestPanel_2D::left_click(core::vector2di pos)
{
   
}

void TestPanel_2D::right_click(core::vector2di pos)
{

        gui::IGUIContextMenu* menu = environment->addContextMenu(core::rect<s32>(pos,core::vector2di(256,256)),0,-1);
        menu->addItem(L"View  ",-1,true,true,false,false);
        menu->addItem(L"Grid  ",GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_GRID_TOGGLE,true,false,true,true);
        menu->addItem(L"Camera ", -1, true, true, false, false);
        menu->setItemChecked(1,this->bShowGrid);

        gui::IGUIContextMenu* submenu;
        submenu = menu->getSubMenu(0);
        submenu->addItem(L"Brushes",GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_VIEW_BRUSHES,true,false,true,true);
        submenu->addItem(L"Geometry ",GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_VIEW_GEOMETRY,true,false,true,true);

        submenu->setItemChecked(0,this->bShowBrushes);
        submenu->setItemChecked(1,this->bShowGeometry);

        submenu = menu->getSubMenu(2);
        submenu->addItem(L"Reset", GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_CAMERA_RESET, true, false, false, false);

        ContextMenuOwner = m_viewPanel;
}

void TestPanel_2D::delete_selected_brushes()
{

}

void TestPanel_2D::OnMenuItemSelected(IGUIContextMenu* menu)
{
    s32 id = menu->getItemCommandId(menu->getSelectedItem());
    gui::IGUIEnvironment* env = device->getGUIEnvironment();
    gui::IGUIElement* root = env->getRootGUIElement();
    switch(id)
    {
    case GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_GRID_TOGGLE:
       // this->toggle_grid();
        break;
    case GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_VIEW_BRUSHES:
       // this->toggle_showBrushes();
        break;
    case GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_VIEW_GEOMETRY:
      //  this->toggle_showGeometry();
        break;
    case GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_CAMERA_RESET:
        this->resetCamera();
        break;
    case GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_DELETE_BRUSH:
        this->delete_selected_brushes();
        break;
    case GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_NODE_PROPERTIES:
        //NodeProperties_Tool::show();
        break;
    case GUI_ID_VIEWPORT_2D_RIGHTCLICK_MENU_ITEM_SAVE_NODE_SELECTION:
        //geo_scene->save_selection();
        break;
    default:
        break;
    }
}

bool TestPanel_2D::GetOrthoScreenCoords(core::vector3df V, core::vector2di &out_coords)
{
    const scene::SViewFrustum* frustum = this->getCamera()->getViewFrustum();
    const core::vector3df cameraPosition = this->getCamera()->getAbsolutePosition();

    vector3df vNearLeftDown = frustum->getNearLeftDown();
    vector3df vNearRightDown = frustum->getNearRightDown();
    vector3df vNearLeftUp = frustum->getNearLeftUp();
    vector3df vNearRightUp = frustum->getNearRightUp();

    vector3df ray = frustum->getNearLeftDown() - frustum->getFarLeftDown();

    core::plane3df aplane(vNearLeftDown,vNearLeftUp,vNearRightDown);
    core::vector3df vIntersect;

    if(aplane.getIntersectionWithLine(V,ray,vIntersect))
    {
        core::vector3df v_X = (vNearRightDown - vNearLeftDown);
        v_X.normalize();
        f32 t_X = v_X.dotProduct(vIntersect-vNearLeftDown);
        t_X /= (vNearRightDown - vNearLeftDown).getLength();

        core::vector3df v_Y = (vNearLeftUp - vNearLeftDown);
        v_Y.normalize();
        f32 t_Y = v_Y.dotProduct(vIntersect-vNearLeftDown);
        t_Y /= (vNearLeftUp - vNearLeftDown).getLength();

        out_coords.X = core::round32(t_X*this->Texture->getOriginalSize().Width);
        out_coords.Y = core::round32((1-t_Y)*this->Texture->getOriginalSize().Height);

        return true;
    }
    return false;

}


void TestPanel_2D::render()
{
    driver->setRenderTarget(getImage(), true, true, video::SColor(255,16,16,16));
    smgr->setActiveCamera(getCamera());

    Active_Camera_Window = this;

    smgr->drawAll();

    getCamera()->render();

    video::SMaterial someMaterial;
    someMaterial.Lighting = false;
    someMaterial.Thickness = 1.0;
    someMaterial.MaterialType = video::EMT_SOLID;

    driver->setTransform(video::ETS_WORLD, core::IdentityMatrix);
    driver->setMaterial(someMaterial);

    if(bShowGrid)
        this->drawGrid(driver,someMaterial);

    driver->setRenderTarget(0, true, true, video::SColor(0,0,0,0));
}

void TestPanel_2D::drawGrid(video::IVideoDriver* driver, const video::SMaterial material)
{

    int far_value=this->getCamera()->getFarValue();
    far_value*=-1;

    if(m_axis==CAMERA_Y_AXIS)
    {
        int interval = this->grid_interval;
        while(viewSize.Width / interval > 36 || viewSize.Height / interval > 36)
        {
            interval = interval<<1;
        }
        int h_lines = viewSize.Width / interval;
        int v_lines = viewSize.Height / interval;
        core::vector3df vDownLeft = this->getCamera()->getAbsolutePosition();
        vDownLeft.Z-=viewSize.Width/2;
        vDownLeft.X-=viewSize.Height/2;
        int start_x = interval*((int)vDownLeft.Z/interval);
        int start_y = interval*((int)vDownLeft.X/interval);

        video::SColor col;
        for(int i=-1;i<h_lines+2;i++)
        {
            if(start_x+i*interval==0)
                col = video::SColor(255,128,128,128);
            else
                col = video::SColor(255,32,32,32);

            driver->draw3DLine(core::vector3df(vDownLeft.X,far_value,start_x+i*interval),core::vector3df(vDownLeft.X+viewSize.Height,far_value,start_x+i*interval), col);
        }

        for(int i=-1;i<v_lines+2;i++)
        {
            if(start_y+i*interval==0)
                col = video::SColor(255,128,128,128);
            else
                col = video::SColor(255,32,32,32);
            driver->draw3DLine(core::vector3df(start_y+i*interval,far_value,vDownLeft.Z),core::vector3df(start_y+i*interval,far_value,vDownLeft.Z+viewSize.Width), col);
        }
    }
    else if(m_axis==CAMERA_X_AXIS)
    {
        int interval = this->grid_interval;
        while(viewSize.Width / interval > 36 || viewSize.Height / interval > 36)
        {
            interval = interval<<1;
        }
        int h_lines = viewSize.Width / interval;
        int v_lines = viewSize.Height / interval;
        core::vector3df vDownLeft = this->getCamera()->getAbsolutePosition();
        vDownLeft.Z-=viewSize.Width/2;
        vDownLeft.Y-=viewSize.Height/2;
        int start_x = interval*((int)vDownLeft.Z/interval);
        int start_y = interval*((int)vDownLeft.Y/interval);

        video::SColor col;
        for(int i=-1;i<h_lines+2;i++)
        {
            if(start_x+i*interval==0)
                col = video::SColor(255,128,128,128);
            else
                col = video::SColor(255,32,32,32);
            driver->draw3DLine(core::vector3df(far_value,vDownLeft.Y,start_x+i*interval),core::vector3df(far_value,vDownLeft.Y+viewSize.Height,start_x+i*interval),col);
        }

        for(int i=-1;i<v_lines+2;i++)
        {
            if(start_y+i*interval==0)
                col = video::SColor(255,128,128,128);
            else
                col = video::SColor(255,32,32,32);
            driver->draw3DLine(core::vector3df(far_value,start_y+i*interval,vDownLeft.Z),core::vector3df(far_value,start_y+i*interval,vDownLeft.Z+viewSize.Width),col);
        }

    }
    else if(m_axis==CAMERA_Z_AXIS)
    {
        int interval = this->grid_interval;
        while(viewSize.Width / interval > 36 || viewSize.Height / interval > 36)
        {
            interval = interval<<1;
        }
        int h_lines = viewSize.Width / interval;
        int v_lines = viewSize.Height / interval;
        core::vector3df vDownLeft = this->getCamera()->getAbsolutePosition();
        vDownLeft.X-=viewSize.Width/2;
        vDownLeft.Y-=viewSize.Height/2;
        int start_x = interval*((int)vDownLeft.X/interval);
        int start_y = interval*((int)vDownLeft.Y/interval);

        video::SColor col;

        for(int i=-1;i<h_lines+2;i++)
        {
            if(start_x+i*interval==0)
                col = video::SColor(255,128,128,128);
            else
                col = video::SColor(255,32,32,32);
            driver->draw3DLine(core::vector3df(start_x+i*interval,vDownLeft.Y,far_value),core::vector3df(start_x+i*interval,vDownLeft.Y+viewSize.Height,far_value),col);
        }

        for(int i=-1;i<v_lines+2;i++)
        {
            if(start_y+i*interval==0)
                col = video::SColor(255,128,128,128);
            else
                col = video::SColor(255,32,32,32);
            driver->draw3DLine(core::vector3df(vDownLeft.X,start_y+i*interval,far_value),core::vector3df(vDownLeft.X+viewSize.Width,start_y+i*interval,far_value),col);
        }
    }
}

bool GetOrthoClickPoint(dimension2d<u32> viewSize, scene::ICameraSceneNode * camera, int clickx, int clicky, vector3df &hit_vec)
{
    const scene::SViewFrustum* frustum = camera->getViewFrustum();
    const core::vector3df cameraPosition = camera->getAbsolutePosition();

    vector3df vNearLeftDown = frustum->getNearLeftDown();
    vector3df vNearRightDown = frustum->getNearRightDown();
    vector3df vNearLeftUp = frustum->getNearLeftUp();
    vector3df vNearRightUp = frustum->getNearRightUp();

    f32 t_X = (f32)clickx / viewSize.Width;
    f32 t_Y = 1.0 - (f32)clicky / viewSize.Height;

    //screen space: Y is horizontal axis
    vector3df X_vec = (vNearRightDown - vNearLeftDown) * t_X;
    vector3df Y_vec = (vNearLeftUp - vNearLeftDown) * t_Y;
    vector3df target = vNearLeftDown + Y_vec + X_vec;

    hit_vec = target;
    return true;
}

