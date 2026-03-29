#pragma once

#include <irrlicht.h>
#include <string>
#include <vector>
#include <random>
#include "fonts.h"

struct Dungeon_Map_Gui_Base;

struct Dungeon_Map_Gui : public irr::gui::IGUIElement
{
    //! constructor
    Dungeon_Map_Gui(Dungeon_Map_Gui_Base* base, irr::gui::IGUIEnvironment* environment, IGUIElement* parent, irr::s32 id, irr::core::rect<irr::s32> rectangle);

    //! destructor
    virtual ~Dungeon_Map_Gui();

    virtual void draw();
    virtual void move(irr::core::vector2di d) override;

    virtual bool OnEvent(const irr::SEvent& event);
    void resize(irr::core::dimension2di);

    void left_click();

    void generate(int seed = 0);
    void rebuild_cache();

    std::wstring text;

    irr::core::vector2di click_pos;
    irr::core::vector2di drag_pos;
    bool bDragging = false;
    bool bMouseDown = false;

    std::vector<char> dungeon_map; //this is the 64x64 grid of chars representing the dungeon

    int tile_w = 0;
    int tile_h = 0;

    Dungeon_Map_Gui_Base* my_base;
};

struct Dungeon_Map_Gui_Base
{
    CachedText_Manager* cached_text = NULL;

    void draw_text(irr::core::position2di origin, const irr::core::rect<irr::s32>& clip_rect);
};