#include <irrlicht.h>
#include "edit_classes.h"
#include "edit_env.h"
#include "GUI_tools.h"
#include "fonts.h"
#include "dungeon.h"
#include "dungeon_gen.h"

using namespace irr;
using namespace gui;
using namespace core;
using namespace video;

//============================================================
// Dungeon_Map_Gui_Base
//

void Dungeon_Map_Gui_Base::draw_text(position2di origin, const rect<s32>& clip_rect)
{
    if (cached_text)
        cached_text->draw(origin, SColor(255, 200, 200, 200), &clip_rect);
}

//============================================================
// Dungeon_Map_Gui
//

Dungeon_Map_Gui::Dungeon_Map_Gui(Dungeon_Map_Gui_Base* base, IGUIEnvironment* environment, IGUIElement* parent, s32 id, rect<s32> rectangle)
    : IGUIElement(EGUIET_ELEMENT, environment, parent, id, rectangle), my_base(base)
{
    dungeon_map.resize(64 * 64, '#');

    AGG_TT_Font* font = (AGG_TT_Font*)environment->getSkin()->getFont();

    if (!my_base->cached_text)
        my_base->cached_text = new CachedText_Manager(font);

    const int COLS = 64;
    const int ROWS = 64;
    dimension2d<u32> char_size = font->getDimension(L"#");
    tile_w = (int)char_size.Width  + 2;
    tile_h = (int)char_size.Height - 4;

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            wchar_t ch[2] = { (wchar_t)dungeon_map[row * COLS + col], 0 };
            CachedText ct;
            font->getCachedText(ct, ch,
                position2di{ col * tile_w, row * tile_h },
                dimension2di{ tile_w, tile_h },
                TKT_ALIGN_LEFT);
            my_base->cached_text->cache(ct, position2di{ 0, 0 });
        }
    }

    my_base->cached_text->fill_positions();

    resize(dimension2di{ COLS * tile_w, ROWS * tile_h });
}

Dungeon_Map_Gui::~Dungeon_Map_Gui()
{
}

void Dungeon_Map_Gui::draw()
{
    if (!IsVisible)
        return;

    IGUISkin* skin = Environment->getSkin();
    skin->draw3DToolBar(this, AbsoluteRect, &AbsoluteRect);

    my_base->draw_text(AbsoluteRect.UpperLeftCorner, AbsoluteClippingRect);

    IGUIElement::draw();
}

void Dungeon_Map_Gui::move(vector2di d)
{
    IGUIElement::move(d);
}

bool Dungeon_Map_Gui::OnEvent(const SEvent& event)
{
    if (event.EventType == EET_MOUSE_INPUT_EVENT)
    {
        vector2di mouse_pos = vector2di{ event.MouseInput.X, event.MouseInput.Y };

        switch (event.MouseInput.Event)
        {
        case EMIE_LMOUSE_LEFT_UP:
            bDragging = false;
            if (mouse_pos == click_pos)
                left_click();
            break;
        case EMIE_LMOUSE_PRESSED_DOWN:
            if (!bMouseDown)
            {
                bDragging = true;
                click_pos = mouse_pos;
                drag_pos  = mouse_pos;
            }
            break;
        case EMIE_MOUSE_MOVED:
            if (bDragging)
            {
                move(mouse_pos - drag_pos);
                drag_pos = mouse_pos;
            }
            break;
        }

        return true;
    }

    return IGUIElement::OnEvent(event);
}

void Dungeon_Map_Gui::resize(dimension2di nsize)
{
    this->DesiredRect = recti{ this->RelativeRect.UpperLeftCorner, nsize };
    this->updateAbsolutePosition();
}

void Dungeon_Map_Gui::left_click()
{
    generate();
}

//============================================================
// Map rendering / generation
//

void Dungeon_Map_Gui::rebuild_cache()
{
    const int COLS = 64;
    const int ROWS = 64;

    AGG_TT_Font* font = my_base->cached_text->my_font;

    my_base->cached_text->text.clear();
    my_base->cached_text->pos.clear();

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            wchar_t ch[2] = { (wchar_t)dungeon_map[row * COLS + col], 0 };
            CachedText ct;
            font->getCachedText(ct, ch,
                position2di{ col * tile_w, row * tile_h },
                dimension2di{ tile_w, tile_h },
                TKT_ALIGN_LEFT);
            my_base->cached_text->cache(ct, position2di{ 0, 0 });
        }
    }

    my_base->cached_text->fill_positions();
}

void Dungeon_Map_Gui::generate(int seed)
{
    generate_dungeon(dungeon_map, (unsigned)seed);
    rebuild_cache();
}
