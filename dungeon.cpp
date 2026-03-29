#include <irrlicht.h>
#include "edit_classes.h"
#include "edit_env.h"
#include "GUI_tools.h"
#include "fonts.h"
#include "dungeon.h"

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
// Dungeon generation
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
    const int W = 64, H = 64;

    std::mt19937 rng(seed == 0 ? std::random_device{}() : (unsigned)seed);
    auto randi = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    };

    std::fill(dungeon_map.begin(), dungeon_map.end(), '#');

    auto in_bounds = [&](int x, int y) { return x >= 0 && x < W && y >= 0 && y < H; };
    auto cell      = [&](int x, int y) -> char& { return dungeon_map[y * W + x]; };
    auto is_open   = [&](int x, int y) { return in_bounds(x, y) && cell(x, y) == '.'; };

    auto carve_room = [&](int x, int y, int w, int h) {
        for (int dy = 0; dy < h; dy++)
            for (int dx = 0; dx < w; dx++)
                if (in_bounds(x + dx, y + dy))
                    cell(x + dx, y + dy) = '.';
    };

    // Room must have a clean 1-cell border of walls before placement
    auto room_fits = [&](int x, int y, int w, int h) -> bool {
        if (!in_bounds(x, y) || !in_bounds(x + w - 1, y + h - 1)) return false;
        for (int dy = -1; dy <= h; dy++)
            for (int dx = -1; dx <= w; dx++)
                if (is_open(x + dx, y + dy)) return false;
        return true;
    };

    auto find_open = [&]() -> position2di {
        std::vector<position2di> open;
        open.reserve(256);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (cell(x, y) == '.') open.push_back({ x, y });
        if (open.empty()) return { W / 2, H / 2 };
        return open[randi(0, (int)open.size() - 1)];
    };

    // Place first room near center
    int fw = randi(4, 8), fh = randi(3, 6);
    carve_room(W / 2 - fw / 2, H / 2 - fh / 2, fw, fh);

    // Walker state
    const int dirs[4][2] = { {1,0}, {-1,0}, {0,1}, {0,-1} };
    auto rand_dir = [&](int& dx, int& dy) { int d = randi(0, 3); dx = dirs[d][0]; dy = dirs[d][1]; };
    auto turn_cw  = [](int& dx, int& dy) { int t = dx; dx = -dy; dy =  t; };
    auto turn_ccw = [](int& dx, int& dy) { int t = dx; dx =  dy; dy = -t; };

    position2di start = find_open();
    int wx = start.X, wy = start.Y;
    int wdx = 1, wdy = 0;

    for (int iter = 0; iter < 300; iter++)
    {
        int r = randi(0, 9);

        if (r == 0) // ~10%: teleport to a random open cell
        {
            position2di p = find_open();
            wx = p.X; wy = p.Y;
            rand_dir(wdx, wdy);
        }
        else if (r <= 4) // ~40%: try to place a room near the walker
        {
            int rw = randi(3, 8), rh = randi(3, 6);
            for (int attempt = 0; attempt < 20; attempt++)
            {
                int rx = wx + randi(-rw, 1);
                int ry = wy + randi(-rh, 1);
                if (room_fits(rx, ry, rw, rh))
                {
                    carve_room(rx, ry, rw, rh);
                    wx = rx + rw / 2;
                    wy = ry + rh / 2;
                    break;
                }
            }
        }
        else // ~70%: walk forward, with occasional L-turns
        {
            if (r <= 6) // ~40% of walks: turn before walking
            {
                if (randi(0, 1) == 0) turn_cw(wdx, wdy);
                else                  turn_ccw(wdx, wdy);
            }

            int len = randi(2, 5);
            for (int i = 0; i < len; i++)
            {
                int nx = wx + wdx, ny = wy + wdy;
                if (!in_bounds(nx, ny)) { rand_dir(wdx, wdy); break; }
                cell(nx, ny) = '.';
                wx = nx; wy = ny;
            }
        }
    }

    rebuild_cache();
}
