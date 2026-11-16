#include <algorithm>
#include <random>
#include "dungeon_gen.h"

//============================================================
// Room/corridor generator
//
// Grows the dungeon one transaction at a time: a corridor plus the room at its end.
// The corridor is planned without touching the map, then a room is fitted at its end
// (or, failing that, at the end of an earlier leg). Only if a room fits is anything
// carved, so a corridor never ends without a room and nothing needs undoing.
//
// Chaining: after placing a room, the next corridor usually starts from one of that
// room's walls (room -> corridor -> room ...). When the chain fails or randomly breaks,
// the next start is a weighted pick from all "hot points" (anchors) in the layout,
// where L-bends are favoured so that they become T-junctions.
//

namespace
{
    struct Pos { int x, y; };
    struct Dir { int dx, dy; };

    inline Pos  operator+(Pos p, Dir d)  { return { p.x + d.dx, p.y + d.dy }; }
    inline Pos  operator-(Pos p, Dir d)  { return { p.x - d.dx, p.y - d.dy }; }
    inline Dir  operator-(Dir d)         { return { -d.dx, -d.dy }; }
    inline bool operator==(Pos a, Pos b) { return a.x == b.x && a.y == b.y; }
    inline bool operator==(Dir a, Dir b) { return a.dx == b.dx && a.dy == b.dy; }
    inline Dir  turn_cw(Dir d)           { return { -d.dy,  d.dx }; }
    inline Dir  turn_ccw(Dir d)          { return {  d.dy, -d.dx }; }

    struct Room
    {
        int x, y, w, h;      // interior rectangle
        Dir entry_dir;       // direction the entering corridor travelled ({0,0} for the first room)
        int entered_from;    // corridor index, -1 for the first room
        int feature;
    };

    struct Corridor
    {
        std::vector<Pos> cells;  // first cell sits in the source's wall, last in the target room's wall
        std::vector<int> bends;  // indices into cells
        int from_feature;
        int to_room;
        int feature;
    };

    enum class FeatureKind { Room, Corridor };
    struct Feature { FeatureKind kind; int index; };

    enum class AnchorKind { RoomSide, Bend, CorridorSide };

    // A hot point: the first cell to dig, and the direction to dig in.
    struct Anchor
    {
        Pos pos;
        Dir dir;
        float weight;
        int feature;
        AnchorKind kind;
    };

    struct Plan
    {
        std::vector<Pos> cells;
        std::vector<Dir> dirs;      // direction of travel into each cell
        std::vector<int> leg_ends;  // index of the last cell of each leg
    };

    class Builder
    {
    public:
        Builder(std::vector<char>& map, unsigned seed, const DungeonParams& p)
            : map(map), P(p), rng(seed == 0 ? std::random_device{}() : seed)
        {
            map.assign(W * H, '#');
            owner.assign(W * H, -1);
            plan_mark.assign(W * H, 0);
        }

        void run();

    private:
        static constexpr int W = DUNGEON_W, H = DUNGEON_H;

        std::vector<char>& map;
        const DungeonParams& P;
        std::mt19937 rng;

        std::vector<int> owner;      // feature id per cell, -1 = rock
        std::vector<int> plan_mark;  // == plan_stamp -> cell is part of the plan being tested
        int plan_stamp = 0;

        std::vector<Feature>  features;
        std::vector<Room>     rooms;
        std::vector<Corridor> corridors;
        std::vector<Anchor>   anchors;

        int   randi(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }
        float randf()               { return std::uniform_real_distribution<float>(0.f, 1.f)(rng); }

        int  idx(Pos p) const      { return p.y * W + p.x; }
        bool in_inner(Pos p) const { return p.x >= 1 && p.x < W - 1 && p.y >= 1 && p.y < H - 1; }
        void new_plan()            { ++plan_stamp; }
        void mark(Pos p)           { plan_mark[idx(p)] = plan_stamp; }

        // Rock = not carved and not part of the current plan.
        bool is_rock(Pos p) const
        {
            if (p.x < 0 || p.x >= W || p.y < 0 || p.y >= H) return true;
            return owner[idx(p)] == -1 && plan_mark[idx(p)] != plan_stamp;
        }

        // A corridor cell must be rock, and must not touch open space
        // orthogonally except for the cell it was dug from.
        bool can_dig(Pos c, Pos from) const
        {
            if (!in_inner(c) || !is_rock(c)) return false;
            const Dir dirs[4] = { {1,0}, {-1,0}, {0,1}, {0,-1} };
            for (Dir d : dirs)
            {
                Pos n = c + d;
                if (n == from) continue;
                if (!is_rock(n)) return false;
            }
            return true;
        }

        bool anchor_valid(const Anchor& a) const { return can_dig(a.pos, a.pos - a.dir); }

        // The room plus its 1-cell wall border must be rock, except `door`
        // (the corridor's last cell, which sits in the border).
        bool room_fits(int x, int y, int w, int h, Pos door) const
        {
            if (x < 1 || y < 1 || x + w > W - 1 || y + h > H - 1) return false;
            for (int dy = -1; dy <= h; dy++)
                for (int dx = -1; dx <= w; dx++)
                {
                    Pos p{ x + dx, y + dy };
                    if (p == door) continue;
                    if (!is_rock(p)) return false;
                }
            return true;
        }

        int  add_feature(FeatureKind k, int index) { features.push_back({ k, index }); return (int)features.size() - 1; }
        void carve(Pos p, int feature)             { map[idx(p)] = '.'; owner[idx(p)] = feature; }

        int  place_room(int x, int y, int w, int h, Dir entry_dir, int entered_from);
        void add_room_anchors(const Room& r);
        void add_corridor_anchors(const Corridor& c, const std::vector<Dir>& dirs);

        void plan_corridor(const Anchor& a, Plan& plan);
        bool fit_room_at(Pos end, Dir d, int& rx, int& ry, int& rw, int& rh);
        int  try_transaction(const Anchor& a);

        void prune_anchors();
        int  pick_anchor(int only_feature, Dir exclude_dir);
    };

    //------------------------------------------------------------
    // Features
    //

    int Builder::place_room(int x, int y, int w, int h, Dir entry_dir, int entered_from)
    {
        Room r{ x, y, w, h, entry_dir, entered_from, -1 };
        int ri = (int)rooms.size();
        r.feature = add_feature(FeatureKind::Room, ri);
        rooms.push_back(r);

        for (int dy = 0; dy < h; dy++)
            for (int dx = 0; dx < w; dx++)
                carve({ x + dx, y + dy }, r.feature);

        add_room_anchors(r);
        return ri;
    }

    void Builder::add_room_anchors(const Room& r)
    {
        // One anchor per wall cell (corners excluded); each side's weights sum to weight_room_side.
        float wx = P.weight_room_side / r.w;
        float wy = P.weight_room_side / r.h;
        for (int i = 0; i < r.w; i++)
        {
            anchors.push_back({ { r.x + i, r.y - 1   }, {  0, -1 }, wx, r.feature, AnchorKind::RoomSide });
            anchors.push_back({ { r.x + i, r.y + r.h }, {  0,  1 }, wx, r.feature, AnchorKind::RoomSide });
        }
        for (int i = 0; i < r.h; i++)
        {
            anchors.push_back({ { r.x - 1,   r.y + i }, { -1, 0 }, wy, r.feature, AnchorKind::RoomSide });
            anchors.push_back({ { r.x + r.w, r.y + i }, {  1, 0 }, wy, r.feature, AnchorKind::RoomSide });
        }
    }

    void Builder::add_corridor_anchors(const Corridor& c, const std::vector<Dir>& dirs)
    {
        // Straight through each bend -> T junction.
        for (int b : c.bends)
            anchors.push_back({ c.cells[b] + dirs[b], dirs[b], P.weight_bend, c.feature, AnchorKind::Bend });

        // One side-branch candidate per leg, away from the leg's ends.
        std::vector<int> ends = c.bends;
        ends.push_back((int)c.cells.size() - 1);
        int start = 0;
        for (int e : ends)
        {
            if (e - start >= 2)
            {
                int i = randi(start + 1, e - 1);
                Dir side = turn_cw(dirs[i]);
                anchors.push_back({ c.cells[i] + side,    side,  P.weight_corridor * 0.5f, c.feature, AnchorKind::CorridorSide });
                anchors.push_back({ c.cells[i] - side, -side,    P.weight_corridor * 0.5f, c.feature, AnchorKind::CorridorSide });
            }
            start = e + 1;
        }
    }

    //------------------------------------------------------------
    // Planning / testing
    //

    void Builder::plan_corridor(const Anchor& a, Plan& plan)
    {
        new_plan();
        plan = {};

        int legs = randi(1, P.corridor_max_bends + 1);
        Dir d = a.dir;
        Pos cur = a.pos - a.dir;  // the source cell we dig out of

        for (int leg = 0; leg < legs; leg++)
        {
            if (leg > 0)
                d = randi(0, 1) ? turn_cw(d) : turn_ccw(d);

            int len = randi(P.corridor_min_leg, P.corridor_max_leg);
            int dug = 0;
            for (; dug < len; dug++)
            {
                Pos n = cur + d;
                if (!can_dig(n, cur)) break;
                plan.cells.push_back(n);
                plan.dirs.push_back(d);
                mark(n);
                cur = n;
            }

            if (dug > 0)
                plan.leg_ends.push_back((int)plan.cells.size() - 1);
            if (dug < len)
                break;  // blocked: keep what we have, the leg ends become candidate room sites
        }
    }

    bool Builder::fit_room_at(Pos end, Dir d, int& rx, int& ry, int& rw, int& rh)
    {
        Pos s = end + d;  // first room cell, straight on from the corridor
        for (int attempt = 0; attempt < P.room_fit_attempts; attempt++)
        {
            int w = randi(P.room_min_w, P.room_max_w);
            int h = randi(P.room_min_h, P.room_max_h);
            int x, y;
            if (d.dx != 0)
            {
                x = d.dx > 0 ? s.x : s.x - w + 1;
                y = s.y - randi(0, h - 1);
            }
            else
            {
                y = d.dy > 0 ? s.y : s.y - h + 1;
                x = s.x - randi(0, w - 1);
            }

            if (room_fits(x, y, w, h, end))
            {
                rx = x; ry = y; rw = w; rh = h;
                return true;
            }
        }
        return false;
    }

    // Returns the new room index, or -1 if nothing was added.
    int Builder::try_transaction(const Anchor& a)
    {
        Plan plan;
        plan_corridor(a, plan);
        if (plan.cells.size() < 2) return -1;

        // Try the full corridor first, then walk it back one leg at a time.
        for (int k = (int)plan.leg_ends.size() - 1; k >= 0; k--)
        {
            int last = plan.leg_ends[k];
            if (last + 1 < 2) continue;

            // Re-stamp only the prefix being tested so discarded legs don't block the room.
            new_plan();
            for (int i = 0; i <= last; i++) mark(plan.cells[i]);

            int rx, ry, rw, rh;
            if (!fit_room_at(plan.cells[last], plan.dirs[last], rx, ry, rw, rh))
                continue;

            // Commit.
            new_plan();  // clear plan marks; everything below is real

            Corridor c;
            c.cells.assign(plan.cells.begin(), plan.cells.begin() + last + 1);
            std::vector<Dir> dirs(plan.dirs.begin(), plan.dirs.begin() + last + 1);
            for (int j = 0; j < k; j++) c.bends.push_back(plan.leg_ends[j]);
            c.from_feature = a.feature;

            int ci = (int)corridors.size();
            c.feature = add_feature(FeatureKind::Corridor, ci);
            for (Pos p : c.cells) carve(p, c.feature);

            int ri = place_room(rx, ry, rw, rh, dirs.back(), ci);
            c.to_room = ri;

            add_corridor_anchors(c, dirs);
            corridors.push_back(std::move(c));
            return ri;
        }
        return -1;
    }

    //------------------------------------------------------------
    // Anchor selection
    //

    void Builder::prune_anchors()
    {
        new_plan();
        anchors.erase(std::remove_if(anchors.begin(), anchors.end(),
            [&](const Anchor& a) { return a.weight < 0.01f || !anchor_valid(a); }),
            anchors.end());
    }

    // Weighted pick. only_feature < 0 means any feature.
    int Builder::pick_anchor(int only_feature, Dir exclude_dir)
    {
        float total = 0.f;
        for (const Anchor& a : anchors)
            if ((only_feature < 0 || a.feature == only_feature) && !(a.dir == exclude_dir))
                total += a.weight;
        if (total <= 0.f) return -1;

        float r = randf() * total;
        int last = -1;
        for (int i = 0; i < (int)anchors.size(); i++)
        {
            const Anchor& a = anchors[i];
            if ((only_feature >= 0 && a.feature != only_feature) || a.dir == exclude_dir)
                continue;
            last = i;
            r -= a.weight;
            if (r <= 0.f) return i;
        }
        return last;
    }

    //------------------------------------------------------------
    // Main loop
    //

    void Builder::run()
    {
        int fw = randi(P.first_room_min_w, P.first_room_max_w);
        int fh = randi(P.first_room_min_h, P.first_room_max_h);
        int active_room = place_room(W / 2 - fw / 2, H / 2 - fh / 2, fw, fh, { 0, 0 }, -1);

        int failures = 0;
        int chain_failures = 0;

        while ((int)rooms.size() < P.target_rooms && failures < P.max_failures)
        {
            prune_anchors();
            if (anchors.empty()) break;

            int ai;
            if (active_room >= 0)
            {
                // Continue the chain from the current room, but not back out of its entry wall.
                const Room& r = rooms[active_room];
                ai = pick_anchor(r.feature, -r.entry_dir);
                if (ai < 0) { active_room = -1; continue; }
            }
            else
            {
                ai = pick_anchor(-1, { 0, 0 });
                if (ai < 0) break;
            }

            Anchor a = anchors[ai];
            int ri = try_transaction(a);

            if (ri >= 0)
            {
                anchors.erase(anchors.begin() + ai);  // new anchors were appended, ai is still valid
                chain_failures = 0;
                active_room = (randf() < P.jump_chance) ? -1 : ri;
            }
            else
            {
                failures++;
                anchors[ai].weight *= 0.5f;
                if (active_room >= 0 && ++chain_failures >= P.chain_failures_before_jump)
                {
                    active_room = -1;
                    chain_failures = 0;
                }
            }
        }
    }
}

void generate_dungeon(std::vector<char>& map, unsigned seed, const DungeonParams& params)
{
    Builder b(map, seed, params);
    b.run();
}
