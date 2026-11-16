#pragma once

#include <vector>

// Standalone dungeon generation, independent of the GUI / Irrlicht.
// The map is a row-major DUNGEON_W x DUNGEON_H grid of chars: '#' = wall, '.' = floor.

constexpr int DUNGEON_W = 64;
constexpr int DUNGEON_H = 64;

struct DungeonParams
{
    // Stopping
    int target_rooms = 18;
    int max_failures = 300;       // total failed corridor+room transactions before giving up

    // Rooms (interior size)
    int first_room_min_w = 5, first_room_max_w = 9;
    int first_room_min_h = 4, first_room_max_h = 7;
    int room_min_w = 3, room_max_w = 9;
    int room_min_h = 3, room_max_h = 7;
    int room_fit_attempts = 12;   // size/offset tries at each candidate corridor end

    // Corridors
    int corridor_min_leg = 3, corridor_max_leg = 9;
    int corridor_max_bends = 2;

    // Chaining room -> corridor -> room
    float jump_chance = 0.15f;          // chance to abandon the chain after a success
    int   chain_failures_before_jump = 3;

    // Hot-point weights
    float weight_bend      = 4.0f;  // straight through an L-bend -> T junction
    float weight_room_side = 1.0f;  // total per room wall side
    float weight_corridor  = 0.5f;  // side branch off a straight corridor segment
};

// Fills `map` (resized to DUNGEON_W * DUNGEON_H) with a room/corridor dungeon.
// seed == 0 picks a non-deterministic seed; any other value is reproducible.
void generate_dungeon(std::vector<char>& map, unsigned seed = 0, const DungeonParams& params = {});
