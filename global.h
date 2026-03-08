#pragma once
#ifndef _TKT_GLOBAL_
#define _TKT_GLOBAL_

struct aligned_float {
	alignas(16) float x;
};

struct TensorDimension
{
	unsigned B;
	unsigned C;
	unsigned H;
	unsigned W;
};

unsigned long long random_number();
float random_f32();

#endif
