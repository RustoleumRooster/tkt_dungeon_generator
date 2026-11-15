#include <irrlicht.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include "vkModules.h"
#include "vkUtilModules.h"
#include "vkAttentionBlock.h"
#include "vkAttentionTest.h"

//============================================================
// Attention_Block finite-difference gradient check
//
// Fake loss: L = sum(out * R)  =>  dL/dout = R exactly, so no loss module is
// needed and L is a dot product on the CPU.
//

namespace
{
	const float  EPS       = 1e-2f;  // central-difference step (fp32 GPU math: don't go much smaller)
	const double TOL       = 5e-2;   // max allowed relative error
	const double REL_FLOOR = 1e-2;   // denominator floor so ~0 gradients don't blow up the ratio

	size_t n_elems(const TensorDimension& d)
	{
		return (size_t)d.B * d.C * d.H * d.W;
	}

	struct Ctx
	{
		Vulkan_Workflow*      wf     = NULL;
		Create_Tensor_Module* cx     = NULL;  // also used as the upload/download helper
		Create_Tensor_Module* cr     = NULL;
		Attention_Block*      blk    = NULL;
		std::vector<float>    x;
		std::vector<float>    R;
		size_t                n_out  = 0;
	};

	std::vector<float> download(Ctx& c, vkBufferResource* buf, size_t n)
	{
		return c.cx->download_from_buffer(buf, n);
	}

	void upload(Ctx& c, vkBufferResource* buf, const std::vector<float>& v)
	{
		c.cx->upload_to_buffer(buf, v);
	}

	double eval_loss(Ctx& c)
	{
		c.wf->forward_step();
		std::vector<float> out = download(c, c.blk->tail_output().X, c.n_out);
		double L = 0.0;
		for (size_t i = 0; i < c.n_out; i++)
			L += (double)out[i] * (double)c.R[i];
		return L;
	}

	struct Result
	{
		double max_rel = 0.0;
		int    n_fail  = 0;
		int    n       = 0;
	};

	// Central differences on selected entries of a device buffer mirrored in `host`.
	Result check_entries(Ctx& c, const char* name, vkBufferResource* buf,
	                     std::vector<float>& host, const std::vector<float>& analytic,
	                     const std::vector<size_t>& idx)
	{
		Result r;
		printf("\n  %-14s %8s %14s %14s %10s\n", name, "index", "analytic", "numeric", "rel err");

		for (size_t i : idx)
		{
			float v = host[i];

			host[i] = v + EPS; upload(c, buf, host); double Lp = eval_loss(c);
			host[i] = v - EPS; upload(c, buf, host); double Lm = eval_loss(c);
			host[i] = v;       upload(c, buf, host);

			double num = (Lp - Lm) / (2.0 * EPS);
			double a   = analytic[i];
			double rel = std::fabs(a - num) / std::max({ std::fabs(a), std::fabs(num), REL_FLOOR });

			bool ok = rel < TOL;
			r.max_rel = std::max(r.max_rel, rel);
			r.n_fail += ok ? 0 : 1;
			r.n++;

			printf("  %-14s %8zu %14.6f %14.6f %10.2e %s\n", "", i, a, num, rel, ok ? "" : "  <-- FAIL");
		}
		return r;
	}

	std::vector<size_t> random_indices(std::mt19937& rng, size_t n, int count)
	{
		std::uniform_int_distribution<size_t> u(0, n - 1);
		std::vector<size_t> v;
		for (int i = 0; i < count; i++)
			v.push_back(u(rng));
		return v;
	}
}

bool run_attention_gradient_check(Vulkan_App* vulkan)
{
	// Small batch keeps the CPU side fast; T = 64 and D = 128 stay multiples of 16
	// as the tiled shaders require.
	const u32 B = 2, ROWS = 8, COLS = 8, D = 128;
	const u32 T = ROWS * COLS;

	printf("\n==================== Attention_Block gradient check ====================\n");
	printf("  B=%u T=%u D=%u   eps=%g  tol=%g\n", B, T, D, EPS, TOL);

	//--------------------------------------------------------
	// Build workflow
	//
	Vulkan_Workflow wf;
	Ctx c;
	c.wf = &wf;

	c.cx = new Create_Tensor_Module();
	c.cx->dimensions = { B, ROWS, COLS, D };

	c.cr = new Create_Tensor_Module();
	c.cr->dimensions = { B, T, 1, D };

	c.blk = new Attention_Block();
	c.blk->input_dimension = { B, ROWS, COLS, D };

	// No optimizer: parameters must stay fixed during the check.
	wf.Modules.push_back(VkMod_Reference{ c.cx });
	wf.Modules.push_back(VkMod_Reference{ c.cr });
	wf.Modules.push_back(VkMod_Reference{ c.blk });

	std::vector<Vulkan_Module*> append_list;
	for (VkMod_Reference& ref : wf.Modules)
		ref.X->build_workflow(append_list);
	for (Vulkan_Module* mod : append_list)
		wf.Modules.push_back(VkMod_Reference{ mod });

	reflect::connect(&c.cx->output_tensor, c.blk);
	reflect::connect(&c.cr->output_tensor, &c.blk->input_grad);
	reflect::connect(&c.cr->output_tensor, &c.blk->gradient_input());

	wf.initialize(vulkan);

	//--------------------------------------------------------
	// Random inputs (fixed seed so failures are reproducible)
	//
	std::mt19937 rng(1234);
	std::normal_distribution<float> normal(0.0f, 1.0f);

	size_t n_x = n_elems(c.cx->dimensions);
	c.n_out    = n_elems(c.cr->dimensions);

	c.x.resize(n_x);
	c.R.resize(c.n_out);
	for (float& v : c.x) v = normal(rng);
	for (float& v : c.R) v = normal(rng);

	upload(c, c.cx->output_tensor.X, c.x);
	upload(c, c.cr->output_tensor.X, c.R);

	auto& p_pos = c.blk->pos_embed->parameters;
	auto& p_ln  = c.blk->layer_norm->parameters;
	auto& p_qkv = c.blk->qkv->parameters;
	auto& p_o   = c.blk->projection_o->parameters;

	size_t n_pos = n_elems(p_pos.dimensions);
	size_t n_ln  = n_elems(p_ln.dimensions);
	size_t n_qkv = n_elems(p_qkv.dimensions);
	size_t n_o   = n_elems(p_o.dimensions);

	bool all_ok = true;

	//--------------------------------------------------------
	// Check 1: identity path. With W_O = 0 the attention branch contributes
	// nothing to dx, so dx must equal R exactly.
	//
	{
		std::vector<float> w_o   = download(c, p_o.X, n_o);
		std::vector<float> zeros(n_o, 0.0f);
		upload(c, p_o.X, zeros);

		wf.forward_step();
		wf.backward_step(false);
		std::vector<float> dx = download(c, c.blk->gradient_output().X, n_x);

		double max_diff = 0.0;
		for (size_t i = 0; i < n_x; i++)
			max_diff = std::max(max_diff, (double)std::fabs(dx[i] - c.R[i]));

		bool ok = max_diff < 1e-5;
		all_ok &= ok;
		printf("\n[1] Identity path (W_O = 0): max |dx - R| = %.3e  %s\n", max_diff, ok ? "PASS" : "FAIL");

		upload(c, p_o.X, w_o);
	}

	//--------------------------------------------------------
	// Check 2: finite differences
	//
	wf.forward_step();
	wf.backward_step(false);

	std::vector<float> dx     = download(c, c.blk->gradient_output().X, n_x);
	std::vector<float> d_pos  = download(c, p_pos.Y, n_pos);
	std::vector<float> d_ln   = download(c, p_ln.Y,  n_ln);
	std::vector<float> d_qkv  = download(c, p_qkv.Y, n_qkv);
	std::vector<float> d_o    = download(c, p_o.Y,   n_o);

	std::vector<float> h_pos  = download(c, p_pos.X, n_pos);
	std::vector<float> h_ln   = download(c, p_ln.X,  n_ln);
	std::vector<float> h_qkv  = download(c, p_qkv.X, n_qkv);
	std::vector<float> h_o    = download(c, p_o.X,   n_o);

	printf("\n[2] Finite differences (L = sum(out * R))\n");

	std::vector<std::pair<const char*, Result>> results;

	results.push_back({ "x", check_entries(c, "x", c.cx->output_tensor.X, c.x, dx,
		random_indices(rng, n_x, 20)) });

	results.push_back({ "pos_embed", check_entries(c, "pos_embed", p_pos.X, h_pos, d_pos,
		random_indices(rng, n_pos, 10)) });

	// LayerNorm params are interleaved: [d*2+0] = gamma, [d*2+1] = beta
	{
		std::uniform_int_distribution<size_t> ud(0, n_ln / 2 - 1);
		std::vector<size_t> g, b;
		for (int i = 0; i < 8; i++) g.push_back(ud(rng) * 2 + 0);
		for (int i = 0; i < 8; i++) b.push_back(ud(rng) * 2 + 1);
		results.push_back({ "ln_gamma", check_entries(c, "ln_gamma", p_ln.X, h_ln, d_ln, g) });
		results.push_back({ "ln_beta",  check_entries(c, "ln_beta",  p_ln.X, h_ln, d_ln, b) });
	}

	// W_qkv is row-major [D x 3D]: column n in [0,D) = Q, [D,2D) = K, [2D,3D) = V.
	// Checking each third separately exercises each gradient path into the shared dQKV buffer.
	{
		const char* names[3] = { "W_qkv[Q]", "W_qkv[K]", "W_qkv[V]" };
		std::uniform_int_distribution<size_t> ud(0, D - 1);
		for (int third = 0; third < 3; third++)
		{
			std::vector<size_t> idx;
			for (int i = 0; i < 8; i++)
			{
				size_t k = ud(rng);
				size_t n = third * D + ud(rng);
				idx.push_back(k * (3 * D) + n);
			}
			results.push_back({ names[third], check_entries(c, names[third], p_qkv.X, h_qkv, d_qkv, idx) });
		}
	}

	results.push_back({ "W_o", check_entries(c, "W_o", p_o.X, h_o, d_o,
		random_indices(rng, n_o, 10)) });

	//--------------------------------------------------------
	// Summary
	//
	printf("\n  %-12s %8s %12s\n", "tensor", "fails", "max rel");
	for (auto& r : results)
	{
		printf("  %-12s %4d/%-3d %12.2e %s\n", r.first, r.second.n_fail, r.second.n,
		       r.second.max_rel, r.second.n_fail ? "FAIL" : "PASS");
		all_ok &= (r.second.n_fail == 0);
	}
	printf("\n  OVERALL: %s\n", all_ok ? "PASS" : "FAIL");
	printf("========================================================================\n\n");

	wf.shutdown(vulkan);
	return all_ok;
}
