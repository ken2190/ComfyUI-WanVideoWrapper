# Multi-GPU Optimization for WanVideoWrapper

> **Status:** Implementation complete (Tasks 1-6). Pending: integration test on container.

**Goal:** Enable WanVideo 14B model to distribute transformer blocks across multiple GPUs (2x or 8x V100 32GB), eliminating CPU↔GPU block swap overhead and achieving near-linear speedup.

**Architecture:** Two strategies behind a single configurable ComfyUI node (`WanVideoMultiGPU`):
- **Layer Split** — distributes 40 transformer blocks across GPUs (best for 2 GPUs)
- **CFG Parallel** — runs cond/uncond passes on separate GPU groups with layer split within each (best for 3+ GPUs)

Auto-selection: 2 GPUs → layer_split, 3+ GPUs → cfg_parallel.

**Tech Stack:** PyTorch multi-GPU (torch.device), ComfyUI custom nodes, ThreadPoolExecutor for CFG parallel.

### Why not Chunk Parallel?

InfiniteTalk chunks are inherently sequential — each iteration depends on the previous chunk's motion frames (`cond_ = videos[:, -cur_motion_frames_num:]`). True chunk parallelism is impossible without breaking temporal coherence. This strategy was evaluated and excluded.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `multi_gpu.py` | Created | ComfyUI node `WanVideoMultiGPU` — GPU config + strategy selection |
| `multi_gpu_utils.py` | Created | Helpers: GPU detection, device validation, CFG device splitting |
| `__init__.py` | Modified | Registered in OPTIONAL_MODULES |
| `wanvideo/modules/model.py` | Modified | `multi_gpu_layer_split()` method + forward loop device boundary handling |
| `nodes_sampler.py` | Modified | Accepts MULTIGPUOPTS, sets up layer split / CFG parallel, ThreadPoolExecutor |
| `multitalk/multitalk_loop.py` | No change needed | Multi-GPU state lives on transformer object |

## Key Design Decisions

1. **Layer Split is non-invasive**: Blocks stay on assigned GPU. Hidden state `x` and kwargs move at device boundaries only.

2. **Device boundary transfer**: `_move_tensors_to_device()` recursively moves all tensor values at GPU split points.

3. **Replaces block_swap**: When multi-GPU is enabled, `blocks_to_swap=0`. The two are mutually exclusive.

4. **CFG Parallel uses deepcopy**: `copy.deepcopy(transformer)` creates the uncond copy. Falls back to layer_split on failure (OOM).

5. **ThreadPoolExecutor for CFG**: CUDA releases GIL, so cond/uncond run truly in parallel. Results gathered on primary GPU.

---

## Completed Tasks

### Task 1: multi_gpu_utils.py ✅
- `get_available_gpu_count()`, `validate_gpu_config()`, `split_devices_for_cfg()`

### Task 2: multi_gpu.py + __init__.py ✅
- Node with inputs: num_gpus (0=auto), strategy (auto/layer_split/cfg_parallel), primary_gpu
- Registered in OPTIONAL_MODULES

### Task 3: Layer Split in model.py ✅
- `multi_gpu_layer_split()` distributes blocks with tqdm progress
- Forward loop handles device boundaries at lines 3307-3317
- Final `x` moved back to main_device at line 3489

### Task 4: nodes_sampler.py wiring ✅
- MULTIGPUOPTS optional input
- Setup at lines 888-927: validates GPUs, handles cfg_parallel (deepcopy + split) and layer_split
- CFG parallel in predict_with_cfg at lines 1566-1639: ThreadPoolExecutor, device transfer, result gathering
- Cache method propagated to transformer_uncond

### Task 5: multitalk_loop.py ✅
- No changes needed — transformer carries multi-GPU state internally

### Task 6: Workflow JSON ✅
- WanVideoMultiGPU node added to `wf_05_infinite_talk_v100_optimized.json`

### Task 7: Integration test ⬜
- [ ] Sync files to remote server
- [ ] Restart container and verify node loads
- [ ] Run workflow and check logs for multi-GPU activity

---

## Remaining Speedup Opportunities (Beyond Multi-GPU)

Analysis of the full InfiniteTalk pipeline reveals these additional bottlenecks, ranked by impact:

### 1. Attention Mode — NOT APPLICABLE on V100 ⚠️
Flash Attention 2 requires sm_80+ (A100/RTX 3090+). Flash Attention 3 requires sm_90+ (Blackwell). V100 is sm_70 — **cannot use Flash Attention**. SDPA with memory-efficient backend is the best available option and is already the default. Sage Attention may work but requires runtime testing. Added GPU capability detection to PerformanceOpts node for user awareness.

### 2. Async VAE Decode with CUDA Streams ✅ IMPLEMENTED
`async_vae` option in PerformanceOpts. Uses `torch.cuda.Stream()` for non-blocking `vae.to(device)` transfers. Overlaps VAE model transfer with decode input preparation. Applied to: initial encode, per-chunk decode, deferred decode at end.
- Files: `multitalk/multitalk_loop.py` (lines 57-64, 293-296, 530-535, 639-643)

### 3. Cache Device Transfers ✅ IMPLEMENTED
`cache_device_tensors` option. Pre-transfers `wananim_pose_latents` and `wananim_face_pixels` to GPU once before sampling loop. Subsequent per-step `.to(device)` calls become no-ops.
- Files: `nodes_sampler.py` (lines 1222-1226)

### 4. Cache Audio Null Tensor ✅ IMPLEMENTED
`cache_audio_null` option. Caches the `torch.zeros_like(multitalk_audio_input)[-1:]` tensor used for audio CFG guidance. Keyed by shape for safety (handles context-windowed cases). Applied at both cfg_scale=1.0 and cfg_scale>1.0 audio CFG paths.
- Files: `nodes_sampler.py` (lines 1679-1683, 1768-1772)

### 5. Block Swap Async Prefetch ✅ IMPLEMENTED
`block_swap_prefetch` (INT 0-4) option. Re-enables CUDA stream for async block prefetch during block swap. Stream was previously disabled globally ("causes issues") — now conditionally enabled only when user opts in. Prefetch depth controls how many blocks ahead to start transferring.
- Files: `utils.py` (line 99, 115), `nodes_sampler.py` (lines 886-887, 2539), `wanvideo/modules/model.py` (lines 3281-3289)
