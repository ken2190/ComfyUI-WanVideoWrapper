# V100 Performance Optimization Node - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable `WanVideoPerformanceOpts` node to ComfyUI-WanVideoWrapper that optimizes inference speed on V100 32GB GPUs (and similar older hardware) through fp16 autocast, cuDNN benchmark, and VAE encode caching.

**Architecture:** A new node outputs a `PERFORMANCEOPTS` dict consumed by `WanVideoSampler`. The sampler applies fp16 autocast and cuDNN benchmark around inference, and passes VAE optimization flags to `multitalk_loop` which skips redundant encodes and optionally transfers motion frames in latent space.

**Tech Stack:** PyTorch (torch.amp.autocast, torch.backends.cudnn), ComfyUI node API

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `performance_opts.py` | **Create** | Node class + registration dicts |
| `nodes_sampler.py` | **Modify** | Accept `PERFORMANCEOPTS` input, apply autocast + cudnn_benchmark, pass opts downstream |
| `multitalk/multitalk_loop.py` | **Modify** | Skip redundant VAE encode, latent-space motion transfer |
| `__init__.py` | **Modify** | Register new module |

---

### Task 1: Create `performance_opts.py` node

**Files:**
- Create: `performance_opts.py`

- [ ] **Step 1: Create the node file with full implementation**

```python
# performance_opts.py
from .utils import log

class WanVideoPerformanceOpts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fp16_autocast": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Wrap sampling in fp16 autocast. Enables V100 Tensor Cores (125 TFLOPS fp16 vs 15.7 TFLOPS fp32). Safe with GGUF models."
                }),
                "cudnn_benchmark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable cuDNN benchmark mode. Autotuning for fixed tensor shapes in video generation. Set False if input sizes vary."
                }),
                "vae_skip_redundant_encode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "In InfiniteTalk mode, reuse conditioning latent from the main encode instead of running a separate VAE encode. Saves one full VAE encode per iteration window."
                }),
                "latent_motion_transfer": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: Transfer motion frames in latent space between iterations, skipping VAE decode+encode cycle. Only works when colormatch is disabled and output_path is empty. Can save ~30%% time per iteration but may affect quality."
                }),
            }
        }

    RETURN_TYPES = ("PERFORMANCEOPTS",)
    RETURN_NAMES = ("performance_opts",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Performance optimizations for older GPUs (V100, etc). Connect to WanVideoSampler."

    def process(self, fp16_autocast, cudnn_benchmark, vae_skip_redundant_encode, latent_motion_transfer):
        opts = {
            "fp16_autocast": fp16_autocast,
            "cudnn_benchmark": cudnn_benchmark,
            "vae_skip_redundant_encode": vae_skip_redundant_encode,
            "latent_motion_transfer": latent_motion_transfer,
        }
        if fp16_autocast:
            log.info("Performance: fp16 autocast enabled (V100 Tensor Core optimization)")
        if cudnn_benchmark:
            log.info("Performance: cuDNN benchmark mode enabled")
        if vae_skip_redundant_encode:
            log.info("Performance: VAE redundant encode skip enabled")
        if latent_motion_transfer:
            log.info("Performance: Latent-space motion transfer enabled (experimental)")
        return (opts,)


NODE_CLASS_MAPPINGS = {
    "WanVideoPerformanceOpts": WanVideoPerformanceOpts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoPerformanceOpts": "WanVideo Performance Opts",
}
```

- [ ] **Step 2: Commit**

```bash
git add performance_opts.py
git commit -m "feat: add WanVideoPerformanceOpts node for V100 optimization"
```

---

### Task 2: Register the new module in `__init__.py`

**Files:**
- Modify: `__init__.py:27-53` (OPTIONAL_MODULES list)

- [ ] **Step 1: Add the module to OPTIONAL_MODULES**

Add this line at the end of the OPTIONAL_MODULES list (before the closing `]`):

```python
    (".performance_opts", "PerformanceOpts"),
```

- [ ] **Step 2: Commit**

```bash
git add __init__.py
git commit -m "feat: register PerformanceOpts module in __init__"
```

---

### Task 3: Wire `performance_opts` input into `WanVideoSampler`

**Files:**
- Modify: `nodes_sampler.py:35-83` (INPUT_TYPES and process signature)
- Modify: `nodes_sampler.py:1175-1176` (autocast in predict_with_cfg)

- [ ] **Step 1: Add optional input to INPUT_TYPES**

In `WanVideoSampler.INPUT_TYPES`, add to the `"optional"` dict (after the `"add_noise_to_samples"` entry at line 71):

```python
                "performance_opts": ("PERFORMANCEOPTS", ),
```

- [ ] **Step 2: Add parameter to process() signature**

In the `process()` method signature at line 83, add `performance_opts=None` to the end of the parameter list:

```python
        ..., start_step=0, end_step=-1, add_noise_to_samples=False, performance_opts=None):
```

- [ ] **Step 3: Apply cuDNN benchmark at start of process()**

After line 85 (`if flowedit_args` check) and before line 86 (`patcher = model`), add:

```python
        # Performance optimizations
        perf_opts = performance_opts or {}
        cudnn_benchmark_prev = torch.backends.cudnn.benchmark
        if perf_opts.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
            log.info("cuDNN benchmark mode enabled for this sampling run")
```

- [ ] **Step 4: Extend autocast in predict_with_cfg**

Replace line 1175-1176:

```python
            autocast_enabled = ("fp8" in model["quantization"] and not transformer.patched_linear)
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype) if autocast_enabled else nullcontext():
```

With:

```python
            fp16_autocast = perf_opts.get("fp16_autocast", False)
            autocast_enabled = ("fp8" in model["quantization"] and not transformer.patched_linear) or fp16_autocast
            autocast_dtype = torch.float16 if fp16_autocast else dtype
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=autocast_dtype) if autocast_enabled else nullcontext():
```

- [ ] **Step 5: Restore cuDNN benchmark after sampling completes**

There are multiple return paths from `process()`:
1. `multitalk_loop` return at line 2045
2. framepack return at line 2187
3. wananimate return at line 2488
4. normal return at line 2632

For the normal return path, add cleanup before line 2632:

```python
        torch.backends.cudnn.benchmark = cudnn_benchmark_prev
```

For the multitalk path: the `multitalk_loop(**locals())` call at line 2045 passes `locals()` which will include `perf_opts` and `cudnn_benchmark_prev`. We handle the restore inside multitalk_loop (Task 4).

For the try/except block wrapping the main loop (line 1731-onwards), add a finally clause to ensure cleanup. Or simpler: add the restore right before each return. The key return paths are:

- Before line 2045: `return multitalk_loop(**locals())` - handled in multitalk_loop
- Before line 2187: `return {"video": gen_video_samples},` - add restore before
- Before line 2488: `return {"video": gen_video_samples...},` - add restore before
- Before line 2632: `return ({...},{...})` - add restore before

Add `torch.backends.cudnn.benchmark = cudnn_benchmark_prev` before each of these 3 non-multitalk return statements.

- [ ] **Step 6: Commit**

```bash
git add nodes_sampler.py
git commit -m "feat: wire performance_opts into WanVideoSampler with fp16 autocast and cuDNN benchmark"
```

---

### Task 4: Optimize VAE encode/decode in multitalk_loop

**Files:**
- Modify: `multitalk/multitalk_loop.py:25-46` (kwargs unpacking)
- Modify: `multitalk/multitalk_loop.py:274-278` (redundant VAE encode)
- Modify: `multitalk/multitalk_loop.py:461-526` (motion frame transfer)

- [ ] **Step 1: Accept performance_opts in kwargs**

At line 46 in the kwargs unpacking tuple, add `'performance_opts'` and `'cudnn_benchmark_prev'` to the keys list. The variable assignments are at lines 27-46:

Add to the end of the tuple at line 35 (before the closing `)`):
```python
        'performance_opts', 'cudnn_benchmark_prev'
    ))
```

And add the corresponding variable names in the tuple at line 27 (before the closing `)`):
```python
        ..., seed_g, gguf_reader, predict_func,
        performance_opts, cudnn_benchmark_prev
    ) = (kwargs.get(k) for k in (
```

Then after line 46, set defaults:
```python
    perf_opts = performance_opts or {}
    skip_redundant_encode = perf_opts.get("vae_skip_redundant_encode", False)
    latent_motion_transfer = perf_opts.get("latent_motion_transfer", False)
```

- [ ] **Step 2: Skip redundant VAE encode in infinitetalk mode**

Replace lines 274-278:

```python
            if mode == "infinitetalk":
                cond_ = cond_image if is_first_clip else cond_frame
                latent_motion_frames = vae.encode(cond_.to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]
            else:
                latent_motion_frames = y[:, :cur_motion_frames_latent_num] # C T H W
```

With:

```python
            if mode == "infinitetalk":
                if skip_redundant_encode:
                    # Reuse the conditioning latent already encoded in y
                    # y contains the encoded padding_frames_pixels_values where cond_ is the first frame(s)
                    latent_motion_frames = y[:, :cur_motion_frames_latent_num]
                    log.debug("Skipped redundant VAE encode for infinitetalk motion frames")
                else:
                    cond_ = cond_image if is_first_clip else cond_frame
                    latent_motion_frames = vae.encode(cond_.to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]
            else:
                latent_motion_frames = y[:, :cur_motion_frames_latent_num] # C T H W
```

- [ ] **Step 3: Implement latent-space motion transfer**

This optimization skips the full VAE decode→pixel extract→encode cycle for intermediate iterations. Instead, it:
1. Caches the diffusion output latent's last frames
2. Uses them directly as motion frame latents in the next iteration
3. Stores pixel-decoded latents in a list only at the end

After line 461 (`del noise, latent_motion_frames`), the current flow is:
```
decode → color match → save/append → update cond frames from pixels → delete
```

We change this to conditionally skip decode for intermediate iterations when latent_motion_transfer is enabled:

Replace lines 461-526 (from `del noise, latent_motion_frames` through `del videos, latent`):

```python
        del noise
        if offload:
            offload_transformer(transformer, remove_lora=False)
            offloaded = True
        if humo_image_cond is not None and humo_reference_count > 0:
            latent = latent[:,:-humo_reference_count]

        # Latent-space motion transfer: skip intermediate decode+encode when possible
        can_skip_decode = (
            latent_motion_transfer
            and not arrive_last_frame
            and not output_path
            and colormatch == "disabled"
            and mode in ("infinitetalk", "multitalk")
        )

        if can_skip_decode:
            # Cache motion frame latents directly from diffusion output
            cached_motion_latent = latent[:, -((motion_frame - 1) // 4 + 1):].detach().clone().cpu()
            # Still need to store latent for final stitching
            gen_video_list.append(("latent", latent.detach().clone().cpu(), is_first_clip, cur_motion_frames_num))
            log.debug(f"Latent motion transfer: skipped VAE decode for iteration {iteration_count}")
            del latent_motion_frames
        else:
            del latent_motion_frames
            cached_motion_latent = None

            vae.to(device)
            videos = vae.decode(latent.unsqueeze(0).to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False)[0].cpu()
            vae.to(offload_device)

            sampling_pbar.close()

            # crop drop_frames from end if enabled
            if mode == "skyreelsv3" and drop_frames > 0 and not arrive_last_frame:
                videos = videos[:, :-drop_frames]

            # optional color correction
            if colormatch != "disabled":
                if colormatch == "reinhard_torch":
                    videos = match_and_blend_colors(videos, original_color_reference, 1.0)
                else:
                    videos = videos.permute(1, 2, 3, 0).float().numpy()
                    from color_matcher import ColorMatcher
                    cm = ColorMatcher()
                    cm_result_list = []
                    for img in videos:
                        if mode == "infinitetalk":
                            cm_result = cm.transfer(src=img, ref=cond_image[0].permute(1, 2, 3, 0).squeeze(0).cpu().float().numpy(), method=colormatch)
                        else:
                            cm_result = cm.transfer(src=img, ref=original_images[0].permute(1, 2, 3, 0).squeeze(0).cpu().float().numpy(), method=colormatch)
                        cm_result_list.append(torch.from_numpy(cm_result).to(vae.dtype))
                    videos = torch.stack(cm_result_list, dim=0).permute(3, 0, 1, 2)

            # save generated samples
            if output_path:
                video_np = videos.clamp(-1.0, 1.0).add(1.0).div(2.0).mul(255).cpu().float().numpy().transpose(1, 2, 3, 0).astype('uint8')
                num_frames_to_save = video_np.shape[0] if is_first_clip else video_np.shape[0] - cur_motion_frames_num
                log.info(f"Saving {num_frames_to_save} generated frames to {output_path}")
                start_idx = 0 if is_first_clip else cur_motion_frames_num
                for i in range(start_idx, video_np.shape[0]):
                    im = Image.fromarray(video_np[i])
                    im.save(os.path.join(output_path, f"frame_{img_counter:05d}.png"))
                    img_counter += 1
            else:
                gen_video_list.append(videos if is_first_clip else videos[:, cur_motion_frames_num:])

        current_condframe_index += 1
        iteration_count += 1

        # decide whether is done
        if arrive_last_frame:
            break

        # update next condition frames
        is_first_clip = False
        cur_motion_frames_num = motion_frame

        if can_skip_decode:
            # Use cached latent directly - no pixel-space round-trip
            pass  # cached_motion_latent is set above, will be used as latent_motion_frames in next iteration
        else:
            cond_ = videos[:, -cur_motion_frames_num:].unsqueeze(0)
            if mode == "infinitetalk":
                cond_frame = cond_
            else:
                cond_image = cond_

        del latent
        if not can_skip_decode:
            del videos
```

- [ ] **Step 4: Use cached_motion_latent in next iteration's VAE encode section**

At line 274 (the infinitetalk encode section), modify to also check for cached_motion_latent:

Replace the skip_redundant_encode block from Step 2 with:

```python
            if mode == "infinitetalk":
                if cached_motion_latent is not None:
                    # Use diffusion output latent directly (latent-space motion transfer)
                    latent_motion_frames = cached_motion_latent.to(device, dtype)
                    cached_motion_latent = None
                    log.debug("Using cached diffusion latent as motion frames")
                elif skip_redundant_encode:
                    latent_motion_frames = y[:, :cur_motion_frames_latent_num]
                    log.debug("Skipped redundant VAE encode for infinitetalk motion frames")
                else:
                    cond_ = cond_image if is_first_clip else cond_frame
                    latent_motion_frames = vae.encode(cond_.to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]
            else:
                if cached_motion_latent is not None:
                    latent_motion_frames = cached_motion_latent.to(device, dtype)
                    cached_motion_latent = None
                    log.debug("Using cached diffusion latent as motion frames")
                else:
                    latent_motion_frames = y[:, :cur_motion_frames_latent_num]
```

Note: When `latent_motion_transfer` is active and we're using `cached_motion_latent`, we still need to run the main VAE encode (line 272) for `y` because `y` is used as the image conditioning signal (`y = torch.cat([msk, y])` at line 296). The cached latent only replaces `latent_motion_frames`.

BUT - when using `cached_motion_latent`, we can skip the main encode too IF we construct `y` from the cached latent + zero padding. This is complex and risky, so we only optimize the `latent_motion_frames` part.

**IMPORTANT**: Initialize `cached_motion_latent = None` before the while loop (around line 93, after `is_first_clip = True`):

```python
    cached_motion_latent = None
```

- [ ] **Step 5: Handle deferred decode for latent entries in gen_video_list**

Before the final output at line 556 (`if not output_path:`), we need to decode any latent entries that were deferred:

Replace lines 556-559:

```python
    if not output_path:
        # Decode any deferred latents from latent_motion_transfer
        decoded_list = []
        for entry in gen_video_list:
            if isinstance(entry, tuple) and entry[0] == "latent":
                _, lat, was_first, motion_num = entry
                vae.to(device)
                decoded = vae.decode(lat.unsqueeze(0).to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False)[0].cpu()
                vae.to(offload_device)
                decoded_list.append(decoded if was_first else decoded[:, motion_num:])
            else:
                decoded_list.append(entry)
        gen_video_samples = torch.cat(decoded_list, dim=1)
    else:
        gen_video_samples = torch.zeros(3, 1, 64, 64) # dummy output
```

- [ ] **Step 6: Restore cuDNN benchmark before returning**

Before the return at line 569, add:

```python
    if cudnn_benchmark_prev is not None:
        torch.backends.cudnn.benchmark = cudnn_benchmark_prev
```

- [ ] **Step 7: Commit**

```bash
git add multitalk/multitalk_loop.py
git commit -m "feat: add VAE encode skip and latent motion transfer optimizations to multitalk loop"
```

---

### Task 5: Final integration verification

- [ ] **Step 1: Verify all imports are correct**

Check that `performance_opts.py` imports work:
- `from .utils import log` - already used throughout the codebase

Check that `nodes_sampler.py` doesn't need new imports:
- `torch.backends.cudnn` is part of torch (already imported)
- `torch.autocast` already used at line 1176
- `perf_opts` is just a dict, no new types needed

Check that `multitalk_loop.py` doesn't need new imports:
- `log` already imported at line 9

- [ ] **Step 2: Verify the `locals()` passthrough works**

When `multitalk_loop(**locals())` is called at line 2045 of `nodes_sampler.py`, `locals()` includes all local variables in `process()`. Since we added `performance_opts` as a parameter and `perf_opts`/`cudnn_benchmark_prev` as local variables, they will all be passed through. The multitalk_loop kwargs unpacking must include these keys.

Verify the kwargs tuple in multitalk_loop.py includes:
- `'performance_opts'` (the original param)
- `'cudnn_benchmark_prev'` (the saved state)
- `'perf_opts'` does NOT need to be passed - it's re-derived from `performance_opts` inside multitalk_loop

- [ ] **Step 3: Commit all files together**

```bash
git add performance_opts.py __init__.py nodes_sampler.py multitalk/multitalk_loop.py
git commit -m "feat: complete V100 performance optimization node integration"
```

---

## Expected Performance Impact

| Optimization | Condition | Estimated Speedup |
|---|---|---|
| fp16 autocast | V100 with bf16/fp32 base_dtype | ~2x on matmul-bound ops |
| cuDNN benchmark | Fixed tensor shapes (video gen) | 5-15% on conv-heavy ops (VAE) |
| Skip redundant encode | InfiniteTalk mode | Saves 1 full VAE encode per iteration (~5-10s each) |
| Latent motion transfer | No colormatch, no output_path | Saves 1 decode + 1 encode per iteration (~10-20s each) |

For the InfiniteTalk workflow with 5 iterations: combined savings of ~100-200s from VAE optimizations alone, plus ~2x from fp16 autocast on the transformer forward passes.