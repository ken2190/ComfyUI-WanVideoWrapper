import torch
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
                    "tooltip": "EXPERIMENTAL: Transfer motion frames in latent space between iterations, skipping VAE decode+encode cycle. Only works when colormatch is disabled and output_path is empty. Can save ~30% time per iteration but may affect quality."
                }),
                "cache_device_tensors": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache tensors on GPU between sampling steps to avoid redundant CPU->GPU transfers. Saves 50+ .to(device) calls per step. Disable only if VRAM is extremely tight."
                }),
                "cache_audio_null": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Pre-compute and cache the null audio tensor used for audio CFG guidance. Avoids recreating torch.zeros_like() every step."
                }),
                "async_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: Use CUDA streams for non-blocking VAE model transfers in InfiniteTalk mode. Overlaps vae.to(device) with noise preparation. May not work on all systems."
                }),
                "block_swap_prefetch": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4,
                    "step": 1,
                    "tooltip": "Number of transformer blocks to prefetch ahead during block swap (single-GPU only). 0=disabled, 1-2 recommended for V100. Uses CUDA streams to overlap PCIe transfer with compute. Ignored when multi-GPU is active."
                }),
            }
        }

    RETURN_TYPES = ("PERFORMANCEOPTS",)
    RETURN_NAMES = ("performance_opts",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Performance optimizations for older GPUs (V100, etc). Connect to WanVideoSampler."

    def process(self, fp16_autocast, cudnn_benchmark, vae_skip_redundant_encode,
                latent_motion_transfer, cache_device_tensors, cache_audio_null,
                async_vae, block_swap_prefetch):
        # Detect GPU capability for logging
        gpu_name = "unknown"
        compute_cap = (0, 0)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)

        opts = {
            "fp16_autocast": fp16_autocast,
            "cudnn_benchmark": cudnn_benchmark,
            "vae_skip_redundant_encode": vae_skip_redundant_encode,
            "latent_motion_transfer": latent_motion_transfer,
            "cache_device_tensors": cache_device_tensors,
            "cache_audio_null": cache_audio_null,
            "async_vae": async_vae,
            "block_swap_prefetch": block_swap_prefetch,
        }

        enabled = [k for k, v in opts.items() if v and k != "block_swap_prefetch"]
        if block_swap_prefetch > 0:
            enabled.append(f"block_swap_prefetch={block_swap_prefetch}")

        log.info(f"Performance opts ({gpu_name}, sm_{compute_cap[0]}{compute_cap[1]}): {', '.join(enabled) if enabled else 'none'}")

        if compute_cap < (8, 0):
            if fp16_autocast:
                log.info("  fp16 autocast: V100 Tensor Cores enabled (125 TFLOPS fp16)")
            log.info(f"  Note: Flash Attention requires sm_80+, using SDPA memory-efficient backend on sm_{compute_cap[0]}{compute_cap[1]}")

        return (opts,)


NODE_CLASS_MAPPINGS = {
    "WanVideoPerformanceOpts": WanVideoPerformanceOpts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoPerformanceOpts": "WanVideo Performance Opts",
}
