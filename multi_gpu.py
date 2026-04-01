from .utils import log
from .multi_gpu_utils import get_available_gpu_count


class WanVideoMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_gpus": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Number of GPUs to use. 0 = auto-detect all available GPUs. Minimum 2 for multi-GPU strategies to activate."
                }),
                "strategy": (["auto", "layer_split", "cfg_parallel"], {
                    "default": "auto",
                    "tooltip": (
                        "Multi-GPU strategy. "
                        "'auto' selects best strategy based on GPU count (2 GPUs → layer_split, 3+ → cfg_parallel). "
                        "'layer_split' distributes 40 transformer blocks across all GPUs (eliminates block swap overhead). "
                        "'cfg_parallel' splits GPUs into 2 groups: cond on first half, uncond on second half. "
                        "Each group uses layer_split internally. Gives ~2x from parallel CFG + layer_split benefits. "
                        "Works with any GPU count >= 2."
                    )
                }),
                "primary_gpu": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 15,
                    "step": 1,
                    "tooltip": "Index of the primary GPU (where embeddings and VAE run). Usually 0."
                }),
            }
        }

    RETURN_TYPES = ("MULTIGPUOPTS",)
    RETURN_NAMES = ("multi_gpu_opts",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = (
        "Multi-GPU configuration for WanVideo. "
        "layer_split: distribute blocks across GPUs (best for 2 GPUs). "
        "cfg_parallel: run cond/uncond passes on separate GPU groups with layer_split within each (best for 3+ GPUs). "
        "Connect to WanVideoSampler."
    )

    def process(self, num_gpus, strategy, primary_gpu):
        available = get_available_gpu_count()

        # Auto strategy selection based on GPU count
        if strategy == "auto":
            if available >= 3:
                strategy = "cfg_parallel"
            else:
                strategy = "layer_split"

        opts = {
            "num_gpus": num_gpus,
            "strategy": strategy,
            "primary_gpu": primary_gpu,
        }

        log.info(f"MultiGPU: strategy={strategy}, num_gpus={num_gpus} (available={available}), primary_gpu={primary_gpu}")
        return (opts,)


NODE_CLASS_MAPPINGS = {
    "WanVideoMultiGPU": WanVideoMultiGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoMultiGPU": "WanVideo Multi-GPU",
}
