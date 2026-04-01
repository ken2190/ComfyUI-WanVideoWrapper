import torch
from .utils import log


def get_available_gpu_count():
    """Return number of CUDA GPUs available, 0 if none."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def validate_gpu_config(num_gpus, primary_gpu=0):
    """Validate GPU configuration and return list of torch.device objects.

    Args:
        num_gpus: Number of GPUs to use. 0 = auto-detect all available.
        primary_gpu: Index of the primary GPU (where embeddings are computed).

    Returns:
        List of torch.device objects to distribute work across.
    """
    available = get_available_gpu_count()
    if available == 0:
        log.warning("MultiGPU: No CUDA GPUs available, falling back to CPU")
        return [torch.device("cpu")]

    target = available if num_gpus <= 0 else min(num_gpus, available)
    if target < 2:
        log.info("MultiGPU: Only 1 GPU available/requested, multi-GPU disabled")
        return [torch.device(f"cuda:{primary_gpu}")]

    if primary_gpu >= available:
        log.warning(f"MultiGPU: primary_gpu={primary_gpu} exceeds available GPUs ({available}), using 0")
        primary_gpu = 0

    # Build device list with primary GPU first
    devices = [torch.device(f"cuda:{primary_gpu}")]
    for i in range(available):
        if i != primary_gpu and len(devices) < target:
            devices.append(torch.device(f"cuda:{i}"))

    log.info(f"MultiGPU: Using {len(devices)} GPUs: {[str(d) for d in devices]}")
    return devices


def split_devices_for_cfg(devices):
    """Split device list into two groups for CFG parallel.

    First group handles conditional (positive) pass, second handles
    unconditional (negative) pass. Each group can use layer_split internally.

    Works with any number of GPUs >= 2:
      2 GPUs → [cuda:0] + [cuda:1]
      3 GPUs → [cuda:0] + [cuda:1, cuda:2]
      4 GPUs → [cuda:0, cuda:1] + [cuda:2, cuda:3]
      6 GPUs → [cuda:0, cuda:1, cuda:2] + [cuda:3, cuda:4, cuda:5]

    Returns:
        Tuple of (cond_devices, uncond_devices).
    """
    n = len(devices)
    if n < 2:
        return devices, devices
    mid = n // 2
    return devices[:mid], devices[mid:]
