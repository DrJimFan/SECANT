"""
Patch dm_control OpenGL initialization to use GPU ID other than the first available.
The intialization function reads from environment variable:

    EGL_DEVICE_ID

Useful for distributed environments to use multiple GPUs evenly.

Source code: https://github.com/deepmind/dm_control/blob/master/dm_control/_render/pyopengl/egl_renderer.py#L47

Issue and workaround: https://github.com/deepmind/dm_control/issues/118
"""
__all__ = ["patch_dm_headless_egl_display", "EGL_ENV_VAR"]


import os
import atexit
import time
from typing import Optional, List, Union
from typing_extensions import Literal
import torch
import dm_control._render.pyopengl.egl_renderer
from dm_control._render.pyopengl import egl_ext as EGL


EGL_ENV_VAR = "EGL_DEVICE_ID"


def _create_initialized_headless_egl_display(device_id: Optional[int] = None):
    """
    Creates an initialized EGL display directly on a device.

    There are three ways:
    - pass an explicit GPU ID as arg to `patch_dm_headless_egl_display`
    - set `EGL_DEVICE_ID` environment variable to an int
    - if `EGL_DEVICE_ID` is undefined, use the first device in CUDA_VISIBLE_DEVICES
    """
    from OpenGL import error

    if device_id is not None:
        device_id = get_physical_device(device_id)
    elif os.environ.get(EGL_ENV_VAR, None) is not None:
        device_id = int(os.environ[EGL_ENV_VAR])
    else:
        device_id = get_physical_device(0, strict=False)

    devices = EGL.eglQueryDevicesEXT()
    if device_id is not None:
        try:
            devices = [devices[device_id]]
        except IndexError:
            raise IndexError(
                f"Your specified device ID {device_id} is out of range, you only have {len(devices)} device(s) available."
            )

    # below is code copied from dm_control
    for device in devices:
        display = EGL.eglGetPlatformDisplayEXT(
            EGL.EGL_PLATFORM_DEVICE_EXT, device, None
        )
        if display != EGL.EGL_NO_DISPLAY and EGL.eglGetError() == EGL.EGL_SUCCESS:
            # `eglInitialize` may or may not raise an exception on failure depending
            # on how PyOpenGL is configured. We therefore catch a `GLError` and also
            # manually check the output of `eglGetError()` here.
            try:
                initialized = EGL.eglInitialize(display, None, None)
            except error.GLError:
                pass
            else:
                if initialized == EGL.EGL_TRUE and EGL.eglGetError() == EGL.EGL_SUCCESS:
                    return display
    return EGL.EGL_NO_DISPLAY


def patch_dm_headless_egl_display(device_id: Optional[int] = None):
    EGL_DISPLAY = _create_initialized_headless_egl_display(device_id)

    if EGL_DISPLAY == EGL.EGL_NO_DISPLAY:
        raise RuntimeError("Cannot initialize a headless EGL display.")
    atexit.register(EGL.eglTerminate, EGL_DISPLAY)

    # monkey patch before DM suite.load() call
    dm_control._render.pyopengl.egl_renderer.EGL_DISPLAY = EGL_DISPLAY


def get_physical_device(gpu_id: int, strict: bool = True) -> Optional[int]:
    """
    Will retrieve the actual device ID from CUDA_VISIBLE_DEVICES.
    E.g. if CUDA_VISIBLE_DEVICES="4,5,6,7" and gpu_id=2, then the physical ID is 6
    """
    visible_ids = get_cuda_visible_devices()
    if gpu_id >= len(visible_ids):
        if strict:
            raise IndexError(
                f"gpu_id {gpu_id} is out of range of "
                f"len(CUDA_VISIBLE_DEVICES) == {len(visible_ids)}: {visible_ids}"
            )
        else:
            return None
    else:
        return visible_ids[gpu_id]


def get_cuda_visible_devices() -> List[int]:
    """
    parse CUDA_VISIBLE_DEVICES
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return list(range(torch.cuda.device_count()))
    device_str = os.environ["CUDA_VISIBLE_DEVICES"].strip(" ,")
    if not device_str:
        return []
    else:
        return [int(g) for g in device_str.split(",")]


def get_seed(
    seed: Union[int, str, None],
    handle_invalid_seed: Literal["none", "system", "raise"] = "none",
) -> Optional[int]:
    """
    Args:
      seed:
        "system": use scrambled int based on system time
        None or int < 0: invalid seed values, see `handle_invalid_seed`
        int >= 0: returns seed
      handle_invalid_seed: None or int < 0
        - "none": returns None
        - "system": returns scrambled int based on system time
        - "raise": raise Exception
    """
    handle_invalid_seed = handle_invalid_seed.lower()
    assert handle_invalid_seed in ["none", "system", "raise"]
    if isinstance(seed, str):
        assert seed in ["system"]
        invalid = False
    else:
        assert seed is None or isinstance(seed, int)
        invalid = seed is None or seed < 0

    if seed == "system" or invalid and handle_invalid_seed == "system":
        # https://stackoverflow.com/questions/27276135/python-random-system-time-seed
        t = int(time.time() * 100000)
        return (
            ((t & 0xFF000000) >> 24)
            + ((t & 0x00FF0000) >> 8)
            + ((t & 0x0000FF00) << 8)
            + ((t & 0x000000FF) << 24)
        )
    elif invalid:
        if handle_invalid_seed == "none":
            return None
        elif handle_invalid_seed == "raise":
            raise ValueError(
                f"Invalid random seed: {seed}, "
                f'must be a non-negative integer or "system"'
            )
        else:
            raise NotImplementedError
    else:
        return seed
