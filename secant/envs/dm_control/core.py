"""
Adapted from https://github.com/denisyarats/dmc2gym/tree/master/dmc2gym
Wrapper around dm_control
"""
import time
from typing import Union, Tuple, Optional, List
from secant.wrappers import FrameStack, TimeLimit
from ._patch import patch_dm_headless_egl_display, get_seed
from .adapter import DMControlAdapter
from .pad_env import ColorWrapper, GreenScreen
from .settings import *


__all__ = ["make_dmc", "ALL_TASKS", "ALL_TASK_TUPLES"]


def _make_dmc(
    task: Union[str, Tuple[str, str]],
    seed: Optional[int] = None,
    visualize_reward: bool = False,
    from_pixels: bool = True,
    image_height: int = 84,
    image_width: int = 84,
    camera_id: int = 0,
    action_repeat: Union[int, "pad", "dreamer", "planet"] = 4,
    frame_stack: Optional[int] = 3,
    episode_length: int = 1000,
    environment_kwargs: Optional[dict] = None,
    time_limit: Optional[int] = None,
    channels_first: bool = True,
    device_id: Optional[int] = None,
    background: str = "original",
    enable_frame_buffer: bool = False,
):
    """
    Don't forget to set the MUJOCO_GL env var. "egl" is recommended. From dm_control page:
        By default, dm_control will attempt to use GLFW first, then EGL, then OSMesa.
        You can set MUJOCO_GL= environment variable to "glfw", "egl", or "osmesa".
        When rendering with EGL, you can also specify which GPU to use for rendering
        by setting the environment variable EGL_DEVICE_ID= to the target GPU ID.

    Args:
        task: full name "cheetah_run", or a tuple (domain, task): ("cheetah", "run")
        background: option to add color/video background to the scene. Need to be
            one of "train", "color_easy", "color_hard", or "video[0-9]".
            The idea and code comes from the
            paper: "Self-Supervised Policy Adaptation during Deployment". Check
            https://nicklashansen.github.io/PAD/ for more details.
        action_repeat: if str, can be one of "pad", "dreamer" or "planet" protocol
    """
    patch_dm_headless_egl_display(device_id)

    assert isinstance(task, (str, tuple)), (
        "either specify full_name or a tuple (domain_name, task_name). "
        'Example: full_name "cheetah_run"  or ("cheetah", "run")'
    )

    domain_name, task_name = dmc_name_to_domain_task(task)

    if isinstance(action_repeat, str):
        action_repeat = dmc_action_repeat(task, protocol=action_repeat, default=4)

    if from_pixels:
        assert (
            not visualize_reward
        ), "cannot use visualize reward when learning from pixels"

    task_kwargs = {}
    seed = get_seed(seed, handle_invalid_seed="system")
    task_kwargs["random"] = seed
    if time_limit is not None:
        task_kwargs["time_limit"] = time_limit

    assert (
        background
        in [
            "original",
            "color_easy",
            "color_hard",
        ]
        or background.startswith("video")
    ), (
        f"Background mode must be 'original', 'color_easy', 'color_hard', or 'video[0-9]', "
        f"received {background}"
    )

    env = DMControlAdapter(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs,
        visualize_reward=visualize_reward,
        from_pixels=from_pixels,
        height=image_height,
        width=image_width,
        camera_id=camera_id,
        action_repeat=action_repeat,
        channels_first=channels_first,
        enable_frame_buffer=enable_frame_buffer,
    )

    # `done` is always False from DMControlWrapper, we need to limit it manually
    env = TimeLimit(
        env, max_episode_steps=(episode_length + action_repeat - 1) // action_repeat
    )

    if background.startswith("video"):
        env = GreenScreen(env, background)

    if frame_stack and frame_stack > 0:
        env = FrameStack(env, k=frame_stack, mode="concat")

    if background != "original":
        env = ColorWrapper(env, background)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env


def make_dmc(
    task: Union[str, Tuple[str, str]],
    seed: Optional[int] = None,
    image_height: int = 84,
    image_width: int = 84,
    frame_stack: Optional[int] = 3,
    action_repeat: Union[int, str] = 4,
    device_id: Optional[int] = None,
    background: str = "original",
    enable_frame_buffer: bool = False,
):
    """
    Add preprocessing wrappers to reproduce the settings in prior works
    Training on visual inputs
    """
    domain_name, task_name = dmc_name_to_domain_task(task)
    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == "quadruped" else 0

    env = _make_dmc(
        task=task,
        seed=seed,
        image_height=image_height,
        image_width=image_width,
        visualize_reward=False,
        from_pixels=True,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        episode_length=1000,
        camera_id=camera_id,
        device_id=device_id,
        background=background,
        enable_frame_buffer=enable_frame_buffer,
    )

    return env
