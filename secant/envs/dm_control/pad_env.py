"""
Env used in PAD. Code adapted from its github repo
    paper: Self-Supervised Policy Adaptation during Deployment
            (https://arxiv.org/abs/2007.04309)
    github: https://github.com/nicklashansen/policy-adaptation-during-deployment
"""
import numpy as np
from numpy.random import randint
import gym
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import importlib.resources
from dm_control.suite import common
from secant.wrappers import FrameStack
from secant.envs.dm_control.adapter import DMControlAdapter

__all__ = ["ColorWrapper", "GreenScreen"]


def _resource_file_path(fname) -> str:
    with importlib.resources.path("secant.envs.dm_control.pad_data", fname) as p:
        return str(p)


class ColorWrapper(gym.Wrapper):
    """Wrapper for the color experiments"""

    def __init__(self, env, mode):
        assert isinstance(env, FrameStack), "color/video env must be FrameStack first"
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self._mode = mode
        self.time_step = 0
        if "color" in self._mode:
            self._load_colors()

    def reset(self):
        self.time_step = 0
        if "color" in self._mode:
            self.randomize()
        if "video" in self._mode:
            # apply greenscreen
            self.reload_physics(
                {
                    "skybox_rgb": [0.2, 0.8, 0.2],
                    "skybox_rgb2": [0.2, 0.8, 0.2],
                    "skybox_markrgb": [0.2, 0.8, 0.2],
                }
            )
        return self.env.reset()

    def step(self, action):
        self.time_step += 1
        return self.env.step(action)

    def randomize(self):
        assert (
            "color" in self._mode
        ), f"can only randomize in color mode, received {self._mode}"
        self.reload_physics(self.get_random_color())

    def _load_colors(self):
        assert self._mode in {"color_easy", "color_hard"}
        self._colors = torch.load(_resource_file_path(f"{self._mode}.pt"))

    def get_random_color(self):
        assert len(self._colors) >= 100, "env must include at least 100 colors"
        return self._colors[randint(len(self._colors))]

    def reload_physics(self, setting_kwargs=None, state=None):
        domain_name = self._get_dmc_wrapper()._domain_name
        if setting_kwargs is None:
            setting_kwargs = {}
        if state is None:
            state = self._get_state()
        self._reload_physics(
            *common.settings.get_model_and_assets_from_setting_kwargs(
                domain_name + ".xml", setting_kwargs
            )
        )
        self._set_state(state)

    def get_state(self):
        return self._get_state()

    def set_state(self, state):
        self._set_state(state)

    def _get_dmc_wrapper(self):
        _env = self.env
        while not isinstance(_env, DMControlAdapter) and hasattr(_env, "env"):
            _env = _env.env
        assert isinstance(_env, DMControlAdapter), "environment is not dmc2gym-wrapped"

        return _env

    def _reload_physics(self, xml_string, assets=None):
        _env = self.env
        while not hasattr(_env, "_physics") and hasattr(_env, "env"):
            _env = _env.env
        assert hasattr(_env, "_physics"), "environment does not have physics attribute"
        _env.physics.reload_from_xml_string(xml_string, assets=assets)

    def _get_physics(self):
        _env = self.env
        while not hasattr(_env, "_physics") and hasattr(_env, "env"):
            _env = _env.env
        assert hasattr(_env, "_physics"), "environment does not have physics attribute"

        return _env._physics

    def _get_state(self):
        return self._get_physics().get_state()

    def _set_state(self, state):
        self._get_physics().set_state(state)


def rgb_to_hsv(r, g, b):
    """Convert RGB color to HSV color"""
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc - minc) / maxc
    rc = (maxc - r) / (maxc - minc)
    gc = (maxc - g) / (maxc - minc)
    bc = (maxc - b) / (maxc - minc)
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h / 6.0) % 1.0
    return h, s, v


def do_green_screen(x, bg):
    """Removes green background from observation and replaces with bg; not optimized for speed"""
    assert isinstance(x, np.ndarray) and isinstance(
        bg, np.ndarray
    ), "inputs must be numpy arrays"
    assert x.dtype == np.uint8 and bg.dtype == np.uint8, "inputs must be uint8 arrays"

    # Get image sizes
    x_h, x_w = x.shape[1:]

    # Convert to RGBA images
    im = TF.to_pil_image(torch.ByteTensor(x))
    im = im.convert("RGBA")
    pix = im.load()
    bg = TF.to_pil_image(torch.ByteTensor(bg))
    bg = bg.convert("RGBA")
    bg = bg.load()

    # Replace pixels
    for x in range(x_w):
        for y in range(x_h):
            r, g, b, a = pix[x, y]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

            min_h, min_s, min_v = (100, 80, 70)
            max_h, max_s, max_v = (185, 255, 255)
            if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
                pix[x, y] = bg[x, y]

    x = np.moveaxis(np.array(im).astype(np.uint8), -1, 0)[:3]

    return x


class GreenScreen(gym.Wrapper):
    """Green screen for video experiments"""

    def __init__(self, env, mode):
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        if "video" in mode:
            self._video = mode
            if not self._video.endswith(".mp4"):
                self._video += ".mp4"
            self._video = _resource_file_path(self._video)
            self._data = self._load_video(self._video)
        else:
            self._video = None
        self._max_episode_steps = env._max_episode_steps

    def _load_video(self, video):
        """Load video from provided filepath and return as numpy array"""
        cap = cv2.VideoCapture(video)
        assert (
            cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100
        ), "width must be at least 100 pixels"
        assert (
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100
        ), "height must be at least 100 pixels"
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty(
            (
                n,
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                3,
            ),
            np.dtype("uint8"),
        )
        i, ret = 0, True
        while i < n and ret:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf[i] = frame
            i += 1
        cap.release()
        return np.moveaxis(buf, -1, 1)

    def reset(self):
        self._current_frame = 0
        return self._greenscreen(self.env.reset())

    def step(self, action):
        self._current_frame += 1
        obs, reward, done, info = self.env.step(action)
        return self._greenscreen(obs), reward, done, info

    def _interpolate_bg(self, bg, size: tuple):
        """Interpolate background to size of observation"""
        bg = torch.from_numpy(bg).float().unsqueeze(0) / 255
        bg = F.interpolate(bg, size=size, mode="bilinear", align_corners=False)
        return (bg * 255).byte().squeeze(0).numpy()

    def _greenscreen(self, obs):
        """Applies greenscreen if video is selected, otherwise does nothing"""
        if self._video:
            bg = self._data[self._current_frame % len(self._data)]  # select frame
            bg = self._interpolate_bg(bg, obs.shape[1:])  # scale bg to observation size
            return do_green_screen(obs, bg)  # apply greenscreen
        return obs

    def apply_to(self, obs):
        """Applies greenscreen mode of object to observation"""
        obs = obs.copy()
        channels_last = obs.shape[-1] == 3
        if channels_last:
            obs = torch.from_numpy(obs).permute(2, 0, 1).numpy()
        obs = self._greenscreen(obs)
        if channels_last:
            obs = torch.from_numpy(obs).permute(1, 2, 0).numpy()
        return obs
