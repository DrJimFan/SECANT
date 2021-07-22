import gym
import carla
import cv2

import queue
import random
import math
import weakref
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from .suite import load_carla_env
from .utils import (
    get_all_vehicles,
    carla_rgb_to_np,
    carla_depth_to_np,
    carla_seg_to_np,
    get_spectator_transform,
    convert_action,
    get_obs_space,
    get_stacked_obs,
)
from secant.utils import AverageMeter
from secant.wrappers import VideoRecorder

ALL_MODALITIES = [
    "rgb",
    "depth",
    "semantic_segmentation",
    "collision",
    "lane_invasion",
    "obstacle_detection",
    "gnss",
    "accelerometer",
    "compass",
    "gyroscope",
    "lidar",
    "radar",
]


class CarlaAdapter(gym.core.Env):
    def __init__(
        self,
        map_name,
        client=None,
        host="localhost",
        client_port=2000,
        npc_v_port=8000,
        use_tm=False,
        sim_delta_seconds=0.05,
        enable_video=False,
        video_settings=None,
        ego_vehicle="tesla.model3",
        ego_vehicle_color=None,
        modalities=None,
        n_cam=5,
        stack_cams=False,
        fov=60.0,
        terminate_on_collision=None,
        collision_penalty=1e-4,
        steer_penalty=1.0,
        delta_distance=0.1,
        delta_t=0.05,
        height=84,
        width=84,
        action_repeat=1,
        frame_stack=None,
        channels_first=True,
        **env_kwargs,
    ):
        """
        stack_cams: let's say n_cams=5 x [3, 84, 84] images
            if stack_cams, [15, 84, 84]. Otherwise [3, 84, 420]  (panaramic)
        """
        modalities = modalities or []
        assert set(modalities).issubset(
            set(ALL_MODALITIES)
        ), f"please specify valid modalities from {ALL_MODALITIES}"
        self._modalities = modalities

        assert (
            "seed" in env_kwargs
        ), "please specify a random seed for deterministic behavior"
        self._seed = env_kwargs.get("seed")

        assert (
            n_cam > 0
        ), f"please provide a valid value for the number of camera, provided {n_cam}"
        self._n_cam = n_cam
        self._stack_cams = stack_cams

        assert fov > 0, f"please provide valid FoV values, provided {fov}"
        self._fov = fov
        self._height = height
        self._width = width

        assert (
            terminate_on_collision is None or terminate_on_collision >= 0.0
        ), f"please provide a valid terminate_on_collision, provided {terminate_on_collision}"
        self._terminate_on_collision = terminate_on_collision

        assert (
            delta_distance >= 0.0
        ), f"please provide a valid value for unit_direction_vec_delta_distance, provided {delta_distance}"
        self._delta_distance = delta_distance

        self._ego = ego_vehicle
        if ego_vehicle_color is None:
            self._ego_vehicle_color = None
        else:
            self._ego_vehicle_color = (
                f"{ego_vehicle_color[0]},{ego_vehicle_color[1]},{ego_vehicle_color[2]}"
            )
        self._player = None

        self._all_sensors = list()

        self._rgb_queue_list = list()
        self._rgb_cam_list = list()
        self._depth_queue_list = list()
        self._depth_cam_list = list()
        self._seg_queue_list = list()
        self._seg_cam_list = list()
        self._render_cam = None
        self._col_sensor = None
        self._inv_sensor = None
        self._obst_sensor = None
        self._gnss_sensor = None
        self._imu_sensor = None
        self._lidar_sensor = None
        self._radar_sensor = None

        self.world = None

        self._impulse_intensity = 0.0
        self._max_impulse_intensity = 0.0
        self._prev_col_frame = 0
        self._curr_col_frame = 0
        self._curr_col_intensity = 0.0

        self._prev_inv_frame = 0
        self._curr_inv_frame = 0
        self._curr_invasion = None

        self._prev_obst_frame = 0
        self._curr_obst_frame = 0
        self._curr_obst_distance = 0.0

        self._gnss_meas = [0.0, 0.0, 0.0]  # [altitude, latitude, longitude]
        self._accelerometer = [0.0, 0.0, 0.0]  # acceleration along [x, y, z] axis
        self._compass = 0.0
        self._gyroscope = [0.0, 0.0, 0.0]  # angular velocity along [x, y, z] axis
        self._lidar_meas = {"horizontal_angle": 0.0, "raw_data": None}
        self._radar_meas = [[], [], [], []]  # [altitude, azimuth, depth, velocity]

        self._env_kwargs = env_kwargs
        self._delta_t = delta_t

        self._collision_penalty = collision_penalty
        self._steer_penalty = steer_penalty

        self._action_repeat = action_repeat
        self._channels_first = channels_first
        self._k = frame_stack
        self._frames = deque([], maxlen=self._k)

        self._action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._n_time_steps = 0
        self._rgb_img_list = list()
        self._depth_img_list = list()
        self._seg_img_list = list()
        self._render_img = None
        self._control = carla.VehicleControl()

        # metrics
        self._progression = 0.0

        self._avg_steer = AverageMeter("steer")
        self._avg_brake = AverageMeter("brake")
        self._avg_throttle = AverageMeter("throttle")
        self._avg_impulse_intensity = AverageMeter("impulse_intensity")

        # recording settings
        if enable_video:
            self._enable_video = True
            self._video_settings = video_settings or {
                "video_dir": "video",
                "fps": 20,
                "prefix": "episode",
            }
            self._recorder = VideoRecorder(
                self._video_settings.get("video_dir", "video"),
                self._video_settings.get("fps", 20),
                self._video_settings.get("prefix", "episode"),
            )
            self._episode_counter = 0
        else:
            self._enable_video = False

        self.seed(self._seed)

        self._use_tm = use_tm
        self.world = load_carla_env(
            map_name,
            client,
            host,
            client_port,
            npc_v_port,
            use_tm,
            sim_delta_seconds,
            **self._env_kwargs,
        )
        self._spawn_ego_and_sensors()

    @property
    def unwrapped(self):
        return self.world

    @property
    def action_space(self):
        return self._action_space

    @property
    def weather(self):
        return self._env_kwargs["weather"]

    @property
    def observation_space(self):
        return get_obs_space(
            self._modalities,
            self._channels_first,
            self._height,
            self._width * self._n_cam,
            self._k,
        )

    def seed(self, seed=None):
        random.seed(seed)
        self._action_space.seed(seed)

    def _spawn_ego_and_sensors(self):
        assert self._ego in get_all_vehicles(
            self.world
        ), f"please specify a vehicle from {get_all_vehicles(self.world)}, provided {self._ego}"
        ego_bp = self.world.get_blueprint_library().find("vehicle." + self._ego)
        ego_bp.set_attribute("role_name", "ego")
        if ego_bp.has_attribute("color"):
            if self._ego_vehicle_color is None:
                ego_color = random.choice(
                    ego_bp.get_attribute("color").recommended_values
                )
            else:
                ego_color = self._ego_vehicle_color
            ego_bp.set_attribute("color", ego_color)

        # spawn the ego
        spawn_points = self.world.get_map().get_spawn_points()
        assert 0 < len(
            spawn_points
        ), "could not found any points to spawn the ego vehicle"
        MAX_TRIAL = 10
        for _ in range(MAX_TRIAL):
            ego_transform = random.choice(spawn_points)
            self._player = self.world.try_spawn_actor(ego_bp, ego_transform)
            if self._player is not None:
                break
        else:
            raise RuntimeError(
                f"All {MAX_TRIAL} spawn actor attempts failed "
                f"due to collision at spawn position."
            )

        # spawn observation cameras
        _k = range(-(self._n_cam // 2), -(self._n_cam // 2) + self._n_cam)
        _k = _k if self._n_cam % 2 else [item + 1 / 2 for item in _k]
        _cam_orientation = [item * self._fov for item in _k]

        for _orientation in _cam_orientation:
            cam_location = carla.Location(z=2.4)
            cam_rotation = carla.Rotation(yaw=_orientation)
            cam_transform = carla.Transform(cam_location, cam_rotation)

            if "rgb" in self._modalities:
                rgb_queue = queue.Queue()
                rgb_cam_bp = self.world.get_blueprint_library().find(
                    "sensor.camera.rgb"
                )
                rgb_cam_bp.set_attribute("image_size_x", str(self._width))
                rgb_cam_bp.set_attribute("image_size_y", str(self._height))
                rgb_cam_bp.set_attribute("fov", str(self._fov))
                rgb_cam = self.world.spawn_actor(
                    rgb_cam_bp, cam_transform, attach_to=self._player
                )
                self._rgb_cam_list.append(rgb_cam)
                self._all_sensors.append(rgb_cam)
                rgb_cam.listen(rgb_queue.put)
                self._rgb_queue_list.append(rgb_queue)

            if "depth" in self._modalities:
                depth_queue = queue.Queue()
                depth_cam_bp = self.world.get_blueprint_library().find(
                    "sensor.camera.depth"
                )
                depth_cam_bp.set_attribute("image_size_x", str(self._width))
                depth_cam_bp.set_attribute("image_size_y", str(self._height))
                depth_cam_bp.set_attribute("fov", str(self._fov))
                depth_cam = self.world.spawn_actor(
                    depth_cam_bp, cam_transform, attach_to=self._player
                )
                self._depth_cam_list.append(depth_cam)
                self._all_sensors.append(depth_cam)
                depth_cam.listen(depth_queue.put)
                self._depth_queue_list.append(depth_queue)

            if "semantic_segmentation" in self._modalities:
                seg_queue = queue.Queue()
                seg_cam_bp = self.world.get_blueprint_library().find(
                    "sensor.camera.semantic_segmentation"
                )
                seg_cam_bp.set_attribute("image_size_x", str(self._width))
                seg_cam_bp.set_attribute("image_size_y", str(self._height))
                seg_cam_bp.set_attribute("fov", str(self._fov))
                seg_cam = self.world.spawn_actor(
                    seg_cam_bp, cam_transform, attach_to=self._player
                )
                self._seg_cam_list.append(seg_cam)
                self._all_sensors.append(seg_cam)
                seg_cam.listen(seg_queue.put)
                self._seg_queue_list.append(seg_queue)

        # place collision detector
        col_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        origin_transform = carla.Transform(
            carla.Location(0, 0, 0), carla.Rotation(0, 0, 0)
        )
        self._col_sensor = self.world.spawn_actor(
            col_bp, origin_transform, attach_to=self._player
        )
        self._col_sensor.listen(
            lambda event: self._on_collision(weakref.ref(self), event)
        )
        self._all_sensors.append(self._col_sensor)

        # place lane invasion detector
        if "lane_invasion" in self._modalities:
            inv_bp = self.world.get_blueprint_library().find(
                "sensor.other.lane_invasion"
            )
            self._inv_sensor = self.world.spawn_actor(
                inv_bp, origin_transform, attach_to=self._player
            )
            self._inv_sensor.listen(
                lambda event: self._on_invasion(weakref.ref(self), event)
            )
            self._all_sensors.append(self._inv_sensor)

        # place obstacle detector
        if "obstacle_detection" in self._modalities:
            obst_bp = self.world.get_blueprint_library().find("sensor.other.obstacle")
            # hit_radius should be properly chosen
            # refer to this issue for more details: https://github.com/carla-simulator/carla/issues/2863
            obst_bp.set_attribute(
                "hit_radius", str(self._env_kwargs.get("hit_radius", 0.25))
            )
            self._obst_sensor = self.world.spawn_actor(
                obst_bp, origin_transform, attach_to=self._player
            )
            self._obst_sensor.listen(
                lambda event: self._on_detect_obstacle(weakref.ref(self), event)
            )
            self._all_sensors.append(self._obst_sensor)

        # spawn render camera
        render_cam_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self._render_cam = self.world.spawn_actor(
            render_cam_bp, get_spectator_transform(self._player.get_transform())
        )
        self._all_sensors.append(self._render_cam)

        def callback_render(carla_img):
            self._render_img = carla_rgb_to_np(carla_img, False)

        self._render_cam.listen(lambda img: callback_render(img))

        # spawn GNSS sensor
        if "gnss" in self._modalities:
            gnss_bp = self.world.get_blueprint_library().find("sensor.other.gnss")
            gnss_bp.set_attribute("noise_seed", str(self._seed))
            # as GNSS measurement is not supposed to vary significantly, reduce the retrieve frequency
            gnss_bp.set_attribute("sensor_tick", str(3.0))
            self._gnss_sensor = self.world.spawn_actor(
                gnss_bp, origin_transform, attach_to=self._player
            )
            self._all_sensors.append(self._gnss_sensor)

            def callback_gnss(carla_gnss_meas):
                self._gnss_meas = [
                    carla_gnss_meas.altitude,
                    carla_gnss_meas.latitude,
                    carla_gnss_meas.longitude,
                ]

            self._gnss_sensor.listen(
                lambda carla_gnss_meas: callback_gnss(carla_gnss_meas)
            )

        # spawn IMU sensor
        if (
            "accelerometer" in self._modalities
            or "compass" in self._modalities
            or "gyroscope" in self._modalities
        ):
            imu_bp = self.world.get_blueprint_library().find("sensor.other.imu")
            # as IMU measurement is not supposed to vary significantly, reduce the retrieve frequency
            imu_bp.set_attribute("sensor_tick", str(3.0))
            self._imu_sensor = self.world.spawn_actor(
                imu_bp, origin_transform, attach_to=self._player
            )
            self._all_sensors.append(self._imu_sensor)

            def callback_imu(meas):
                self._accelerometer = [
                    meas.accelerometer.x,
                    meas.accelerometer.y,
                    meas.accelerometer.z,
                ]
                self._compass = meas.compass
                self._gyroscope = [meas.gyroscope.x, meas.gyroscope.y, meas.gyroscope.z]

            self._imu_sensor.listen(lambda carla_imu_meas: callback_imu(carla_imu_meas))

        # spawn LIDAR sensor
        if "lidar" in self._modalities:
            lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
            self._lidar_sensor = self.world.spawn_actor(
                lidar_bp, origin_transform, attach_to=self._player
            )
            self._all_sensors.append(self._lidar_sensor)

            def callback_lidar(carla_lidar_meas):
                self._lidar_meas["horizontal_angle"] = carla_lidar_meas.horizontal_angle
                self._lidar_meas["raw_data"] = carla_lidar_meas.raw_data

            self._lidar_sensor.listen(
                lambda carla_lidar_meas: callback_lidar(carla_lidar_meas)
            )

        # spawn radar sensor
        if "radar" in self._modalities:
            radar_bp = self.world.get_blueprint_library().find("sensor.other.radar")
            self._radar_sensor = self.world.spawn_actor(
                radar_bp, origin_transform, attach_to=self._player
            )
            self._all_sensors.append(self._radar_sensor)

            def callback_radar(m):
                self._radar_meas = np.reshape(
                    np.frombuffer(m.raw_data, dtype=np.dtype("f4")), (len(m), 4)
                )

            self._radar_sensor.listen(
                lambda carla_radar_meas: callback_radar(carla_radar_meas)
            )

    def _on_collision(self, weakself, event):
        _self = weakself()
        if not _self:
            return
        impulse = event.normal_impulse
        self._curr_col_frame = event.frame
        self._curr_col_intensity = math.sqrt(
            impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2
        )

    def _on_invasion(self, weakself, event):
        _self = weakself()
        if not _self:
            return
        self._curr_inv_frame = event.frame
        self._curr_invasion = event.crossed_lane_markings

    def _on_detect_obstacle(self, weakself, event):
        _self = weakself()
        if not _self:
            return
        self._curr_obst_frame = event.frame
        self._curr_obst_distance = event.distance

    def _update_detector_data(self):
        if self._curr_col_frame != self._prev_col_frame:
            self._impulse_intensity = self._curr_col_intensity
        else:
            self._impulse_intensity = 0.0
        self._prev_col_frame = self._curr_col_frame

        if self._curr_inv_frame == self._prev_inv_frame:
            self._curr_invasion = None
        self._prev_inv_frame = self._curr_inv_frame

        if self._curr_obst_frame == self._prev_obst_frame:
            self._curr_obst_distance = 0.0
        self._prev_obst_frame = self._curr_obst_frame

    def _tick(self):
        self._n_time_steps += 1
        self.world.tick()
        self._update_detector_data()

        _spectator = self.world.get_spectator()
        _spectator.set_transform(get_spectator_transform(self._player.get_transform()))

        if self._rgb_queue_list is not None:
            self._rgb_img_list = []
            for _rgb_queue in self._rgb_queue_list:
                self._rgb_img_list.append(
                    carla_rgb_to_np(_rgb_queue.get(), self._channels_first)
                )

        if self._depth_queue_list is not None:
            self._depth_img_list = []
            for _depth_queue in self._depth_queue_list:
                self._depth_img_list.append(
                    carla_depth_to_np(_depth_queue.get(), self._channels_first)
                )

        if self._seg_queue_list is not None:
            self._seg_img_list = []
            for _seg_queue in self._seg_queue_list:
                self._seg_img_list.append(
                    carla_seg_to_np(_seg_queue.get(), self._channels_first)
                )

        if self._enable_video:
            img = self.render(mode="rgb_array")
            if isinstance(img, np.ndarray):
                self._recorder.add_frame(img)

    def _get_obs(self):
        _obs_rgb = None
        _obs_depth = None
        _obs_seg = None
        if "rgb" in self._modalities:
            if self._stack_cams:
                _obs_rgb = np.concatenate(tuple(self._rgb_img_list), axis=0)
            else:
                _obs_rgb = (
                    np.concatenate(tuple(self._rgb_img_list), axis=2)
                    if self._channels_first
                    else np.concatenate(tuple(self._rgb_img_list), axis=1)
                )
        if "depth" in self._modalities:
            _obs_depth = (
                np.concatenate(tuple(self._depth_img_list), axis=2)
                if self._channels_first
                else np.concatenate(tuple(self._depth_img_list), axis=1)
            )
        if "semantic_segmentation" in self._modalities:
            _obs_seg = (
                np.concatenate(tuple(self._seg_img_list), axis=2)
                if self._channels_first
                else np.concatenate(tuple(self._seg_img_list), axis=1)
            )

        _obs_with_all_modalities = {
            "rgb": _obs_rgb,
            "depth": _obs_depth,
            "semantic_segmentation": _obs_seg,
            "collision": self._impulse_intensity,
            "lane_invasion": self._curr_invasion,
            "obstacle_detection": self._curr_obst_distance,
            "gnss": self._gnss_meas,
            "accelerometer": self._accelerometer,
            "compass": self._compass,
            "gyroscope": self._gyroscope,
            "lidar": self._lidar_meas,
            "radar": self._radar_meas,
        }
        obs = {}
        for key in self._modalities:
            obs[key] = _obs_with_all_modalities[key]
        return obs

    def _is_done(self):
        if (
            self._terminate_on_collision
            and self._curr_col_intensity >= self._terminate_on_collision
        ):
            self._impulse_intensity = self._curr_col_intensity
            return True
        else:
            return False

    def _apply_control(self, control):
        self._control = control
        if control is not None:
            self._player.apply_control(control)

    def _compute_unit_direction(self):
        _map = self.world.get_map()
        _curr_wp = _map.get_waypoint(self._player.get_location(), project_to_road=True)
        _next_wp = _curr_wp.next(self._delta_distance)[0]
        _next_wp = _next_wp.transform.location
        _curr_wp = _curr_wp.transform.location
        _direction_vec = _next_wp - _curr_wp
        _norm = math.sqrt(
            _direction_vec.x ** 2 + _direction_vec.y ** 2 + _direction_vec.z ** 2
        )
        unit_direction_vec = _direction_vec * (1 / _norm)
        return unit_direction_vec

    def _compute_reward(self):
        _unit_direction_vec = self._compute_unit_direction()
        _ego_v = self._player.get_velocity()
        reward = (
            _unit_direction_vec.x * _ego_v.x
            + _unit_direction_vec.y * _ego_v.y
            + _unit_direction_vec.z * _ego_v.z
        )
        reward = reward * self._delta_t
        # calculate the cumulative progression
        self._progression += reward
        # penalize collision
        reward -= self._collision_penalty * self._impulse_intensity
        # penalize steering
        reward -= self._steer_penalty * abs(self._control.steer)
        return reward

    def step(self, action):
        assert (isinstance(action, list) or isinstance(action, np.ndarray)) and len(
            action
        ) == 2, f"please provide a valid action pair [throttle_brake, steer], provided {action}"

        action = convert_action(action)
        reward = 0.0
        for _ in range(self._action_repeat):
            self._tick()
            self._apply_control(action)
            reward += self._compute_reward()
            done = self._is_done()

            # update cumulative metrics
            self._avg_steer.update(abs(action.steer))
            self._avg_brake.update(action.brake)
            self._avg_throttle.update(action.throttle)
            if done:
                break

        obs = self._get_obs()
        _v = self._player.get_velocity()
        _w = self._player.get_angular_velocity()
        _a = self._player.get_acceleration()
        _loc = self._player.get_location()
        self._max_impulse_intensity = max(
            self._max_impulse_intensity, self._impulse_intensity
        )
        self._avg_impulse_intensity.update(self._impulse_intensity)
        extra = {
            "velocity": np.array([_v.x, _v.y, _v.z]),
            "angular_velocity": np.array([_w.x, _w.y, _w.z]),
            "acceleration": np.array([_a.x, _a.y, _a.z]),
            "location": np.array([_loc.x, _loc.y, _loc.z]),
            "progression": self._progression,
            "crash_intensity": self._impulse_intensity,
            "max_crash_intensity": self._max_impulse_intensity,
            "average_crash_intensity": self._avg_impulse_intensity.avg,
            "average_steer": self._avg_steer.avg,
            "average_brake": self._avg_brake.avg,
            "average_throttle": self._avg_throttle.avg,
        }
        if self._k is not None:
            self._frames.append(obs)
            return get_stacked_obs(self._frames, self._modalities), reward, done, extra
        else:
            return obs, reward, done, extra

    def reset(self):
        if self._enable_video:
            self._recorder.stop()
            self._episode_counter += 1
            self._recorder.start(self._episode_counter)

        self._n_time_steps = 0
        self._rgb_img_list = list()
        self._depth_img_list = list()
        self._curr_col_intensity = 0.0
        self._control = carla.VehicleControl()
        self._progression = 0.0
        self._impulse_intensity = 0.0

        self._avg_steer.reset()
        self._avg_brake.reset()
        self._avg_throttle.reset()
        self._avg_impulse_intensity.reset()
        self._max_impulse_intensity = 0.0

        if self._rgb_queue_list:
            for _rgb_queue in self._rgb_queue_list:
                with _rgb_queue.mutex:
                    _rgb_queue.queue.clear()
        if self._depth_queue_list:
            for _depth_queue in self._depth_queue_list:
                with _depth_queue.mutex:
                    _depth_queue.queue.clear()

        _player_reset_transform = self.world.get_map().get_spawn_points()
        _player_reset_transform = random.choice(_player_reset_transform)
        self._player.set_transform(_player_reset_transform)
        self._player.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
        self._player.set_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
        self._tick()
        if self._k is not None:
            _obs = self._get_obs()
            for _ in range(self._k):
                self._frames.append(_obs)
            return get_stacked_obs(self._frames, self._modalities)
        else:
            return self._get_obs()

    def render(self, mode="human", backend="cv2"):
        self._render_cam.set_transform(
            get_spectator_transform(self._player.get_transform())
        )
        img = self._render_img

        if mode == "rgb_array":
            return img
        elif mode == "human":
            if backend == "matplotlib":
                plt.imshow(img, aspect="auto")
                plt.show()
            elif backend == "cv2":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("CARLA", img)
                cv2.waitKey(100)
            else:
                raise AssertionError("only 'matplotlib' and 'cv2' are supported")
        else:
            raise AssertionError(
                f"mode should be either 'rgb_array' or 'human', received {mode}"
            )

    def close(self):
        if self._use_tm:
            print(
                'WARNING: DO NOT call `env.close()` when "TrafficManager" is used '
                "(i.e. `use_traffic_manager = True`) as it will cause re-instantiation"
                " of the environment to hang. This call is ignored."
            )
            return

        for sensor in self._all_sensors:
            sensor.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
