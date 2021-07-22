import re
import random
import numpy as np
from typing import Optional

import carla

from gym import spaces

# the following cannot be imported directly
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor


def get_all_vehicles(world):
    blueprints = [bp for bp in world.get_blueprint_library().filter("vehicle")]
    vehicles = [blueprint.id for blueprint in blueprints]
    return [vehicle.replace("vehicle.", "") for vehicle in vehicles]


def carla_rgb_to_np(carla_rgb, channels_first):
    if carla_rgb:
        img = np.frombuffer(carla_rgb.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (carla_rgb.height, carla_rgb.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        return img.transpose(2, 0, 1).copy() if channels_first else img
    else:
        return None


def carla_depth_to_np(carla_depth, channels_first):
    if carla_depth:
        raw = np.frombuffer(carla_depth.raw_data, dtype=np.dtype("uint8"))
        raw = np.reshape(raw, (carla_depth.height, carla_depth.width, 4))
        converted = (raw[:, :, 2] + raw[:, :, 1] * 256 + raw[:, :, 0] * 256 ** 2) / (
            256 ** 3 - 1
        )
        return (
            np.expand_dims(converted, 0)
            if channels_first
            else np.expand_dims(converted, 2)
        )
    else:
        return None


def carla_seg_to_np(carla_seg, channels_first):
    carla_seg.convert(carla.ColorConverter.CityScapesPalette)
    return carla_rgb_to_np(carla_seg, channels_first)


def convert_action(action: np.ndarray):
    throttle_brake, steer = float(action[0]), float(action[1])
    assert (
        -1.0 <= throttle_brake <= 1.0
    ), f"please provide a valid value for -1 <= throttle_brake <= 1: {throttle_brake}"
    assert (
        -1.0 <= steer <= 1.0
    ), f"please provide a valid value for steer, provided {steer}"

    if throttle_brake >= 0.0:
        throttle = throttle_brake
        brake = 0.0
    else:
        throttle = 0.0
        brake = -throttle_brake

    return carla.VehicleControl(
        throttle=throttle,
        steer=steer,
        brake=brake,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
    )


def get_spectator_transform(vehicle_transform, z_offset=4.0, fixed_pitch=-30.0):
    vehicle_loc = vehicle_transform.location
    vehicle_rot = vehicle_transform.rotation
    vehicle_rot.pitch = fixed_pitch
    vehicle_rot.roll = 0.0
    spectator_loc = carla.Location(0.0, 0.0, z_offset) + vehicle_loc
    return carla.Transform(spectator_loc, vehicle_rot)


def get_all_available_maps(client):
    maps = [m.replace("/Game/Carla/Maps/", "") for m in client.get_available_maps()]
    return maps


def get_weather_presets():
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    presets_snake_case = [camel_to_snake(camel) for camel in presets]
    return presets_snake_case


def apply_custom_weather(weather, weather_args):
    if "azimuth" in weather_args:
        weather.sun_azimuth_angle = weather_args["azimuth"]
    if "altitude" in weather_args:
        weather.sun_altitude_angle = weather_args["altitude"]
    if "clouds" in weather_args:
        weather.cloudiness = weather_args["clouds"]
    if "rain" in weather_args:
        weather.precipitation = weather_args["rain"]
    if "puddles" in weather_args:
        weather.precipitation_deposits = weather_args["puddles"]
    if "wind" in weather_args:
        weather.wind_intensity = weather_args["wind"]
    if "fog" in weather_args:
        weather.fog_density = weather_args["fog"]
    if "fogdist" in weather_args:
        weather.fog_distance = weather_args["fogdist"]
    if "fogfalloff" in weather_args:
        weather.fog_falloff = weather_args["fogfalloff"]
    if "wetness" in weather_args:
        weather.wetness = weather_args["wetness"]
    return weather


def spawn_npc_vehicles(real_client, world, tm, npc_v_port, syn_master, **kwargs):
    npc_vehicles_list = []
    n_vehicles = kwargs.get("num_vehicles", 10)
    colors = kwargs.get("vehicle_color")
    if colors is not None:
        if len(colors) > 1:
            same_color = False
            colors = colors[1:]
        else:
            same_color = True

    assert (
        n_vehicles >= 0
    ), f"please provide a valid value for number of npc vehicles, provided {n_vehicles}"
    if tm is not None:
        tm.global_percentage_speed_difference(
            kwargs.get("npc_vehicles_speed_difference", 30.0)
        )

    if colors:
        assert len(colors) == 1 or len(colors) == n_vehicles, (
            f"number of provided color settings must match the number of NPC vehicles, "
            f"got {len(colors)} colors and {n_vehicles} NPC vehicles instead"
        )

    # get all vehicles
    bp_npc_vehicles = world.get_blueprint_library().filter("vehicle.*")

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if n_vehicles <= number_of_spawn_points:
        random.shuffle(spawn_points)
    else:
        print(
            f"Warning: request {n_vehicles} vehicles to be spawned, "
            f"but could only find {number_of_spawn_points} spawn points"
        )
        n_vehicles = number_of_spawn_points

    # now spawn npc vehicles
    batch = []
    for i, transform in enumerate(spawn_points):
        if i >= n_vehicles:
            break
        bp = random.choice(bp_npc_vehicles)
        if bp.has_attribute("color"):
            if colors is None:
                color = random.choice(bp.get_attribute("color").recommended_values)
            else:
                if same_color:
                    color = colors[0]
                    color = f"{color[0]},{color[1]},{color[2]}"
                else:
                    item = colors[len(npc_vehicles_list)]
                    color = f"{item[0]},{item[1]},{item[2]}"
            bp.set_attribute("color", color)
        if bp.has_attribute("driver_id"):
            driver_id = random.choice(bp.get_attribute("driver_id").recommended_values)
            bp.set_attribute("driver_id", driver_id)
        bp.set_attribute("role_name", "autopilot")

        # spawn the cars and set their autopilot and light state all together
        if tm is not None:
            batch.append(
                SpawnActor(bp, transform).then(
                    SetAutopilot(FutureActor, True, tm.get_port())
                )
            )
        else:
            batch.append(
                SpawnActor(bp, transform).then(
                    SetAutopilot(FutureActor, True, npc_v_port)
                )
            )

        for response in real_client.apply_batch_sync(batch, syn_master):
            if not response.error:
                npc_vehicles_list.append(response.actor_id)

    print(f"spawned {len(npc_vehicles_list)} vehicles")
    return real_client, world, tm, npc_vehicles_list


def spawn_npc_walkers(real_client, world, syn_master, **kwargs):
    assert (
        kwargs.get("num_walkers", 10) >= 0
    ), f"please provide a valid value for number of npc walkers, provided {kwargs.get('num_walkers', 10)}"

    walkers_list = []
    all_id = []
    bps = world.get_blueprint_library().filter("walker.pedestrian.*")

    spawn_points = []

    i = 0
    while i < kwargs.get("num_walkers", 10):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)
            i += 1

    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(bps)
        # set as not invincible
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        # set the max speed
        if walker_bp.has_attribute("speed"):
            if random.random() > kwargs.get("npc_walkers_run_percentage", 0.5):
                # walking
                walker_speed.append(
                    walker_bp.get_attribute("speed").recommended_values[1]
                )
            else:
                # running
                walker_speed.append(
                    walker_bp.get_attribute("speed").recommended_values[2]
                )
        else:
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = real_client.apply_batch_sync(batch, syn_master)

    walker_speed2 = []
    for i in range(len(results)):
        if not results[i].error:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2

    batch = []
    walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    for i in range(len(walkers_list)):
        batch.append(
            SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"])
        )
    results = real_client.apply_batch_sync(batch, syn_master)
    for i in range(len(results)):
        if not results[i].error:
            walkers_list[i]["con"] = results[i].actor_id

    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    if not kwargs.get("sync", True) or not syn_master:
        world.wait_for_tick()
    else:
        world.tick()

    world.set_pedestrians_cross_factor(kwargs.get("npc_walkers_cross_percentage", 0.1))
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

    print(f"spawned {len(walkers_list)} walkers")
    return real_client, world, walkers_list, all_id


def get_obs_space(
    modalities, channels_first, img_height, img_width, k: Optional[int] = None
):
    shape = [3, img_height, img_width] if channels_first else [img_height, img_width, 3]
    shape = ((k,) + tuple(shape)) if k else shape
    rgb = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    seg = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    shape = [1, img_height, img_width] if channels_first else [img_height, img_width, 1]
    shape = ((k,) + tuple(shape)) if k else shape
    depth = spaces.Box(low=0.0, high=1.0, shape=shape)

    shape = (k,) if k else (1,)
    col = spaces.Box(low=0.0, high=np.inf, shape=shape)

    lane_inv = spaces.Dict(
        {
            "color": spaces.Discrete(
                7
            ),  # Standard, Blue, Green, Red, White, Yellow, Other
            "lane_change": spaces.Discrete(4),  # NONE, Right, Left, Both
            "type": spaces.Discrete(
                11
            ),  # NONE, Other, Broken, Solid, SolidSolid, SolidBroken, BrokenSolid, BrokenBroken, BottsDots, Grass, Curb
            "width": spaces.Box(low=0.0, high=np.inf, shape=shape),
        }
    )

    obst_detection = spaces.Box(low=0.0, high=np.inf, shape=shape)

    gnss = spaces.Dict(
        {
            "altitude": spaces.Box(low=-np.inf, high=np.inf, shape=shape),
            "latitude": spaces.Box(low=-90.0, high=90.0, shape=shape),
            "longitude": spaces.Box(low=-180.0, high=180.0, shape=shape),
        }
    )

    shape = (k, 3) if k else (3,)
    accelerometer = spaces.Box(low=-np.inf, high=np.inf, shape=shape)
    gyroscope = spaces.Box(low=-np.inf, high=np.inf, shape=shape)
    shape = (k,) if k else (1,)
    compass = spaces.Box(low=-0.5 * np.pi, high=0.5 * np.pi, shape=shape)

    shp = (k, 4) if k else (4,)
    lidar = spaces.Dict(
        {
            "horizontal_angle": spaces.Box(
                low=-0.5 * np.pi, high=0.5 * np.pi, shape=shape
            ),
            "raw_data": spaces.Box(low=-np.inf, high=np.inf, shape=shp),
        }
    )

    radar = spaces.Dict(
        {
            "altitude": spaces.Box(low=-0.5 * np.pi, high=0.5 * np.pi, shape=shape),
            "azimuth": spaces.Box(low=0.0, high=2 * np.pi, shape=shape),
            "depth": spaces.Box(low=0.0, high=np.inf, shape=shape),
            "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=shape),
        }
    )

    all_modalities = {
        "rgb": rgb,
        "depth": depth,
        "semantic_segmentation": seg,
        "collision": col,
        "lane_invasion": lane_inv,
        "obstacle_detection": obst_detection,
        "gnss": gnss,
        "accelerometer": accelerometer,
        "compass": compass,
        "gyroscope": gyroscope,
        "lidar": lidar,
        "radar": radar,
    }

    obs_space = {}
    for key in modalities:
        obs_space[key] = all_modalities[key]
    return spaces.Dict(obs_space)


def get_stacked_obs(frames, modalities):
    rgb_stacked = None
    if "rgb" in modalities:
        rgb_stacked = []
        for frame in frames:
            rgb_stacked.append(np.expand_dims(frame["rgb"], 0))
        rgb_stacked = np.concatenate(rgb_stacked, axis=0)

    depth_stacked = None
    if "depth" in modalities:
        depth_stacked = []
        for frame in frames:
            depth_stacked.append(np.expand_dims(frame["depth"], 0))
        depth_stacked = np.concatenate(depth_stacked, axis=0)

    seg_stacked = None
    if "semantic_segmentation" in modalities:
        seg_stacked = []
        for frame in frames:
            seg_stacked.append(np.expand_dims(frame["semantic_segmentation"], 0))
        seg_stacked = np.concatenate(seg_stacked, axis=0)

    col_stacked = None
    if "collision" in modalities:
        col_stacked = []
        for frame in frames:
            col_stacked.append(np.expand_dims(frame["collision"], 0))
        col_stacked = np.concatenate(col_stacked, axis=0)

    inv_stacked = None
    if "lane_invasion" in modalities:
        inv_stacked = []
        for frame in frames:
            inv_stacked.append(frame["lane_invasion"])

    obst_stacked = None
    if "obstacle_detection" in modalities:
        obst_stacked = []
        for frame in frames:
            obst_stacked.append(np.expand_dims(frame["obstacle_detection"], 0))
        obst_stacked = np.concatenate(obst_stacked, axis=0)

    gnss_stacked = None
    if "gnss" in modalities:
        gnss_stacked = []
        for frame in frames:
            gnss_stacked.append(np.expand_dims(frame["gnss"], 0))
        gnss_stacked = np.concatenate(gnss_stacked, axis=0)

    a_stacked = None
    if "accelerometer" in modalities:
        a_stacked = []
        for frame in frames:
            a_stacked.append(np.expand_dims(frame["accelerometer"], 0))
        a_stacked = np.concatenate(a_stacked, axis=0)

    compass_stacked = None
    if "compass" in modalities:
        compass_stacked = []
        for frame in frames:
            compass_stacked.append(np.expand_dims(frame["compass"], 0))
        compass_stacked = np.concatenate(compass_stacked, axis=0)

    gyro_stacked = None
    if "gyroscope" in modalities:
        gyro_stacked = []
        for frame in frames:
            gyro_stacked.append(np.expand_dims(frame["gyroscope"], 0))
        gyro_stacked = np.concatenate(gyro_stacked, axis=0)

    lidar_stacked = None
    if "lidar" in modalities:
        lidar_stacked = []
        for frame in frames:
            lidar_stacked.append(frame["lidar"])

    radar_stacked = None
    if "radar" in modalities:
        radar_stacked = []
        for frame in frames:
            radar_stacked.append(frame["radar"])

    _obs_with_all_modalities = {
        "rgb": rgb_stacked,
        "depth": depth_stacked,
        "semantic_segmentation": seg_stacked,
        "collision": col_stacked,
        "lane_invasion": inv_stacked,
        "obstacle_detection": obst_stacked,
        "gnss": gnss_stacked,
        "accelerometer": a_stacked,
        "compass": compass_stacked,
        "gyroscope": gyro_stacked,
        "lidar": lidar_stacked,
        "radar": radar_stacked,
    }

    stacked_obs = {}
    for key in modalities:
        stacked_obs[key] = _obs_with_all_modalities[key]
    return stacked_obs


def snake_to_camel(word):
    return "".join(x.capitalize() or "_" for x in word.split("_"))


def camel_to_snake(camel):
    camel = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", camel).lower()
