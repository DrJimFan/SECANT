import time

from typing import Optional, List
from .adapter import CarlaAdapter
from secant.wrappers import TimeLimit

from carla import Client


__all__ = ["make_carla"]

ALL_WEATHERS = [
    "clear_noon",
    "wet_sunset",
    "wet_cloudy_noon",
    "soft_rain_sunset",
    "mid_rain_sunset",
    "hard_rain_noon",
]

def _make_carla(
    map_name: str,
    env_kwargs: dict,
    ego_vehicle: str = "tesla.model3",
    seed: Optional[int] = None,
    client: Optional[Client] = None,
    host: Optional[str] = "localhost",
    client_port: int = 2000,
    use_traffic_manager: bool = False,
    npc_vehicles_port: int = 8000,
    sim_delta_seconds: float = 0.05,
    enable_video: bool = False,
    video_settings: Optional[dict] = None,
    weather: str = "default",
    weather_args: Optional[dict] = None,
    vehicle_color: Optional[List[list]] = None,
    modalities: Optional[list] = None,
    n_cam: int = 5,
    stack_cams: bool = False,
    fov: float = 60.0,
    episode_length: Optional[int] = 1000,
    terminate_on_collision: Optional[float] = None,
    collision_penalty: float = 5e-4,
    steer_penalty: float = 1.0,
    delta_distance: float = 0.1,
    delta_t: float = 0.05,
    image_height: int = 84,
    image_width: int = 84,
    action_repeat: int = 1,
    frame_stack: Optional[int] = None,
    channels_first: bool = True,
):
    """
    Low-level make

    Args:
        map_name: Str, specifies the town name. Use "see" to see a full list of available towns.
        env_kwargs:
            timeout: Float, sets in seconds the maximum time a network call is allowed before blocking it and
                raising a timeout exceeded error. Default: 30.0.
            no_rendering: Boolean. Only use True for faster debug. Default: False.
            sync: Boolean, sets the synchronization mode, default: True.
            num_vehicles: Int, sets the number of NPC vehicles, default: 10.
            num_walkers: Int, sets the number of NPC pedestrians, default: 10.
            npc_vehicles_distance: Float, specifies the distance between NPC vehicles. Default: 2.0. Can be configured
                only when `use_traffic_manager = True`.
            npc_vehicles_speed_difference: Float, specifies the speed percentage difference between NPC
                vehicles. Default: 30.0. Can be configured only when `use_traffic_manager = True`.
            npc_walkers_run_percentage: Float, specifies the percentage of running NPC walkers. Default: 0.5.
            npc_walkers_cross_percentage: Float, specifies the percentage of NPC walkers that will
                cross the roads. Default: 0.1.
        ego_vehicle: Str, specifies the model name of the ego vehicle. Use "see" to see a full list of available models.
        seed: Int, specifies a random seed for deterministic behaviours.
        client: CARLA client, default: None.
        host: The CARLA host server, default: "localhost".
        client_port: TCP port of the client to listen to, default: 2000.
        use_traffic_manager: bool, whether or not to use "carla.TrafficManager" to coordinate NPC vehicles.
        npc_vehicles_port: TCP port of npc vehicles. When `use_traffic_manager = True`, it specifies the port the
            traffic manager. When `use_traffic_manager = False`, it specifies a port which npc vehicles will directly
            connect to.
        sim_delta_seconds: Specifies the elapsed time between simulation steps. DO NOT use a value greater than 0.1.
            Refer to CARLA's doc for more info: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
        enable_video: bool, specifies recording or not.
        video_settings:
            video_dir: str, directory where recorded videos will be stored.
            recording_fps: int.
            prefix: str, specifies the prefix of file names, e.g. "episode_1.mp4", "episode_2.mp4" ...
        weather: Str, specifies a preset weather. Use "see" to see all available presets. Default: "default".
        weather_args:
            azimuth: Float. The azimuth angle of the sun in degrees. Values range from 0 to 360.
            altitude: Float. Altitude angle of the sun in degrees.
                Values range from -90 to 90 corresponding to midnight and midday each.
            clouds: Float. From 0 to 100, being 0 a clear sky and 100 one completely covered with clouds.
            rain: Float. Rain intensity values range from 0 to 100, being 0 none at all and 100 a heavy rain.
            puddles: Float. Determines the creation of puddles. Values range from 0 to 100, being 0 none at
                all and 100 a road completely capped with water. Puddles are created with static noise,
                meaning that they will always appear at the same locations.
            wind: Float. Controls the strength of the wind with values from 0, no wind at all, to 100,
                a strong wind. The wind does affect rain direction and leaves from trees,
                so this value is restricted to avoid animation issues.
            fog: Float. Fog concentration. It only affects the RGB camera sensor. Values range from 0 to 100.
            fogdist: Float. Fog start distance (in meters). Values range from 0 to infinite.
            fogfalloff: Float. Density of the fog (as in specific mass) from 0 to infinity. The bigger the
                value, the more dense and heavy it will be, and the fog will reach smaller heights.
            wetness: Float. Wetness intensity. Values range from 0 to 100.
        vehicle_color: List of [R, G, B] colors or None. If `None`, all vehicles will use recommended colors randomly.
            If a single list, all vehicles, including the ego vehicle and NPC vehicles, will use the same color.
            If multiple lists, all vehicles' colors will be set accordingly, with the first list specifying
            the color of the ego vehicle.
        modalities: List of strings. Specify required modalities.
            All supported modalities can be found by 'secant.envs.carla.adapter.ALL_MODALITIES'
        n_cam: Int, specifies the number of cameras. Default: 5.
        fov: Float, specifies the field of view of each cameras.
        episode_length: Int, specifies the length of an episode. Default: 1000.
        terminate_on_collision: Optional[float]. When a float is provided, it specifies a threshold of collision
            intensity. A collision with intensity exceeding that threshold will terminate the
            episode. When None, an episode will terminate only if the number of steps exceeds the
            maximum episode step. Default: 2000.0.
        collision_penalty: Float, specifies the weight on penalizing collision. Default: 1e-4.
        steer_penalty: Float, specifies the weight on penalizing steer. Default: 1.0.
        delta_distance: Float, specifies the distance between two waypoints used to compute the unit direction vector.
        delta_t: Float, specifies the time difference used to calculate progression and hence the reward
        image_width: Int, specifies the width of image captured by each camera. Default: 84.
        image_height: Int, specifies the height of image captured by each camera. Default: 84.
        action_repeat: Int, specifies how many times the same action will be repeated. Default: 1.
        frame_stack: Optional[int], specifies the number of frames to be stacked. Default: None.
        channels_first: Boolean. True for CHW, False for HWC. Default: True.
    """
    if seed is None:
        seed = int(time.time() * 1000000) % 1000000000
    else:
        assert isinstance(seed, int) and seed >= 0
    env_kwargs["seed"] = seed

    assert isinstance(map_name, str)

    assert isinstance(weather, str)
    env_kwargs["weather"] = weather

    assert isinstance(weather_args, dict) or weather_args is None
    env_kwargs["weather_args"] = weather_args

    assert vehicle_color is None or isinstance(vehicle_color, list)
    ego_vehicle_color = None if vehicle_color is None else vehicle_color[0]
    env_kwargs["vehicle_color"] = vehicle_color

    env = CarlaAdapter(
        map_name=map_name,
        client=client,
        host=host,
        client_port=client_port,
        npc_v_port=npc_vehicles_port,
        use_tm=use_traffic_manager,
        sim_delta_seconds=sim_delta_seconds,
        enable_video=enable_video,
        video_settings=video_settings,
        ego_vehicle=ego_vehicle,
        ego_vehicle_color=ego_vehicle_color,
        modalities=modalities,
        n_cam=n_cam,
        stack_cams=stack_cams,
        fov=fov,
        terminate_on_collision=terminate_on_collision,
        collision_penalty=collision_penalty,
        steer_penalty=steer_penalty,
        delta_distance=delta_distance,
        delta_t=delta_t,
        height=image_height,
        width=image_width,
        action_repeat=action_repeat,
        frame_stack=frame_stack,
        channels_first=channels_first,
        **env_kwargs
    )

    # shorten episode length
    if episode_length is not None:
        max_episode_steps = (episode_length + action_repeat - 1) // action_repeat
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env


def make_carla(
    map_name: str,
    seed: Optional[int] = None,
    client: Optional[Client] = None,
    host: Optional[str] = "localhost",
    client_port: int = 2000,
    use_traffic_manager: bool = False,
    npc_vehicles_port: int = 8000,
    sim_delta_seconds: float = 0.05,
    stack_cams: bool = False,
    enable_video: bool = False,
    video_settings: Optional[dict] = None,
    weather: str = "default",
    weather_args: Optional[dict] = None,
    modalities: Optional[list] = None,
    frame_stack: Optional[int] = None,
    action_repeat: int = 1,
    num_walkers: int = 10,
    num_vehicles: int = 10,
    image_height: int = 84,
    terminate_on_collision: Optional[float] = None,
):
    modalities = modalities or ["rgb"]
    env = _make_carla(
        map_name=map_name,
        env_kwargs={
            "timeout": 30.0,
            "no_rendering": False,
            "sync": True,
            "num_vehicles": num_vehicles,
            "num_walkers": num_walkers,
            "npc_vehicles_distance": 2.0,
            "npc_vehicles_speed_difference": 30.0,
            "npc_walkers_run_percentage": 0.5,
            "npc_walkers_cross_percentage": 0.1,
        },
        ego_vehicle="tesla.model3",
        seed=seed,
        client=client,
        host=host,
        client_port=client_port,
        use_traffic_manager=use_traffic_manager,
        npc_vehicles_port=npc_vehicles_port,
        sim_delta_seconds=sim_delta_seconds,
        enable_video=enable_video,
        video_settings=video_settings,
        weather=weather,
        weather_args=weather_args,
        modalities=modalities,
        n_cam=5,
        stack_cams=stack_cams,
        fov=60.0,
        episode_length=1000,
        terminate_on_collision=terminate_on_collision,
        collision_penalty=1e-4,
        steer_penalty=1.0,
        delta_distance=0.1,
        delta_t=0.05,
        image_height=image_height,
        image_width=image_height,
        action_repeat=action_repeat,
        frame_stack=frame_stack,
        channels_first=True,
    )
    return env
