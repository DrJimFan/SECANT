"""
Adapted from https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/spawn_npc.py
Wrapper around CARLA.
"""
import carla
import random
import collections

from .utils import (
    get_all_available_maps,
    get_weather_presets,
    apply_custom_weather,
    spawn_npc_vehicles,
    spawn_npc_walkers,
    snake_to_camel,
)


def load_carla_env(
    map_name,
    client=None,
    host="localhost",
    client_port=2000,
    npc_v_port=8000,
    use_tm=False,
    delta_seconds=0.05,
    **env_kwargs,
):
    assert (
        "seed" in env_kwargs
    ), "please specify a random seed for deterministic behaviours"
    random.seed(env_kwargs.get("seed"))

    if client is None:
        _client = carla.Client(host, client_port)
    else:
        _client = client
    _client.set_timeout(env_kwargs.get("timeout", 30.0))

    syn_master = False

    # load map
    map_name = map_name.capitalize()
    assert map_name in get_all_available_maps(
        _client
    ), f"please specify a valid map name from {get_all_available_maps(_client)}"
    world = _client.load_world(map_name)
    _settings = world.get_settings()

    # set up traffic manager
    _tm = _client.get_trafficmanager(npc_v_port) if use_tm else None
    if use_tm:
        _tm.set_global_distance_to_leading_vehicle(
            env_kwargs.get("npc_vehicles_distance", 2.0)
        )

    if env_kwargs.get("sync", True):
        if use_tm:
            _tm.set_synchronous_mode(True)
        if not _settings.synchronous_mode:
            syn_master = True
            _settings.synchronous_mode = True
            if delta_seconds == 0:
                print(
                    "Warning: Synchronous mode with variable time-step "
                    "leads to non reliable simulations. "
                    "Use fixed time-step instead."
                )
    else:
        syn_master = False

    # set fixed_delta_seconds
    assert isinstance(delta_seconds, float)
    _settings.fixed_delta_seconds = delta_seconds
    world.apply_settings(_settings)

    # set weather
    if env_kwargs.get("weather", "default") == "custom":
        assert (
            env_kwargs.get("weather_args") is not None
        ), "please provide valid weather values"
        world.set_weather(
            apply_custom_weather(world.get_weather(), env_kwargs.get("weather_args"))
        )
    else:
        assert (
            env_kwargs.get("weather", "default") in get_weather_presets()
        ), f"please specify a valid weather preset from {get_weather_presets()}"
        if env_kwargs.get("weather_args") is not None:
            print("Warning: Custom weather values will be ignored when using a preset.")
        world.set_weather(
            getattr(
                carla.WeatherParameters,
                snake_to_camel(env_kwargs.get("weather", "default")),
            )
        )

    _actor_dict = collections.defaultdict(list)

    # spawn npc vehicles
    _client, world, _tm, _actor_dict["npc_vehicles"] = spawn_npc_vehicles(
        _client, world, _tm, npc_v_port, syn_master, **env_kwargs
    )

    # spawn npc walkers
    _client, world, walkers_list, all_id = spawn_npc_walkers(
        _client, world, syn_master, **env_kwargs
    )
    _actor_dict["npc_walkers_list"] = walkers_list
    _actor_dict["all_id"] = all_id
    return world
