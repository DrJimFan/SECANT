from dm_control.suite import ALL_TASKS as ALL_TASK_TUPLES


__all__ = [
    "dmc_expand_alias",
    "dmc_action_repeat",
    "dmc_name_to_domain_task",
    "ALL_TASKS",
    "ALL_TASK_TUPLES",
]

_ALIASES = {
    "cr": "cheetah_run",
    "ww": "walker_walk",
    "ws": "walker_stand",
    "re": "reacher_easy",
    "fs": "finger_spin",
    "fte": "finger_turn_easy",
    "bicc": "ball_in_cup_catch",
    "cs": "cartpole_swingup",
    "cb": "cartpole_balance",
}

ALL_TASKS = [f"{domain}_{task}" for domain, task in ALL_TASK_TUPLES]
ALL_TASKS.extend(_ALIASES.keys())


def dmc_expand_alias(task: str):
    return _ALIASES.get(task, task)


def dmc_action_repeat(task: str, protocol: str = "planet", default: int = 4):
    """
    Action repeat settings from:
    - PlaNet
    - Dreamer: always 2
    - Policy adaptation during deployment (PAD)
    - Default: uses the default value
    """
    task = dmc_expand_alias(task)
    protocol = protocol.lower()
    assert protocol in ["planet", "dreamer", "default", "pad"]
    assert default > 0

    if protocol == "dreamer":
        return 2
    elif protocol == "planet":
        return {
            "cartpole_swingup": 8,
            "cheetah_run": 4,
            "walker_walk": 2,
            "ball_in_cup_catch": 4,
            "finger_spin": 2,
            "reacher_easy": 4,
        }.get(task, default)
    elif protocol == "pad":
        if task.startswith("cartpole"):
            return 8
        elif task.startswith("finger"):
            return 2
        else:
            return 4
    elif protocol == "default":
        return default
    else:
        raise NotImplementedError(f"Unknown protocol: {protocol}")


def dmc_name_to_domain_task(name):
    if isinstance(name, (tuple, list)):
        assert len(name) == 2
        return tuple(name)
    name = dmc_expand_alias(name)
    assert "_" in name, f"{name}"
    if name == "ball_in_cup_catch":
        domain_name = "ball_in_cup"
        task_name = "catch"
    elif name == "point_mass_easy":
        domain_name = "point_mass"
        task_name = "easy"
    else:
        domain_name, task_name = name.split("_", 1)
    assert (
        domain_name,
        task_name,
    ) in ALL_TASK_TUPLES, f"task ({domain_name}, {task_name}) does not exist"
    return domain_name, task_name
