import os


def is_installed():
    try:
        # From dm_control Github page:
        # By default, dm_control will attempt to use GLFW first, then EGL, then OSMesa.
        # You can set MUJOCO_GL= environment variable to "glfw", "egl", or "osmesa".
        # When rendering with EGL, you can also specify which GPU to use for rendering
        # by setting the environment variable EGL_DEVICE_ID= to the target GPU ID.
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        import dm_control
        import dm_env

        return True
    except ImportError:
        return False


if is_installed():
    from .core import (
        make_dmc,
        ALL_TASKS,
        dmc_expand_alias,
        dmc_action_repeat,
        dmc_name_to_domain_task,
    )

else:
    _ERROR_MSG = """
Deepmind control suite is not installed. Please follow the docs to install it.
    """.strip()
