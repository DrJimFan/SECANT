def is_installed():
    try:
        import carla

        return True
    except ImportError:
        return False


if is_installed():
    from .core import make_carla, ALL_WEATHERS

else:
    _ERROR_MSG = """
CARLA is not installed. Please follow the docs to install it.
    """.strip()
