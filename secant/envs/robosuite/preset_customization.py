import copy
import numpy as np

# -------------------------------------------
# ------------ Default Settings -------------
# -------------------------------------------

DEFAULT_TEXTURE_ALIAS = {
    "floor": "floorplane",
    "wall": "walls_mat",
    "table_legs": "table_legs_metal",
    "table": "table_ceramic",
    "door": "MatDarkWood",
    "door_handle": "MatMetal",
    "lift_object": "redwood_mat",
    "na_metal1": "smetal",
    "na_metal2": "bmetal",
    "pp_table1": "light-wood",
    "pp_table2": "dark-wood",
    "stack_object1": "greenwood_mat",
    "stack_object2": "redwood_mat",
    "handoff_hammer_head": "metal_mat",
    "handoff_hammer_body": "wood_mat",
    "ta_lift_pot": "pot_mat",
    "ta_lift_handle1": "handle1_mat",
    "ta_lift_handle2": "handle2_mat",
    "ta_pih_plate": "plate_mat",
    "ta_pih_stick": "greenwood_mat",
}

DEFAULT_TEXTURE_TYPE = {
    "floor": "2d",
    "wall": "2d",
    "table_legs": "cube",
    "table": "2d",
    "door": "cube",
    "door_handle": "cube",
    "lift_object": "cube",
    "na_metal1": "cube",
    "na_metal2": "cube",
    "pp_table1": "2d",
    "pp_table2": "2d",
    "stack_object1": "cube",
    "stack_object2": "cube",
    "handoff_hammer_head": "cube",
    "handoff_hammer_body": "cube",
    "ta_lift_pot": "cube",
    "ta_lift_handle1": "cube",
    "ta_lift_handle2": "cube",
    "ta_pih_plate": "cube",
    "ta_pih_stick": "cube",
}

DEFAULT_TASK_TEXTURE_LIST = {
    "Door": ["floor", "wall", "table_legs", "table", "door", "door_handle"],
    "Lift": ["floor", "wall", "table_legs", "table", "lift_object"],
    "NutAssembly": ["floor", "wall", "table_legs", "table", "na_metal1", "na_metal2"],
    "NutAssemblyRound": [
        "floor",
        "wall",
        "table_legs",
        "table",
        "na_metal1",
        "na_metal2",
    ],
    "NutAssemblySingle": [
        "floor",
        "wall",
        "table_legs",
        "table",
        "na_metal1",
        "na_metal2",
    ],
    "NutAssemblySquare": [
        "floor",
        "wall",
        "table_legs",
        "table",
        "na_metal1",
        "na_metal2",
    ],
    "PickPlace": ["floor", "wall", "table_legs", "pp_table1", "pp_table2"],
    "PickPlaceBread": ["floor", "wall", "table_legs", "pp_table1", "pp_table2"],
    "PickPlaceCan": ["floor", "wall", "table_legs", "pp_table1", "pp_table2"],
    "PickPlaceCereal": ["floor", "wall", "table_legs", "pp_table1", "pp_table2"],
    "PickPlaceMilk": ["floor", "wall", "table_legs", "pp_table1", "pp_table2"],
    "PickPlaceSingle": ["floor", "wall", "table_legs", "pp_table1", "pp_table2"],
    "Stack": ["floor", "wall", "table_legs", "table", "stack_object1", "stack_object2"],
    "TwoArmHandover": [
        "floor",
        "wall",
        "table_legs",
        "table",
        "handoff_hammer_head",
        "handoff_hammer_body",
    ],
    "TwoArmLift": [
        "floor",
        "wall",
        "table_legs",
        "table",
        "ta_lift_pot",
        "ta_lift_handle1",
        "ta_lift_handle2",
    ],
    "TwoArmPegInHole": ["floor", "wall", "ta_pih_plate", "ta_pih_stick"],
    "Wipe": ["floor", "wall", "table_legs", "table"],
}


# -------------------------------------------
# ---- SECANT's Preset Configurations ----
# -------------------------------------------

PRESET_TEXTURE_CONFIG = {
    # random seed
    "seed": 100,
    # a dictionary that maps alias names of target objects (see DEFAULT_TEXTURE_ALIAS)
    # to some texture candidates. Keys should be subsets of keys from
    # DEFAULT_TEXTURE_ALIAS'. Values should be a texture candidate or a list of texture
    # candidates. If a list of texture candidates is provided, a random candidate will
    # be selected and applied to the corresponding target object.
    # A texture candidate is either a string representing the path to a image texture
    # file, or a tuple of RGB values normalized to [0, 1]. Note that you can use the
    # names of robosuite's builtin texture files as texture candidate.
    # Check `secant.envs.robosuite.ALL_TEXTURES` for all the builtin texture names,
    # or `secant.envs.robosuite.TEXTURES` to get a dictionary that maps these names
    # to their source files.
    "tex_candidate": {
        "floor": [
            "WoodTiles",
            "WoodPanels",
            "WoodDark",
            "WoodLight",
            "WoodgrainGray",
            "FeltGray",
            "Dirt",
        ],
        "wall": [
            "PlasterCream",
            "PlasterPink",
            "PlasterYellow",
            "PlasterGray",
            "PlasterWhite",
            "BricksWhite",
            "Clay",
        ],
        "table_legs": ["Metal", "SteelBrushed", "SteelScratched"],
        "table": [
            "Ceramic",
            "PlasterCream",
            "PlasterYellow",
            "PlasterGray",
            "WoodDark",
            "WoodLight",
        ],
        "door": ["WoodDark", "WoodLight", "WoodGreen", "WoodRed", "WoodBlue"],
        "door_handle": ["Brass", "Metal", "SteelBrushed", "SteelScratched"],
        "lift_object": ["WoodGreen", "WoodRed", "WoodBlue"],
        "na_metal1": [
            "Brass",
            "SteelBrushed",
            "Metal",
            "WoodGreen",
            "WoodRed",
            "WoodBlue",
        ],
        "na_metal2": [
            "Brass",
            "SteelBrushed",
            "Metal",
            "WoodGreen",
            "WoodRed",
            "WoodBlue",
        ],
        "pp_table1": [
            "WoodDark",
            "Ceramic",
            "PlasterCream",
            "PlasterYellow",
            "PlasterGray",
            "WoodLight",
            "WoodGreen",
            "WoodRed",
            "WoodBlue",
        ],
        "pp_table2": [
            "WoodDark",
            "Ceramic",
            "PlasterCream",
            "PlasterYellow",
            "PlasterGray",
            "WoodLight",
            "WoodGreen",
            "WoodRed",
            "WoodBlue",
        ],
        "stack_object1": ["WoodGreen", "WoodRed", "WoodBlue"],
        "stack_object2": ["WoodGreen", "WoodRed", "WoodBlue"],
        "handoff_hammer_head": ["Brass", "Metal", "SteelBrushed", "SteelScratched"],
        "handoff_hammer_body": [
            "WoodDark",
            "WoodLight",
            "WoodGreen",
            "WoodRed",
            "WoodBlue",
        ],
        "ta_lift_pot": ["WoodGreen", "WoodRed", "WoodBlue"],
        "ta_lift_handle1": ["WoodGreen", "WoodRed", "WoodBlue"],
        "ta_lift_handle2": ["WoodGreen", "WoodRed", "WoodBlue"],
        "ta_pih_plate": ["WoodGreen", "WoodRed", "WoodBlue"],
        "ta_pih_stick": ["WoodGreen", "WoodRed", "WoodBlue"],
    },
    # List of sets, where each set contains some texture keys. Each texture key from a
    # set will be assigned a different texture from the other keys in the same set.
    # This arg is used to enforce texture difference on important objects so that they
    # are more easily identified by vision algorithms. If None, no constraint is used.
    "tex_diff_constraint": [
        ["na_metal1", "na_metal2"],
        ["pp_table1", "pp_table2"],
        ["stack_object1", "stack_object2"],
        ["ta_lift_pot", "ta_lift_handle1", "ta_lift_handle2"],
        ["ta_pih_plate", "ta_pih_stick"],
    ],
    # List of object alias names whose texture need to be modified. If None, use
    # SECANT's default texture list in DEFAULT_TASK_TEXTURE_LIST[task]
    "tex_to_change": None,
    # A dictionary that maps object alias names to their texture types. Check
    # http://www.mujoco.org/book/XMLreference.html#texture on how texture types affect
    # rendering effect
    "tex_type": DEFAULT_TEXTURE_TYPE,
}

# Arguments to pass to the TextureModder of Robosuite
# https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/mjmod.py#L779
PRESET_COLOR_CONFIG = {
    "seed": 100,  # random seed
    "geom_names": None,  # all geoms are randomized
    "randomize_local": True,  # sample nearby colors
    "randomize_material": True,  # randomize material reflectance / shininess / specular
    "local_rgb_interpolation": 0.2,
    "local_material_interpolation": 0.3,
    "texture_variations": [
        "rgb",
        "checker",
        "noise",
        # "gradient",
    ],  # all texture variation types
    "randomize_skybox": True,  # by default, randomize skybox too
}

# Arguments to pass to the CameraModder of Robosuite
# https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/mjmod.py#L516
PRESET_CAMERA_CONFIG = {
    "seed": 100,  # random seed
    "camera_names": None,  # all cameras are randomized
    "randomize_position": True,
    "randomize_rotation": True,
    "randomize_fovy": True,
    "position_perturbation_size": 0.01,
    "rotation_perturbation_size": 0.087,
    "fovy_perturbation_size": 5.0,
}

# Arguments to pass to the LightingModder of Robosuite
# https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/mjmod.py#L60
PRESET_LIGHTING_CONFIG = {
    "seed": 100,  # random seed
    "light_names": None,  # all lights are randomized
    "randomize_position": True,
    "randomize_direction": True,
    "randomize_specular": True,
    "randomize_ambient": True,
    "randomize_diffuse": True,
    "randomize_active": True,
    "position_perturbation_size": 0.1,
    "direction_perturbation_size": 0.35,
    "specular_perturbation_size": 0.1,
    "ambient_perturbation_size": 0.1,
    "diffuse_perturbation_size": 0.1,
}

# -------------------------------------------
# ----------- Preset Collection  ------------
# -------------------------------------------

# All available preset names mapped to their corresponding arguments for env.reset().
# Used for the custom_reset_config argument of RobosuiteAdapter.
ALL_PRESET_ARGUMENTS = {
    "SECANT": dict(
        custom_texture="SECANT",
        custom_color="SECANT",
        custom_camera="SECANT",
        custom_light="SECANT",
    ),
}

# All available preset names for texture/color/camera/lighting mapped to their configs.
ALL_TEXTURE_PRESETS = {
    "SECANT": PRESET_TEXTURE_CONFIG,
}
ALL_COLOR_PRESETS = {"SECANT": PRESET_COLOR_CONFIG}
ALL_CAMERA_PRESETS = {"SECANT": PRESET_CAMERA_CONFIG}
ALL_LIGHTING_PRESETS = {"SECANT": PRESET_LIGHTING_CONFIG}


def get_custom_reset_config(task, mode, scene_id):
    custom_seed = TASK_RANDOM_SEED[task][mode][scene_id]
    custom_texture = custom_color = custom_camera = custom_light = custom_seed
    custom_reset_config = {}
    if custom_texture:
        custom_texture_config = copy.deepcopy(PRESET_TEXTURE_CONFIG)
        custom_texture_config["tex_candidate"] = TASK_TEX_CANDIDATE[task][mode][
            scene_id
        ]
        custom_texture_config["seed"] = custom_texture
        custom_reset_config["custom_texture"] = custom_texture_config
    if custom_color:
        custom_color_config = copy.deepcopy(PRESET_COLOR_CONFIG)
        custom_color_config["seed"] = custom_color
        custom_reset_config["custom_color"] = custom_color_config
    if custom_camera:
        custom_camera_config = copy.deepcopy(PRESET_CAMERA_CONFIG)
        custom_camera_config["seed"] = custom_camera
        custom_reset_config["custom_camera"] = custom_camera_config
    if custom_light:
        custom_light_config = copy.deepcopy(PRESET_LIGHTING_CONFIG)
        custom_light_config["randomize_active"] = False
        custom_light_config["seed"] = custom_light
        custom_reset_config["custom_light"] = custom_light_config
    if not custom_reset_config:
        custom_reset_config = None
    return custom_reset_config


"""
Default texture config by robosuite:
    wall: PlasterCream
    floor: WoodTiles
    table: Ceramic
    door: WoodDark
    pp_table1: WoodLight
    pp_table2: WoodDark

Texture candidate for domain randomization
"""

DR_TEX_CANDIDATE = {
    "wall": ["PlasterPink", "PlasterYellow", "PlasterGray", "PlasterWhite", "Clay"],
    "floor": [
        "PlasterPink",
        "PlasterYellow",
        "PlasterGray",
        "PlasterWhite",
        "PlasterCream",
        "WoodDark",
        "WoodLight",
        "SteelBrushed",
        "SteelScratched",
        "Brass",
    ],
    "table": [
        "PlasterPink",
        "PlasterYellow",
        "PlasterGray",
        "PlasterWhite",
        "PlasterCream",
        "WoodDark",
        "WoodLight",
        "SteelBrushed",
        "SteelScratched",
        "Brass",
        "Metal",
    ],
    "door": [
        "PlasterPink",
        "PlasterYellow",
        "PlasterGray",
        "PlasterWhite",
        "PlasterCream",
        "SteelBrushed",
        "SteelScratched",
        "Brass",
        "Metal",
    ],
    "pp_table": [
        "PlasterPink",
        "PlasterYellow",
        "PlasterGray",
        "PlasterWhite",
        "PlasterCream",
        "SteelBrushed",
        "SteelScratched",
        "Brass",
        "Metal",
    ],
}

EVAL_TEX_CANDIDATE = {
    "eval-easy": [
        "WoodGreen",
        "WoodRed",
        "WoodBlue",
        "WoodgrainGray",
        "FeltGray",
        "Dirt",
    ]
    + [f"Custom{i:02d}" for i in range(1, 6)],
    "eval-hard": ["WoodPanels", "BricksWhite"]
    + [f"Custom{i:02d}" for i in range(6, 26)],
    "eval-extreme": [f"Custom{i:02d}" for i in range(26, 41)],
}

TASK_RANDOM_SEED = {
    "Door": {
        "train": [0, 2, 6, 8, 12, 13, 15, 16, 19, 23, 25],
        "eval-easy": [105, 106, 113, 114, 120, 130, 132, 163, 224, 281],
        "eval-hard": [16, 22, 102, 171, 190, 235, 309, 338, 343, 369],
        "eval-extreme": [26, 29, 31, 100, 112, 148, 229, 275, 291, 369],
    },
    "TwoArmPegInHole": {
        "train": [0, 1, 3, 6, 7, 22, 14, 16, 17, 19, 11],
        "eval-easy": [26, 35, 57, 95, 101, 108, 130, 271, 324, 359],
        "eval-hard": [110, 113, 116, 127, 131, 174, 219, 267, 283, 421],
        "eval-extreme": [11, 101, 108, 122, 132, 145, 166, 327, 349, 429],
    },
    "NutAssemblyRound": {
        "train": [0, 1, 4, 6, 8, 9, 15, 16, 20, 22, 23],
        "eval-easy": [36, 57, 66, 82, 107, 110, 113, 252, 310, 369],
        "eval-hard": [58, 66, 105, 119, 128, 183, 255, 287, 303, 374],
        "eval-extreme": [12, 22, 42, 67, 77, 91, 192, 195, 255, 279],
    },
    "TwoArmLift": {
        "train": [0, 3, 10, 12, 14, 18, 20, 22, 23, 24, 25],
        "eval-easy": [77, 119, 149, 207, 237, 214, 221, 245, 479, 540],
        "eval-hard": [17, 52, 56, 66, 84, 94, 99, 123, 200, 380],
        "eval-extreme": [26, 41, 49, 59, 68, 163, 244, 246, 257, 550],
    },
}

TASK_TEX_CANDIDATE = {
    "Door": {
        "train": [
            {
                "floor": "WoodTiles",
                "wall": "PlasterCream",
                "table_legs": "SteelBrushed",
                "table": "Ceramic",
                "door": "WoodDark",
                "door_handle": "Brass",
            },
            {
                "floor": "WoodDark",
                "wall": "PlasterGray",
                "table_legs": "Metal",
                "table": "PlasterGray",
                "door": "SteelScratched",
                "door_handle": "SteelScratched",
            },
            {
                "floor": "Brass",
                "wall": "PlasterYellow",
                "table_legs": "SteelScratched",
                "table": "WoodLight",
                "door": "PlasterYellow",
                "door_handle": "Metal",
            },
            {
                "floor": "PlasterYellow",
                "wall": "PlasterWhite",
                "table_legs": "Metal",
                "table": "SteelScratched",
                "door": "PlasterWhite",
                "door_handle": "Metal",
            },
            {
                "floor": "WoodDark",
                "wall": "PlasterGray",
                "table_legs": "SteelScratched",
                "table": "PlasterYellow",
                "door": "Brass",
                "door_handle": "Metal",
            },
            {
                "floor": "WoodLight",
                "wall": "PlasterWhite",
                "table_legs": "SteelScratched",
                "table": "PlasterGray",
                "door": "Metal",
                "door_handle": "Brass",
            },
            {
                "floor": "SteelScratched",
                "wall": "PlasterPink",
                "table_legs": "Metal",
                "table": "PlasterCream",
                "door": "PlasterYellow",
                "door_handle": "Metal",
            },
            {
                "floor": "Brass",
                "wall": "PlasterYellow",
                "table_legs": "SteelBrushed",
                "table": "WoodDark",
                "door": "PlasterGray",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "PlasterGray",
                "wall": "Clay",
                "table_legs": "SteelBrushed",
                "table": "SteelScratched",
                "door": "SteelBrushed",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "WoodLight",
                "wall": "PlasterWhite",
                "table_legs": "SteelScratched",
                "table": "PlasterYellow",
                "door": "PlasterPink",
                "door_handle": "Brass",
            },
            {
                "floor": "PlasterWhite",
                "wall": "PlasterPink",
                "table_legs": "SteelBrushed",
                "table": "Metal",
                "door": "SteelBrushed",
                "door_handle": "Metal",
            },
        ],
        "eval-easy": [
            {
                "floor": "Custom03",
                "wall": "Custom05",
                "table_legs": "Metal",
                "table": "Custom02",
                "door": "Custom04",
                "door_handle": "Metal",
            },
            {
                "floor": "WoodRed",
                "wall": "WoodgrainGray",
                "table_legs": "SteelScratched",
                "table": "Custom04",
                "door": "Dirt",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "Custom02",
                "wall": "WoodGreen",
                "table_legs": "SteelBrushed",
                "table": "FeltGray",
                "door": "WoodRed",
                "door_handle": "Brass",
            },
            {
                "floor": "WoodGreen",
                "wall": "FeltGray",
                "table_legs": "SteelScratched",
                "table": "Dirt",
                "door": "Custom02",
                "door_handle": "SteelScratched",
            },
            {
                "floor": "WoodgrainGray",
                "wall": "Custom05",
                "table_legs": "Metal",
                "table": "WoodBlue",
                "door": "Custom01",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "Custom05",
                "wall": "WoodRed",
                "table_legs": "SteelBrushed",
                "table": "Custom03",
                "door": "WoodGreen",
                "door_handle": "Metal",
            },
            {
                "floor": "FeltGray",
                "wall": "Dirt",
                "table_legs": "SteelBrushed",
                "table": "Custom04",
                "door": "Custom02",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "Custom03",
                "wall": "WoodgrainGray",
                "table_legs": "SteelBrushed",
                "table": "Dirt",
                "door": "WoodGreen",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "Custom01",
                "wall": "WoodBlue",
                "table_legs": "Metal",
                "table": "Custom03",
                "door": "WoodGreen",
                "door_handle": "Metal",
            },
            {
                "floor": "Custom04",
                "wall": "Custom03",
                "table_legs": "Metal",
                "table": "WoodBlue",
                "door": "Dirt",
                "door_handle": "SteelBrushed",
            },
        ],
        "eval-hard": [
            {
                "floor": "Custom23",
                "wall": "Custom10",
                "table_legs": "SteelBrushed",
                "table": "Custom16",
                "door": "Custom19",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "Custom19",
                "wall": "Custom22",
                "table_legs": "SteelBrushed",
                "table": "Custom07",
                "door": "Custom09",
                "door_handle": "Brass",
            },
            {
                "floor": "Custom16",
                "wall": "Custom20",
                "table_legs": "Metal",
                "table": "BricksWhite",
                "door": "Custom07",
                "door_handle": "SteelScratched",
            },
            {
                "floor": "Custom07",
                "wall": "Custom09",
                "table_legs": "SteelBrushed",
                "table": "Custom10",
                "door": "Custom19",
                "door_handle": "Brass",
            },
            {
                "floor": "BricksWhite",
                "wall": "Custom20",
                "table_legs": "SteelScratched",
                "table": "Custom06",
                "door": "Custom24",
                "door_handle": "Brass",
            },
            {
                "floor": "WoodPanels",
                "wall": "Custom14",
                "table_legs": "Metal",
                "table": "Custom08",
                "door": "Custom11",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "Custom17",
                "wall": "Custom19",
                "table_legs": "SteelBrushed",
                "table": "Custom10",
                "door": "Custom18",
                "door_handle": "SteelScratched",
            },
            {
                "floor": "Custom10",
                "wall": "Custom15",
                "table_legs": "Metal",
                "table": "Custom22",
                "door": "Custom12",
                "door_handle": "Metal",
            },
            {
                "floor": "Custom12",
                "wall": "Custom25",
                "table_legs": "SteelBrushed",
                "table": "Custom14",
                "door": "Custom19",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "Custom08",
                "wall": "Custom15",
                "table_legs": "SteelScratched",
                "table": "Custom09",
                "door": "Custom13",
                "door_handle": "Metal",
            },
        ],
        "eval-extreme": [
            {
                "floor": "Custom35",
                "wall": "Custom40",
                "table_legs": "SteelBrushed",
                "table": "Custom34",
                "door": "Custom37",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "Custom37",
                "wall": "Custom39",
                "table_legs": "SteelBrushed",
                "table": "Custom30",
                "door": "Custom36",
                "door_handle": "SteelScratched",
            },
            {
                "floor": "Custom29",
                "wall": "Custom27",
                "table_legs": "SteelScratched",
                "table": "Custom34",
                "door": "Custom31",
                "door_handle": "SteelScratched",
            },
            {
                "floor": "Custom39",
                "wall": "Custom36",
                "table_legs": "Metal",
                "table": "Custom35",
                "door": "Custom32",
                "door_handle": "Brass",
            },
            {
                "floor": "Custom28",
                "wall": "Custom33",
                "table_legs": "Metal",
                "table": "Custom29",
                "door": "Custom39",
                "door_handle": "SteelScratched",
            },
            {
                "floor": "Custom34",
                "wall": "Custom35",
                "table_legs": "SteelScratched",
                "table": "Custom37",
                "door": "Custom32",
                "door_handle": "SteelScratched",
            },
            {
                "floor": "Custom36",
                "wall": "Custom28",
                "table_legs": "SteelBrushed",
                "table": "Custom30",
                "door": "Custom37",
                "door_handle": "Metal",
            },
            {
                "floor": "Custom32",
                "wall": "Custom40",
                "table_legs": "SteelScratched",
                "table": "Custom31",
                "door": "Custom34",
                "door_handle": "SteelBrushed",
            },
            {
                "floor": "Custom30",
                "wall": "Custom33",
                "table_legs": "SteelBrushed",
                "table": "Custom34",
                "door": "Custom32",
                "door_handle": "Metal",
            },
            {
                "floor": "Custom26",
                "wall": "Custom33",
                "table_legs": "SteelScratched",
                "table": "Custom27",
                "door": "Custom29",
                "door_handle": "Metal",
            },
        ],
    },
    "TwoArmPegInHole": {
        "train": [
            {
                "floor": "WoodTiles",
                "wall": "PlasterCream",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "PlasterCream",
                "wall": "PlasterYellow",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "WoodLight",
                "wall": "PlasterWhite",
                "ta_pih_plate": "WoodBlue",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Brass",
                "wall": "PlasterYellow",
                "ta_pih_plate": "WoodBlue",
                "ta_pih_stick": "WoodRed",
            },
            {
                "floor": "PlasterPink",
                "wall": "PlasterGray",
                "ta_pih_plate": "WoodGreen",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "WoodDark",
                "wall": "PlasterGray",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "SteelBrushed",
                "wall": "Clay",
                "ta_pih_plate": "WoodGreen",
                "ta_pih_stick": "WoodRed",
            },
            {
                "floor": "Brass",
                "wall": "PlasterYellow",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "PlasterPink",
                "wall": "PlasterGray",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "PlasterGray",
                "wall": "Clay",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "PlasterWhite",
                "wall": "PlasterPink",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
        ],
        "eval-easy": [
            {
                "floor": "Custom03",
                "wall": "WoodgrainGray",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "WoodgrainGray",
                "wall": "Custom03",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "FeltGray",
                "wall": "WoodBlue",
                "ta_pih_plate": "WoodBlue",
                "ta_pih_stick": "WoodRed",
            },
            {
                "floor": "WoodGreen",
                "wall": "Dirt",
                "ta_pih_plate": "WoodBlue",
                "ta_pih_stick": "WoodRed",
            },
            {
                "floor": "FeltGray",
                "wall": "Custom05",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Dirt",
                "wall": "WoodBlue",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "Custom05",
                "wall": "WoodRed",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "Custom01",
                "wall": "Custom03",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "WoodgrainGray",
                "wall": "Custom01",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "WoodBlue",
                "wall": "WoodgrainGray",
                "ta_pih_plate": "WoodGreen",
                "ta_pih_stick": "WoodRed",
            },
        ],
        "eval-hard": [
            {
                "floor": "Custom08",
                "wall": "Custom11",
                "ta_pih_plate": "WoodGreen",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "Custom15",
                "wall": "BricksWhite",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Custom19",
                "wall": "Custom10",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "Custom20",
                "wall": "Custom17",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Custom21",
                "wall": "Custom10",
                "ta_pih_plate": "WoodBlue",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "WoodPanels",
                "wall": "Custom07",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "Custom14",
                "wall": "Custom12",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Custom22",
                "wall": "Custom12",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Custom16",
                "wall": "Custom10",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Custom17",
                "wall": "Custom07",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
        ],
        "eval-extreme": [
            {
                "floor": "Custom32",
                "wall": "Custom31",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "Custom38",
                "wall": "Custom31",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Custom26",
                "wall": "Custom31",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "Custom34",
                "wall": "Custom29",
                "ta_pih_plate": "WoodBlue",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Custom30",
                "wall": "Custom37",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Custom35",
                "wall": "Custom26",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "Custom33",
                "wall": "Custom38",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "Custom37",
                "wall": "Custom28",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
            {
                "floor": "Custom31",
                "wall": "Custom26",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodBlue",
            },
            {
                "floor": "Custom29",
                "wall": "Custom35",
                "ta_pih_plate": "WoodRed",
                "ta_pih_stick": "WoodGreen",
            },
        ],
    },
    "NutAssemblyRound": {
        "train": [
            {
                "floor": "WoodTiles",
                "wall": "PlasterCream",
                "table_legs": "SteelBrushed",
                "table": "Ceramic",
                "na_metal1": "SteelScratched",
                "na_metal2": "Brass",
            },
            {
                "floor": "PlasterCream",
                "wall": "PlasterYellow",
                "table_legs": "Metal",
                "table": "PlasterYellow",
                "na_metal1": "WoodBlue",
                "na_metal2": "WoodGreen",
            },
            {
                "floor": "SteelBrushed",
                "wall": "Clay",
                "table_legs": "Metal",
                "table": "PlasterCream",
                "na_metal1": "Metal",
                "na_metal2": "SteelBrushed",
            },
            {
                "floor": "Brass",
                "wall": "PlasterYellow",
                "table_legs": "Metal",
                "table": "WoodLight",
                "na_metal1": "Metal",
                "na_metal2": "SteelBrushed",
            },
            {
                "floor": "PlasterYellow",
                "wall": "PlasterWhite",
                "table_legs": "SteelBrushed",
                "table": "SteelScratched",
                "na_metal1": "WoodGreen",
                "na_metal2": "WoodBlue",
            },
            {
                "floor": "PlasterGray",
                "wall": "Clay",
                "table_legs": "SteelScratched",
                "table": "Brass",
                "na_metal1": "WoodRed",
                "na_metal2": "Brass",
            },
            {
                "floor": "SteelScratched",
                "wall": "PlasterPink",
                "table_legs": "SteelBrushed",
                "table": "PlasterCream",
                "na_metal1": "Brass",
                "na_metal2": "WoodBlue",
            },
            {
                "floor": "Brass",
                "wall": "PlasterYellow",
                "table_legs": "SteelBrushed",
                "table": "WoodDark",
                "na_metal1": "SteelBrushed",
                "na_metal2": "Metal",
            },
            {
                "floor": "PlasterWhite",
                "wall": "PlasterPink",
                "table_legs": "Metal",
                "table": "Brass",
                "na_metal1": "WoodGreen",
                "na_metal2": "Metal",
            },
            {
                "floor": "WoodDark",
                "wall": "PlasterGray",
                "table_legs": "Metal",
                "table": "PlasterPink",
                "na_metal1": "WoodBlue",
                "na_metal2": "WoodRed",
            },
            {
                "floor": "PlasterWhite",
                "wall": "PlasterPink",
                "table_legs": "Metal",
                "table": "Metal",
                "na_metal1": "SteelBrushed",
                "na_metal2": "WoodBlue",
            },
        ],
        "eval-easy": [
            {
                "floor": "Custom02",
                "wall": "Custom01",
                "table_legs": "SteelScratched",
                "table": "WoodGreen",
                "na_metal1": "WoodBlue",
                "na_metal2": "Metal",
            },
            {
                "floor": "FeltGray",
                "wall": "WoodBlue",
                "table_legs": "SteelScratched",
                "table": "WoodgrainGray",
                "na_metal1": "WoodBlue",
                "na_metal2": "Brass",
            },
            {
                "floor": "Custom01",
                "wall": "Custom04",
                "table_legs": "SteelScratched",
                "table": "Custom05",
                "na_metal1": "WoodRed",
                "na_metal2": "WoodBlue",
            },
            {
                "floor": "FeltGray",
                "wall": "Custom01",
                "table_legs": "SteelBrushed",
                "table": "WoodRed",
                "na_metal1": "WoodGreen",
                "na_metal2": "Metal",
            },
            {
                "floor": "Dirt",
                "wall": "WoodBlue",
                "table_legs": "SteelScratched",
                "table": "Custom05",
                "na_metal1": "Brass",
                "na_metal2": "Metal",
            },
            {
                "floor": "Custom01",
                "wall": "WoodBlue",
                "table_legs": "SteelBrushed",
                "table": "FeltGray",
                "na_metal1": "Brass",
                "na_metal2": "WoodRed",
            },
            {
                "floor": "Custom02",
                "wall": "WoodGreen",
                "table_legs": "SteelScratched",
                "table": "FeltGray",
                "na_metal1": "WoodBlue",
                "na_metal2": "WoodRed",
            },
            {
                "floor": "Custom03",
                "wall": "FeltGray",
                "table_legs": "SteelScratched",
                "table": "WoodBlue",
                "na_metal1": "WoodGreen",
                "na_metal2": "SteelBrushed",
            },
            {
                "floor": "FeltGray",
                "wall": "Dirt",
                "table_legs": "SteelBrushed",
                "table": "Custom02",
                "na_metal1": "WoodRed",
                "na_metal2": "WoodBlue",
            },
            {
                "floor": "WoodgrainGray",
                "wall": "Custom02",
                "table_legs": "SteelScratched",
                "table": "WoodRed",
                "na_metal1": "WoodBlue",
                "na_metal2": "Metal",
            },
        ],
        "eval-hard": [
            {
                "floor": "Custom13",
                "wall": "Custom16",
                "table_legs": "SteelBrushed",
                "table": "Custom11",
                "na_metal1": "WoodGreen",
                "na_metal2": "Brass",
            },
            {
                "floor": "Custom09",
                "wall": "Custom13",
                "table_legs": "SteelScratched",
                "table": "BricksWhite",
                "na_metal1": "WoodRed",
                "na_metal2": "WoodBlue",
            },
            {
                "floor": "Custom09",
                "wall": "Custom16",
                "table_legs": "Metal",
                "table": "Custom06",
                "na_metal1": "Brass",
                "na_metal2": "Metal",
            },
            {
                "floor": "WoodPanels",
                "wall": "Custom22",
                "table_legs": "SteelScratched",
                "table": "Custom09",
                "na_metal1": "Metal",
                "na_metal2": "WoodGreen",
            },
            {
                "floor": "Custom10",
                "wall": "Custom20",
                "table_legs": "SteelBrushed",
                "table": "Custom07",
                "na_metal1": "Metal",
                "na_metal2": "WoodRed",
            },
            {
                "floor": "Custom10",
                "wall": "Custom24",
                "table_legs": "SteelBrushed",
                "table": "Custom14",
                "na_metal1": "WoodBlue",
                "na_metal2": "WoodRed",
            },
            {
                "floor": "Custom22",
                "wall": "Custom21",
                "table_legs": "SteelBrushed",
                "table": "Custom15",
                "na_metal1": "WoodGreen",
                "na_metal2": "SteelBrushed",
            },
            {
                "floor": "Custom07",
                "wall": "BricksWhite",
                "table_legs": "SteelScratched",
                "table": "Custom10",
                "na_metal1": "WoodGreen",
                "na_metal2": "WoodRed",
            },
            {
                "floor": "Custom17",
                "wall": "Custom18",
                "table_legs": "SteelScratched",
                "table": "Custom21",
                "na_metal1": "WoodGreen",
                "na_metal2": "WoodBlue",
            },
            {
                "floor": "Custom13",
                "wall": "Custom23",
                "table_legs": "SteelScratched",
                "table": "BricksWhite",
                "na_metal1": "WoodRed",
                "na_metal2": "Brass",
            },
        ],
        "eval-extreme": [
            {
                "floor": "Custom31",
                "wall": "Custom39",
                "table_legs": "SteelScratched",
                "table": "Custom38",
                "na_metal1": "WoodGreen",
                "na_metal2": "WoodRed",
            },
            {
                "floor": "Custom27",
                "wall": "Custom37",
                "table_legs": "Metal",
                "table": "Custom33",
                "na_metal1": "WoodBlue",
                "na_metal2": "WoodRed",
            },
            {
                "floor": "Custom39",
                "wall": "Custom34",
                "table_legs": "SteelScratched",
                "table": "Custom35",
                "na_metal1": "WoodGreen",
                "na_metal2": "WoodBlue",
            },
            {
                "floor": "Custom30",
                "wall": "Custom37",
                "table_legs": "SteelBrushed",
                "table": "Custom28",
                "na_metal1": "WoodGreen",
                "na_metal2": "Metal",
            },
            {
                "floor": "Custom35",
                "wall": "Custom38",
                "table_legs": "SteelBrushed",
                "table": "Custom40",
                "na_metal1": "WoodRed",
                "na_metal2": "WoodBlue",
            },
            {
                "floor": "Custom27",
                "wall": "Custom35",
                "table_legs": "SteelScratched",
                "table": "Custom32",
                "na_metal1": "Metal",
                "na_metal2": "WoodGreen",
            },
            {
                "floor": "Custom26",
                "wall": "Custom30",
                "table_legs": "SteelScratched",
                "table": "Custom29",
                "na_metal1": "Metal",
                "na_metal2": "SteelBrushed",
            },
            {
                "floor": "Custom32",
                "wall": "Custom27",
                "table_legs": "SteelBrushed",
                "table": "Custom36",
                "na_metal1": "SteelBrushed",
                "na_metal2": "WoodRed",
            },
            {
                "floor": "Custom26",
                "wall": "Custom30",
                "table_legs": "SteelBrushed",
                "table": "Custom37",
                "na_metal1": "WoodGreen",
                "na_metal2": "SteelBrushed",
            },
            {
                "floor": "Custom36",
                "wall": "Custom26",
                "table_legs": "Metal",
                "table": "Custom39",
                "na_metal1": "SteelBrushed",
                "na_metal2": "WoodGreen",
            },
        ],
    },
    "TwoArmLift": {
        "train": [
            {
                "floor": "WoodTiles",
                "wall": "PlasterCream",
                "table_legs": "SteelBrushed",
                "table": "Ceramic",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "WoodLight",
                "wall": "PlasterWhite",
                "table_legs": "SteelBrushed",
                "table": "PlasterWhite",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "PlasterWhite",
                "wall": "PlasterPink",
                "table_legs": "Metal",
                "table": "Metal",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodBlue",
                "ta_lift_handle2": "WoodGreen",
            },
            {
                "floor": "WoodDark",
                "wall": "PlasterGray",
                "table_legs": "SteelScratched",
                "table": "PlasterYellow",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "SteelBrushed",
                "wall": "Clay",
                "table_legs": "SteelBrushed",
                "table": "PlasterWhite",
                "ta_lift_pot": "WoodGreen",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "PlasterYellow",
                "wall": "PlasterWhite",
                "table_legs": "SteelBrushed",
                "table": "SteelBrushed",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodGreen",
            },
            {
                "floor": "PlasterWhite",
                "wall": "PlasterPink",
                "table_legs": "Metal",
                "table": "Brass",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "WoodDark",
                "wall": "PlasterGray",
                "table_legs": "Metal",
                "table": "PlasterPink",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "WoodLight",
                "wall": "PlasterWhite",
                "table_legs": "SteelBrushed",
                "table": "PlasterYellow",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "SteelBrushed",
                "wall": "Clay",
                "table_legs": "SteelBrushed",
                "table": "PlasterGray",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodGreen",
            },
            {
                "floor": "PlasterWhite",
                "wall": "PlasterPink",
                "table_legs": "Metal",
                "table": "Metal",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodBlue",
                "ta_lift_handle2": "WoodGreen",
            },
        ],
        "eval-easy": [
            {
                "floor": "WoodRed",
                "wall": "WoodGreen",
                "table_legs": "SteelBrushed",
                "table": "WoodBlue",
                "ta_lift_pot": "WoodGreen",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "WoodGreen",
                "wall": "WoodRed",
                "table_legs": "SteelScratched",
                "table": "Dirt",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "Custom03",
                "wall": "WoodBlue",
                "table_legs": "SteelBrushed",
                "table": "WoodgrainGray",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "FeltGray",
                "wall": "WoodRed",
                "table_legs": "SteelBrushed",
                "table": "Custom03",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "Dirt",
                "wall": "Custom03",
                "table_legs": "Metal",
                "table": "FeltGray",
                "ta_lift_pot": "WoodGreen",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "WoodBlue",
                "wall": "WoodGreen",
                "table_legs": "SteelScratched",
                "table": "Custom05",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodBlue",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "Custom05",
                "wall": "WoodgrainGray",
                "table_legs": "SteelScratched",
                "table": "Custom02",
                "ta_lift_pot": "WoodGreen",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "Custom04",
                "wall": "Custom01",
                "table_legs": "SteelScratched",
                "table": "Custom03",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodBlue",
                "ta_lift_handle2": "WoodGreen",
            },
            {
                "floor": "Custom04",
                "wall": "Custom03",
                "table_legs": "Metal",
                "table": "WoodGreen",
                "ta_lift_pot": "WoodGreen",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "Custom03",
                "wall": "Custom05",
                "table_legs": "SteelScratched",
                "table": "WoodRed",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodBlue",
            },
        ],
        "eval-hard": [
            {
                "floor": "Custom23",
                "wall": "Custom07",
                "table_legs": "SteelScratched",
                "table": "WoodPanels",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodBlue",
                "ta_lift_handle2": "WoodGreen",
            },
            {
                "floor": "Custom11",
                "wall": "Custom22",
                "table_legs": "SteelBrushed",
                "table": "Custom10",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "Custom21",
                "wall": "Custom22",
                "table_legs": "Metal",
                "table": "Custom09",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "Custom09",
                "wall": "Custom13",
                "table_legs": "SteelScratched",
                "table": "BricksWhite",
                "ta_lift_pot": "WoodGreen",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "Custom15",
                "wall": "Custom22",
                "table_legs": "Metal",
                "table": "Custom07",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "Custom08",
                "wall": "Custom25",
                "table_legs": "SteelBrushed",
                "table": "Custom15",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "Custom14",
                "wall": "Custom15",
                "table_legs": "Metal",
                "table": "Custom21",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodBlue",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "Custom23",
                "wall": "Custom20",
                "table_legs": "SteelScratched",
                "table": "Custom08",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodGreen",
            },
            {
                "floor": "Custom09",
                "wall": "Custom22",
                "table_legs": "Metal",
                "table": "Custom06",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodGreen",
            },
            {
                "floor": "Custom17",
                "wall": "Custom11",
                "table_legs": "SteelScratched",
                "table": "Custom13",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodRed",
            },
        ],
        "eval-extreme": [
            {
                "floor": "Custom35",
                "wall": "Custom40",
                "table_legs": "Metal",
                "table": "Custom34",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "Custom36",
                "wall": "Custom37",
                "table_legs": "Metal",
                "table": "Custom33",
                "ta_lift_pot": "WoodGreen",
                "ta_lift_handle1": "WoodBlue",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "Custom27",
                "wall": "Custom40",
                "table_legs": "SteelBrushed",
                "table": "Custom29",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodGreen",
            },
            {
                "floor": "Custom30",
                "wall": "Custom34",
                "table_legs": "SteelBrushed",
                "table": "Custom28",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "Custom29",
                "wall": "Custom35",
                "table_legs": "SteelScratched",
                "table": "Custom40",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodGreen",
            },
            {
                "floor": "Custom36",
                "wall": "Custom31",
                "table_legs": "SteelScratched",
                "table": "Custom38",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodBlue",
            },
            {
                "floor": "Custom35",
                "wall": "Custom32",
                "table_legs": "Metal",
                "table": "Custom39",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "Custom37",
                "wall": "Custom27",
                "table_legs": "SteelScratched",
                "table": "Custom32",
                "ta_lift_pot": "WoodGreen",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodRed",
            },
            {
                "floor": "Custom27",
                "wall": "Custom30",
                "table_legs": "SteelBrushed",
                "table": "Custom36",
                "ta_lift_pot": "WoodBlue",
                "ta_lift_handle1": "WoodRed",
                "ta_lift_handle2": "WoodGreen",
            },
            {
                "floor": "Custom33",
                "wall": "Custom35",
                "table_legs": "Metal",
                "table": "Custom30",
                "ta_lift_pot": "WoodRed",
                "ta_lift_handle1": "WoodGreen",
                "ta_lift_handle2": "WoodBlue",
            },
        ],
    },
}
