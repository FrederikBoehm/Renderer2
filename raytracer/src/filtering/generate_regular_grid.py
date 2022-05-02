import json
from common_functions import getMaxBounds, getCamera, toFile
import numpy as np
import math

def generateRegularGrid():
    with open('relevant_models.json', 'r') as f:
        relevant_models_dict = json.load(f)

    with open('../../../build/raytracer/filtering/volume_description.json', 'r') as f:
        volume_description_dict = json.load(f)

    volume_max_bounds = getMaxBounds(volume_description_dict)
    camera = getCamera([-30.0, 30.0, 30.0], [0, 30, 0], [0.0, 1.0, 0.0], 1920, 1080, 90)

    relevant_models = []
    relevant_models.extend(relevant_models_dict["Large"])
    relevant_models.extend(relevant_models_dict["Medium"])
    relevant_models.extend(relevant_models_dict["Small"])
    relevant_models = sorted(relevant_models, key=lambda v: v["ExpectedHeight"], reverse=True)

    #Calculate front length to roughly center scene around origin
    grid_front_length = 0
    for model in relevant_models:
        model_name = model["Name"]
        scaling = model["ExpectedHeight"] / model["Dimensions"][1]
        bbmin = np.array(volume_max_bounds[model_name]["BBMin"])
        bbmax = np.array(volume_max_bounds[model_name]["BBMax"])
        model_dimensions = scaling * (bbmax - bbmin)
        bounding_radius = math.sqrt((model_dimensions[0] / 2) ** 2 + (model_dimensions[2] / 2) ** 2)
        grid_front_length += 2 * bounding_radius

    cam_pos = camera["Pos"]
    grid_direction_x = np.cross(np.array([0, 1, 0]), cam_pos)
    grid_direction_x /= np.linalg.norm(grid_direction_x)
    grid_direction_z = -np.array([cam_pos[0], 0, cam_pos[2]])
    grid_direction_z /= np.linalg.norm(grid_direction_z)

    current_pos = -grid_direction_x * grid_front_length * 0.5
    scene_description = []
    for model in relevant_models:
        model_name = model["Name"]
        scaling = model["ExpectedHeight"] / model["Dimensions"][1]
        bbmin = np.array(volume_max_bounds[model_name]["BBMin"])
        bbmax = np.array(volume_max_bounds[model_name]["BBMax"])
        model_dimensions = scaling * (bbmax - bbmin)
        bounding_radius = math.sqrt((model_dimensions[0] / 2) ** 2 + (model_dimensions[2] / 2) ** 2)
        bounding_circle_pos = current_pos + grid_direction_x * bounding_radius

        model_pos = (2.0 * bounding_circle_pos - scaling * (bbmin + bbmax)) * 0.5
        model_pos[1] = 0

        scene_description.append((
            "Mesh",
            "../../raytracer/assets/" + model["Path"],
            model["Name"] + ".obj",
            list(model_pos),
            model["Orientation"],
            [scaling, scaling, scaling]
        ))

        lods = volume_description_dict[model_name]
        for lod_size in sorted(lods):
            bounding_circle_pos = bounding_circle_pos + 2.0 * bounding_radius * grid_direction_z
            model_pos = (2.0 * bounding_circle_pos - scaling * (bbmin + bbmax)) * 0.5
            model_pos[1] = 0
            scene_description.append((
                "Medium",
                "./filtering/" + model_name + "/",
                "filtered_mesh_" + lod_size,
                list(model_pos),
                model["Orientation"],
                [scaling, scaling, scaling]
            ))

        current_pos += grid_direction_x * 2 * bounding_radius + 0.0001

    toFile(scene_description, "scenedescription_regular_grid.json")
    print()

if __name__ == "__main__":
    generateRegularGrid()