import json
import numpy as np
import math
from scipy.spatial import ConvexHull
import argparse
from common_functions import getMaxBounds, getCamera, toFile

RADIUS = 1000
ACCEL_CLASSES = 10

def vec4(vec, v):
    return np.array([vec[0], vec[1], vec[2], v])

def uniformSampleCircle(radius):
    r = math.sqrt(np.random.uniform()) * radius
    theta = 2 * math.pi * np.random.uniform()
    return np.array([r * math.cos(theta), 0, r * math.sin(theta)])

def getTreeDescription(relevant_models, index, volume_descriptions, volume_max_bounds, camera, generate_gt):
    model = relevant_models[index]
    model_name = model["Name"]
    model_space_height = model["Dimensions"][1]
    expected = model["ExpectedHeight"]
    randomHeight = np.random.normal(expected, expected * 0.01)
    scaling = randomHeight / model_space_height
    bounding_circle_pos = uniformSampleCircle(RADIUS)
    bbmin = np.array(volume_max_bounds[model_name]["BBMin"])
    bbmax = np.array(volume_max_bounds[model_name]["BBMax"])
    # bbmin = np.array(model["BBMin"])
    # bbmax = np.array(model["BBMax"])
    model_pos = (2.0 * bounding_circle_pos - scaling * (bbmin + bbmax)) * 0.5
    model_pos[1] = 0
    model_dimensions = scaling * (bbmax - bbmin)
    bounding_radius = math.sqrt((model_dimensions[0] / 2) ** 2 + (model_dimensions[2] / 2) ** 2)


    lod = selectLOD(camera, model_pos, scaling, volume_descriptions[model["Name"]], generate_gt)
    if lod == "0":
    # if True:
        # dimensions = np.array(model["Dimensions"]) * scaling

        description = {
            "Type": "Mesh",
            "Directory": "../../raytracer/assets/" + model["Path"],
            "Filename": model["Name"] + ".obj",
            "Pos": model_pos,
            "BoundingCirclePos": bounding_circle_pos,
            "Orientation": model["Orientation"],
            "Scaling": [scaling, scaling, scaling],
            "BoundingRadius": bounding_radius
        }
        return description
    else:
        description = {
            "Type": "Medium",
            "Directory": "./filtering/" + model_name + "/",
            "Filename": "filtered_mesh_" + lod + ".nvdb",
            "Pos": model_pos,
            "BoundingCirclePos": bounding_circle_pos,
            "Orientation": model["Orientation"],
            "Scaling": [scaling, scaling, scaling],
            "BoundingRadius": bounding_radius
        }
        return description
        


def project(p, camera):
    d = camera["NearPlaneDistance"]
    cam_pos = camera["Pos"]
    # z = cam_pos[2] - p[2]
    z = -p[2]
    x = d / z * p[0]
    y = d / z * p[1]
    return [x, y, -math.copysign(d, z)] # right handed coordinates

def projectLOD(lod, camera, scaling, tree_pos):
    p0 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P0"]) + np.array(tree_pos), 1.0)), camera)
    p1 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P1"]) + np.array(tree_pos), 1.0)), camera)
    p2 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P2"]) + np.array(tree_pos), 1.0)), camera)
    p3 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P3"]) + np.array(tree_pos), 1.0)), camera)
    p4 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P4"]) + np.array(tree_pos), 1.0)), camera)
    p5 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P5"]) + np.array(tree_pos), 1.0)), camera)
    p6 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P6"]) + np.array(tree_pos), 1.0)), camera)
    p7 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P7"]) + np.array(tree_pos), 1.0)), camera)

    return {
        "P0": p0,
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "P4": p4,
        "P5": p5,
        "P6": p6,
        "P7": p7
    }

def outsideFrustum(p, camera):
    pixel_size = camera["PixelSize"]
    return p[2] > 0.0 or math.fabs(p[0]) > camera["SensorWidth"] * 0.5 * pixel_size or math.fabs(p[1]) > camera["SensorHeight"] * 0.5 * pixel_size

def approximateNumVoxels(lod, lod_pos, camera):
    # lod_pos: center of bounding circle
    num_voxels = lod["NumVoxels"]
    return math.ceil(math.pow(num_voxels[0] * num_voxels[1] * num_voxels[2], 2/3))

def countCloseVoxels(lod_size, lod, lod_pos, scaling, camera):
    start_pos = lod_pos + scaling * np.array(lod["BBMin"])
    num_voxels = lod["NumVoxels"]
    lod_size = float(lod_size)
    end_pos = lod_pos + scaling * np.array(lod["BBMax"])
    model_dimension = end_pos - start_pos
    voxel_size2 = model_dimension / np.array(num_voxels)
    voxel_size = np.array([lod_size, lod_size, lod_size])
    plane_origin = (start_pos + end_pos) * 0.5
    plane_normal = camera["Pos"] - plane_origin
    plane_normal /= np.linalg.norm(plane_normal)
    p = -np.dot(plane_origin, plane_normal)
    indices = [np.indices((num_voxels[0],))[0], np.indices((num_voxels[1],))[0], np.indices((num_voxels[2],))[0]]
    joined = np.vstack(np.meshgrid(indices[0], indices[1], indices[2])).reshape(3, -1).T
    positions = start_pos + joined * voxel_size2 + 0.5 * voxel_size2
    distances = np.abs(np.dot(positions, plane_normal) + p)
    num_intersected_voxels = np.sum((distances < voxel_size2[0] * 0.5).astype(int))
    # num_intersected_voxels = 0
    # for x in range(num_voxels[0]):
    #     for y in range(num_voxels[1]):
    #         for z in range(num_voxels[2]):
    #             current_pos = start_pos + np.array([x, y, z]) * voxel_size2 + 0.5 * voxel_size2 # +0.5 to get to center of voxel
    #             distance_to_plane = math.fabs(np.dot(plane_normal, current_pos) + p)
    #             if distance_to_plane < lod_size * 0.5:
    #                 num_intersected_voxels += 1

    return num_intersected_voxels

def gamma(n):
    return (n * np.finfo(np.float32).eps) / (1 - n * np.finfo(np.float32).eps)

def intersectBB(ray_origin, ray_dir, bb_min, bb_max):
    # t0 = 0
    t1 = np.finfo(np.float32).max
    g = gamma(3)
    for i in range(3):
        inv_ray_dir = 1 / ray_dir[i]
        t_near = (bb_min[i] - ray_origin[i]) * inv_ray_dir
        t_far = (bb_max[i] - ray_origin[i]) * inv_ray_dir

        if t_near > t_far:
            temp = t_near
            t_near = t_far
            t_far = temp

        t_far *= 1 + 2 * g
        # t0 = t_near if t_near > t0 else t0
        t1 = t_far if t_far < t1 else t1

    return ray_origin + t1 * ray_dir

def approximateNumVoxels2(lod, lod_pos, scaling, camera):
    bb_min = np.array(lod["BBMin"])
    bb_max = np.array(lod["BBMax"])
    start_pos = lod_pos + scaling * bb_min
    end_pos = lod_pos + scaling * bb_max
    plane_origin = (start_pos + end_pos) * 0.5
    plane_normal = camera["Pos"] - plane_origin
    plane_normal /= np.linalg.norm(plane_normal)

    xz_dir = np.cross(plane_normal, np.array((0.0, 1.0, 0.0)))
    xz_dir /= np.linalg.norm(xz_dir)
    bb_intersection1 = intersectBB(plane_origin, xz_dir, start_pos, end_pos)

    perpendicular = np.cross(xz_dir, plane_normal)
    perpendicular /= np.linalg.norm(perpendicular)
    bb_intersection2 = intersectBB(plane_origin, perpendicular, start_pos, end_pos)

    model_dimension = end_pos - start_pos
    voxel_size = model_dimension / np.array(lod["NumVoxels"])

    num_voxels1 = 2 * (plane_origin - bb_intersection1) / voxel_size
    num_voxels2 = 2 * (plane_origin - bb_intersection2) / voxel_size
    # length1 = np.linalg.norm(plane_origin - bb_intersection1)
    # length2 = np.linalg.norm(plane_origin - bb_intersection2)
    #
    #
    # num_voxels1 = np.linalg.norm(length1 / voxel_size[0])
    # num_voxels2 = np.linalg.norm(length2 / voxel_size)

    return math.ceil(np.linalg.norm(num_voxels1) * np.linalg.norm(num_voxels2))


def selectLOD(camera, tree_pos, scaling, lods, generate_gt):
    if generate_gt:
        return "0"

    # use coarsest volume if outside frustum
    max_lod_size = "0"
    for lod_size in lods:
        if float(lod_size) > float(max_lod_size):
            max_lod_size = lod_size

    max_lod = lods[max_lod_size]
    max_lod_projected = projectLOD(max_lod, camera, scaling, tree_pos)

    points_outside_frustum = 0
    for point_name, point_value in max_lod_projected.items():
        if outsideFrustum(point_value, camera):
            points_outside_frustum += 1

    if points_outside_frustum == 8:
        return max_lod_size

    convex_hull = ConvexHull(np.array([
        [max_lod_projected["P0"][0], max_lod_projected["P0"][1]],
        [max_lod_projected["P1"][0], max_lod_projected["P1"][1]],
        [max_lod_projected["P2"][0], max_lod_projected["P2"][1]],
        [max_lod_projected["P3"][0], max_lod_projected["P3"][1]],
        [max_lod_projected["P4"][0], max_lod_projected["P4"][1]],
        [max_lod_projected["P5"][0], max_lod_projected["P5"][1]],
        [max_lod_projected["P6"][0], max_lod_projected["P6"][1]],
        [max_lod_projected["P7"][0], max_lod_projected["P7"][1]]
    ]))

    pixel_area = camera["PixelSize"] ** 2
    num_covered_pixels = math.ceil(convex_hull.volume / pixel_area) # since ConvexHull is defined in N dimensions, volume gives area in 2d

    best_match = "0"
    for lod_size in sorted(lods):
        # close_voxels = countCloseVoxels(lod_size, lods[lod_size], tree_pos, scaling, camera)
        # approximated1 = approximateNumVoxels(lods[lod_size], None, camera)
        approximated2 = approximateNumVoxels2(lods[lod_size], tree_pos, scaling, camera)
        if approximated2 > num_covered_pixels:
            best_match = lod_size
        else:
            return best_match

    return best_match


    # world_to_view = camera["WorldToView"]
    #
    #
    # lod = lods["0.400000"]
    # view_dir = (camera["LookAt"] - camera["Pos"]) / np.linalg.norm(camera["LookAt"] - camera["Pos"])
    # view_plane_origin = camera["LookAt"] + view_dir * camera["NearPlaneDistance"]
    # view_plane_origin_cam_space = np.dot(camera["WorldToView"], vec4(view_plane_origin, 1.0))
    # scaling = 0.017187480335105507
    # p0_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P0"]), 1.0))
    # p1_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P1"]), 1.0))
    # p2_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P2"]), 1.0))
    # p3_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P3"]), 1.0))
    # p4_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P4"]), 1.0))
    # p5_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P5"]), 1.0))
    # p6_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P6"]), 1.0))
    # p7_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P7"]), 1.0))
    #
    # p0 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P0"]), 1.0)), camera)
    # p1 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P1"]), 1.0)), camera)
    # p2 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P2"]), 1.0)), camera)
    # p3 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P3"]), 1.0)), camera)
    # p4 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P4"]), 1.0)), camera)
    # p5 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P5"]), 1.0)), camera)
    # p6 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P6"]), 1.0)), camera)
    # p7 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P7"]), 1.0)), camera)
    # # world_connection = vec4(lod["P0"], 1.0) - vec4(lod["P7"], 1.0)
    # # view_connection = p0 - p7
    # # world_dist = np.linalg.norm(world_connection)
    # # view_dist = np.linalg.norm(view_connection)
    # # transformed = np.dot(camera["WorldToView"], vec4([0, 0, 1], 1))
    # # transformed2 = np.dot(camera["WorldToView"], vec4([0, 0, 1.00001], 1))
    # return "0.400000"

def checkCollision(object1, object2):
    connection = object1["BoundingCirclePos"] - object2["BoundingCirclePos"]
    distance = np.sqrt(np.dot(connection, connection))

    # return distance < object1["BoundingRadius"] or distance < object2["BoundingRadius"]
    return distance <= object1["BoundingRadius"] + object2["BoundingRadius"]



def getRelevantAccelerationClasses(description):
    pos = description["BoundingCirclePos"]
    distance = np.sqrt(np.dot(pos, pos))
    bounding_radius = description["BoundingRadius"]
    upper_limit = distance + bounding_radius
    lower_limit = distance - bounding_radius

    class_size = RADIUS / ACCEL_CLASSES
    squared_radius = RADIUS ** 2
    min_class = int(ACCEL_CLASSES * lower_limit ** 2 / squared_radius)
    max_class = int(ACCEL_CLASSES * upper_limit ** 2 / squared_radius)
    return list(range(min_class, min(max_class + 1, ACCEL_CLASSES)))



def generateRandomForest(generate_gt):
    # tree_cdf = [0.9, 0.98, 1.0] # large, medium, small
    tree_cdf = [0.0, 1.0, 1.0]
    np.random.seed(0)

    with open('relevant_models.json', 'r') as f:
        relevant_models_dict = json.load(f)

    with open('volume_description.json', 'r') as f:
        volume_description_dict = json.load(f)

    volume_max_bounds = getMaxBounds(volume_description_dict)
    camera = getCamera([ -680.0, 25.1, 680.0 ], [ -675.0, 25.0, 675.0 ], [ 0.0, 1.0, 0.0 ], 1920, 1080, 90)
    # camera = getCamera([ 1.0,  8.114, -1.38300563], [1.71138167,  8.11342003, -1.38300563], [0.0, 1.0, 0.0], 1920, 1080, 90)

    acceleration = []
    for c in range(ACCEL_CLASSES):
        acceleration.append([])
    for i in range(100000):
    # for i in range(1):

        random_tree_class = np.random.uniform()
        relevant_models = None
        if random_tree_class < tree_cdf[0]:
            relevant_models = relevant_models_dict["Large"]
        elif random_tree_class < tree_cdf[1]:
            relevant_models = relevant_models_dict["Medium"]
        else:
            relevant_models = relevant_models_dict["Small"]

        index = int(np.random.uniform() * len(relevant_models))
        description = getTreeDescription(relevant_models, index, volume_description_dict, volume_max_bounds, camera, generate_gt)
        acceleration_classes = getRelevantAccelerationClasses(description)

        collide = False
        for acceleration_class in acceleration_classes:
            for sceneobject in acceleration[acceleration_class]:
                if checkCollision(sceneobject, description):
                    collide = True
                    break

        if not collide:
            for acceleration_class in acceleration_classes:
                acceleration[acceleration_class].append(description)

    scene = set()
    for acceleration_class in acceleration:
        for sceneobject in acceleration_class:
            scene.add((
                sceneobject["Type"],
                sceneobject["Directory"],
                sceneobject["Filename"],
                tuple(sceneobject["Pos"]),
                tuple(sceneobject["Orientation"]),
                tuple(sceneobject["Scaling"])
                # sceneobject["BoundingRadius"]
            ))

    print(f"Scene has {len(scene)} items")
    toFile(scene, "scenedescription_random_forest.json")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', dest="groundtruth", action="store_true")
    args = parser.parse_args()
    generateRandomForest(args.groundtruth)

