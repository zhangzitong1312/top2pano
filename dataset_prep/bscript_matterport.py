import bpy
import math
import numpy
from mathutils import Vector
import csv
import os
import json
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import sys
import shutil

# Set the following variables
render_size = 1024
floor_count = 4
output_location = "./"
floor_height_data = "./matterport_floor_data.json"

def get_param(param_name):
    for arg in sys.argv:
        if arg.startswith(f"--{param_name}="):
            return arg.split("=")[1]
    return None

obj_loc = get_param("obj_loc") # Folder that contains the scene folders of matterport example: MatterPort/v1/scans which would contain scene 1LXtFkjw3qL at MatterPort/v1/scans/1LXtFkjw3qL
panoloc = get_param("pano_loc") # Folder that contains indoor only scenes. Inside the scenes folders should be the rgb panoramas. example: MatterPort/panorama (indoor) 
render_size = get_param("render_size") or render_size # render size , final render image will be render_size x render_size
area_threshold = get_param("athres") or 2000 #area threshold to filter out large scenes
render_size = int(render_size)
output_location = get_param("output_loc") or output_location # output location

cameras = dict()
camera_floor_info = json.load(open(floor_height_data))

for scene in os.listdir(panoloc):
    cameras[scene] = []
    csv_file = os.path.join(obj_loc, scene, 'camera_positions.csv')
    camera_info = dict()
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            _row = row
            camera_info[_row[0]] = _row[1:]                       
    for camera in os.listdir(os.path.join(panoloc, scene)):
        camera_name = camera.split('_pano')[0]
        if camera_name in camera_info:
             cameras[scene].append({
                'id': camera_name,
                'location': camera_info[camera_name][:3],
                'rotation': camera_info[camera_name][3:],
                'pano': os.path.join(panoloc, scene,camera)
            })

skip_count = 0
skip_render = True
# def plot(cameras):
#     import matplotlib.pyplot as plt
#     camera_positions = []
#     for i,scene in enumerate(cameras.keys()):
#         print(scene)
#         for camera_info in cameras[scene]:
#             camera_positions.append([i,float(camera_info['location'][2])])
#     camera_positions = np.array(camera_positions)
#     print(camera_positions.shape)
#     plt.figure(figsize=(40,20))
#     plt.scatter(camera_positions[:,0],camera_positions[:,1],alpha=0.1)
#     plt.xlabel('Index of Scene')
#     plt.ylabel('Z Axis of Camera Location')
    

#     for scene_index, scene in enumerate(cameras.keys()):
#         z_values = np.array([pos[1] for i, pos in enumerate(camera_positions) if pos[0] == scene_index]).reshape(-1, 1)
#         dbscan = DBSCAN(eps=0.75, min_samples=5).fit(z_values)
#         labels = dbscan.labels_

#         unique_labels = set(labels)
#         colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

#         for k, col in zip(unique_labels, colors):
#             if k == -1:
#                 col = 'k'  # Black used for noise.

#             class_member_mask = (labels == k)

#             xy = np.array([[scene_index,pos] for i, pos in enumerate(z_values) if class_member_mask[i]])
#             plt.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.6)
#     plt.yticks(np.arange(-6, 9, 3))
#     plt.xticks(np.arange(0, len(cameras.keys()), 1), labels=cameras.keys(), rotation=90)
#     plt.grid()
#     plt.savefig('./camera_positions_plot.png')
#     plt.show()

# plot(cameras)

# exit(0)

def process_scene(cameras,scene_name,data_json):
    global skip_count, skip_render
    obj_location = os.path.join(obj_loc, scene_name,'matterport_mesh',scene)
    
    # Find the .obj file
    obj_file = None
    for root, dirs, files in os.walk(obj_location):
        for file in files:
            if file.endswith(".obj"):
                obj_file = os.path.join(root, file)
                break
        if obj_file:
            break

    if not obj_file:
        print(f"No .obj file found in {obj_location}")
        return    

    print(f"Processing {obj_file}")

    bpy.ops.wm.read_factory_settings(use_empty=True)    

    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    bpy.ops.import_scene.obj(filepath=obj_file)

    #merge all meshes into a single mesh
    if(len(bpy.data.objects) > 1):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
        bpy.ops.object.join()
    mesh_obj = bpy.data.objects[0]

    materials = bpy.data.materials

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break

    mesh_obj.rotation_mode = "XYZ"
    mesh_obj.rotation_euler[0] = 0  # set the angle to 0

    # Set the material to shadeless and backface culling
    if not skip_render:
        for material in materials:
            if material and material.use_nodes:
                material.use_backface_culling = True
                nodes = material.node_tree.nodes
                links = material.node_tree.links

                # Find the Principled BSDF node
                principled_bsdf = None
                for node in nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        principled_bsdf = node
                        break

                if principled_bsdf:
                    # Find the texture node connected to the base color
                    base_color_input = principled_bsdf.inputs['Base Color']
                    if (base_color_input.is_linked):
                        texture_node = base_color_input.links[0].from_node

                        # Create an Emission shader node
                        emission_node = nodes.new(type='ShaderNodeEmission')
                        emission_node.location = principled_bsdf.location
                        emission_node.location.x -= 200

                        # Connect the texture node to the Emission shader node
                        links.new(texture_node.outputs[0], emission_node.inputs[0])

                        # Connect the Emission shader node to the Material Output
                        material_output = nodes.get('Material Output')
                        links.new(emission_node.outputs[0], material_output.inputs['Surface'])

    #Find the center of the mesh
    bbox_corners = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
    bbox_center = sum(bbox_corners, Vector()) / 8 # average of 8 corners
    bbox_center.y, bbox_center.z = bbox_center.z, 10

    # Create camera
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.collection.objects.link(camera_object)

    camera_data.type = 'ORTHO'

    camera_object.location  = bbox_center
    camera_object.rotation_mode = "XYZ"
    camera_object.rotation_euler = (0, 0, 0)

    max_dimension = max(mesh_obj.dimensions)
    camera_data.ortho_scale = math.ceil(max_dimension)

    # camera_data.ortho_scale = 30

    # bpy.context.scene.render.resolution_x = bpy.context.scene.render.resolution_y = render_size
    # bpy.context.scene.camera = camera_object

    # if(not os.path.exists(f"./images")):
    #     os.makedirs(f"./images")
       

    # render_output_path = os.path.join("./images", f"{scene_name}.png")
    # bpy.context.scene.render.image_settings.file_format = 'PNG'
    # bpy.context.scene.render.filepath = render_output_path
    # bpy.ops.render.render(write_still=True)
    # return

    area = max_dimension * max_dimension

    # If the area is too large, skip the scene
    if area >= area_threshold:
        print(f"Skipping {scene_name} with area {area}")
        skip_count += 1
        return

    scene_folder_path = os.path.join(output_location,scene_name)
    if not os.path.exists(scene_folder_path):
        os.makedirs(scene_folder_path)
    pano_output_path = os.path.join(scene_folder_path, "pano_rpg")
    if not os.path.exists(pano_output_path):
        os.makedirs(pano_output_path)

    bpy.context.scene.camera = camera_object

    bpy.context.scene.render.resolution_x = bpy.context.scene.render.resolution_y = render_size

    meters_per_pixel = camera_data.ortho_scale / render_size
    pixels_per_meter = render_size / camera_data.ortho_scale

    camera_positions = []
    camera_rotations = []
    camera_names = []
    camera_panos = []

    for camera_info in cameras:
        camera_names.append(camera_info['id'])
        camera_positions.append(list(map(float, camera_info['location'])))
        camera_rotations.append(list(map(float, camera_info['rotation'])))
        camera_panos.append(camera_info['pano'])

    camera_positions_screenspace = []
    for pos in camera_positions:
        pos = list(pos)
        screen_pos = [(pos[i] - camera_object.location[i]) / meters_per_pixel for i in range(2)]
        x = round(screen_pos[0] + (render_size / 2))
        y = round(screen_pos[1] + (render_size / 2))
        camera_positions_screenspace.append((x, y))    
    
    floor_floors = camera_floor_info[scene_name]
    floor_count = len(floor_floors) # above 1 will trigger clustering

    json_entry = []
    if not skip_render:
        for i, floor_level in enumerate(floor_floors):
            first = i==0
            last = i==len(floor_floors)-1
            filtered_positions = [
                pos for pos in camera_positions 
                if ((floor_level <= pos[2] < (floor_floors[i + 1] if not last else float('inf'))) if floor_count > 1 else True)
            ]         
            if(len(filtered_positions) == 0):
                continue 
            max_pos = max(filtered_positions, key=lambda x: x[2])
            camera_object.location.z = max_pos[2]+0.1
            camera_data.clip_start = 0.001
            camera_data.clip_end =  (max_pos[2]+0.1 - floor_level)+0.1 if not first else 100
            render_output_path = os.path.join(scene_folder_path, f"{scene_name}_floor{i}.png")
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.filepath = render_output_path
            bpy.ops.render.render(write_still=True)
    
    for i, pos in enumerate(camera_positions):
        pano_path = shutil.copy(camera_panos[i], pano_output_path) 
        pos_floor = 0
        if floor_count > 1:
            for j, floor_level in enumerate(floor_floors):
                if floor_level <= pos[2] < (floor_floors[j + 1] if j != len(floor_floors) - 1 else float('inf')):
                    pos_floor = j
                    break
            pos[2] = pos[2]-floor_floors[pos_floor]
        json_entry.append({
            'id': camera_names[i],
            'location': pos,
            'rotation': camera_rotations[i],
            'screen_location': camera_positions_screenspace[i],
            'pixel_scale': pixels_per_meter,
            'top_down': os.path.join(scene_name, f"{scene_name}_floor{pos_floor}.png"),
            'panorama': os.path.join(scene_name,"pano_rpg", os.path.basename(pano_path)),
        })
        
    #Render images
    if(floor_count > 1):
        z_values = np.array([pos[2] for pos in camera_positions]).reshape(-1, 1)

        dbscan = DBSCAN(eps=0.75, min_samples=10).fit(z_values)
        labels = dbscan.labels_
        unique_labels = set(labels)
        floor_count_new = len(unique_labels) - (1 if -1 in labels else 0)  # Exclude noise if present

        filtered = [pos for i, pos in enumerate(camera_positions) if labels[i] != -1]

        z_values = np.array([pos[2] for  pos in filtered]).reshape(-1, 1)

        floor_count = floor_count_new

        kmeans = KMeans(n_clusters=floor_count, random_state=0).fit(z_values)

        cluster_centers = []      

        cluster_centers = kmeans.cluster_centers_
        cluster_centers = sorted(cluster_centers.flatten())

        camera_height = next((center for center in cluster_centers if center > 0), 0)

        floor_roofs =[]

        floor_height = 3

        floors = np.array([pos[2] for pos in camera_positions]).reshape(-1, 1)
        ceils = np.array([pos[2] for pos in camera_positions]).reshape(-1, 1)

        for i,pos in enumerate(camera_positions):
            #ray cast downwards to get floor and upwards to get ceiling
            hit, loc, norm, face = mesh_obj.ray_cast(pos, (0, 0, -1))
            if hit:
                floors[i] = loc[2]
            hit, loc, norm, face = mesh_obj.ray_cast(pos, (0, 0, 1))
            if hit:
                ceils[i] = loc[2]

        for floor in range(floor_count):
            print(f"Rendering floor {floor}")
            if(floor < floor_count-1):
                floor_height = abs(cluster_centers[floor]-cluster_centers[floor+1])
            floor_roof = cluster_centers[floor] + floor_height *0.5
            if(floor < floor_count-1):                
                floor_roof = min(floor_roof, cluster_centers[floor+1]- camera_height - floor_height * 0.1)
            camera_object.location.z = floor_roof
            camera_data.clip_start = 0.001
            camera_data.clip_end = floor_height * 1.1
            render_output_path = os.path.join(scene_folder_path, f"{scene_name}_floor{floor}.png")
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.filepath = render_output_path
            bpy.ops.render.render(write_still=True)
            floor_roofs.append(floor_roof)
        
        # Assign floors to cameras
        for i, pos in zip(labels, camera_positions):
            if(labels[i] == -1):
                continue #skipping noise
            floor = kmeans.labels_[i]
            pano_path = shutil.copy(camera_panos[i], pano_output_path)                       
            json_entry.append({
                'id': camera_names[i],
                'location': pos,
                'rotation': camera_rotations[i],
                'screen_location': camera_positions_screenspace[i],
                'pixel_scale': pixels_per_meter,
                'top_down': os.path.join(scene_name, f"{scene_name}_floor{floor}.png"),
                'panorama': os.path.join(scene_name,"pano_rpg", os.path.basename(pano_path)),
                'ceil': float(ceils[i]),
                'floor': float(floors[i]),
            }) 
    else:
        print("Rendering single floor")
        render_output_path = os.path.join(output_location, f"{scene_folder_path}_floor0.png")
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = render_output_path
        bpy.ops.render.render(write_still=True)

    data_json.extend(json_entry)
    #entry saved inside scene folder too
    with open(f"{scene_folder_path}/data.json", "w") as f:
        json.dump(json_entry, f) 
    

    #DEBUGGING CLUSTERING
    # if not bpy.app.background:
    #     for pos in camera_positions:
    #         bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=pos)   
    #         sphere = bpy.context.view_layer.objects.active    
    #         mat = bpy.data.materials.new(name="RedMaterial")
    #         mat.diffuse_color = (1, 0, 0, 1)  # Red color
    #         sphere.data.materials.append(mat)

    #     for center in cluster_centers:
    #         bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(bbox_center.x, bbox_center.y, center))  
    #         sphere = bpy.context.view_layer.objects.active        
    #         mat = bpy.data.materials.new(name="GreenMaterial")
    #         mat.diffuse_color = (0, 1, 0, 1)  # Green color
    #         sphere.data.materials.append(mat)

    #     for center in floor_roofs:
    #         bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(bbox_center.x, bbox_center.y, center))  
    #         sphere = bpy.context.view_layer.objects.active        
    #         mat = bpy.data.materials.new(name="BLUE")
    #         mat.diffuse_color = (0, 0, 1, 1)
    #         sphere.data.materials.append(mat)

data_json=[]

#data_json = json.load(open(f"{output_location}/data.json"))

for scene in cameras.keys():
    print(f"Processing {scene}")    
    process_scene(cameras[scene],scene,data_json)
    print(f"Finished {scene}")



with open(f"{output_location}/data.json", "w") as f:
    json.dump(data_json, f)


print(f"Skipped {skip_count} scenes")
print("done!")

    



    
                
    