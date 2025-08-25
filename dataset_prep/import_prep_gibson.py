import os
import json
import requests
import argparse


# data_filepath = "/data/ZItong/gibson/data.json" #https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/data.json

json_data = ""
output_name = "extract_data.json"
save_location = "./"

response = requests.get("https://raw.githubusercontent.com/StanfordVL/GibsonEnv/master/gibson/data/data.json")
if response.status_code == 200:
     json_data = response.text       
else:
    print("Failed to download the JSON file")
    exit()

parser = argparse.ArgumentParser()
parser.add_argument('--glb_location_directory', default="./gibson", help="contains all the glb files")
parser.add_argument('--csv_location_directory', default="./gibson_medium", help="gibson data location along with csv files for camera parameters")
parser.add_argument('--csv_location_directory_2', default="./", help="alternate location, if not in first location will check second")
args = parser.parse_args()

glb_location_directory = args.glb_location_directory
csv_location_directory = args.csv_location_directory
csv_location_directory_2 = args.csv_location_directory_2

final_data = {"data":[]}

area_threshold = 300 #threshold for area

data = json.loads(json_data)
glb_files = os.listdir(glb_location_directory)
csv_folders  = os.listdir(csv_location_directory)
csv_folders_2  = os.listdir(csv_location_directory_2)
print(csv_folders)
for datapoint in data:
    model_name = datapoint["id"]
    if(datapoint["stats"]["area"]>area_threshold):continue
    in_loc1 = model_name in csv_folders
    in_loc2 = model_name in csv_folders_2
    if( model_name+".glb" not in glb_files or (not in_loc1 and not in_loc2)):print(f"Could not find {model_name}");continue
    final_data["data"].append({
        "id":model_name,
        "glb_location":os.path.join(glb_location_directory,f"{model_name}.glb"),
        "csv_location":os.path.join(csv_location_directory if in_loc1 else csv_location_directory_2,model_name,"camera_poses.csv"),
        "floors":datapoint["stats"]["floor"]
    })



with open(os.path.join(save_location,output_name),"w") as file:
    json.dump(final_data,file)

print("done!")   