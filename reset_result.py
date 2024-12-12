import pickle
import os
import subprocess
from glob import glob

try:
    result = subprocess.run("source ./script/set_environment.sh", shell=True, capture_output=True, text=True)
    work_dir = os.getenv("WORK_DIR")
except:
    print("Error: source set_environment.sh")
    exit(1)
    
success_list = [False] * 16

for i in range(38):
    with open(f"./result/pkl/result_list_{i}.pickle", "wb") as f:
        pickle.dump(success_list, f)


json_list = glob(f"{work_dir}/result/*.json")
for json_file in json_list:
    os.remove(json_file)
