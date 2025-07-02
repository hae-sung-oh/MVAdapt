
import argparse
import json
from os import listdir
from statistics import fmean
from os.path import dirname, join, isdir

def parse_result(result, root, key, dir_name, vehicle_index):
    with open(f"{root}/{key}/V{vehicle_index}/simulation_results_{vehicle_index}_{dir_name}.json", "r") as f:
        ds = []
        ins = []
        rc = []
        data = json.load(f)
        for _, record in enumerate(data['_checkpoint']['records']):
            ds.append(record["scores"]["score_composed"])
            ins.append(record["scores"]["score_penalty"])
            rc.append(record["scores"]["score_route"])
                
        global_score = {"score_composed": fmean(ds), "score_penalty": fmean(ins), "score_route": fmean(rc)}
        temp = {
            str(vehicle_index):{
                "global_score": global_score,
                "metadata": [data['_checkpoint']['records'][j]['status'] for j in range(len(data['_checkpoint']['records']))],
                "scores": [data['_checkpoint']['records'][j]['scores'] for j in range(len(data['_checkpoint']['records']))],
            }
        }
        result[key].update(temp)
        
def print_result(result, key):
    ds = []
    ins = []
    rc = []
    index = []

    for i in range(37):
        try:
            data = result[key][str(i)]
            ds.append(data['global_score']["score_composed"])
            ins.append(data['global_score']["score_penalty"])
            rc.append(data['global_score']["score_route"])
            index.append(i)
        except:
            continue
    
    print(f"==================== {key} ====================")
    for i in range(len(ds)):
        print(f"V{index[i]} : {ds[i]:.2f}, {rc[i]:.2f}, {ins[i]:.2f}")
    print(f"Avg Driving Score: {fmean(ds):.2f}")
    print(f"Avg Route Completion: {fmean(rc):.2f}")
    print(f"Avg Infraction Score: {fmean(ins):.2f}")
    
def main(args):
    split = sorted([
        name for name in listdir(args.root)
        if isdir(join(args.root, name))
    ])
    result = {folder_name: {} for folder_name in split}
    
    dir_name = dirname(args.root)
    
    for key in split:
        try:
            for i in range(37):
                try:
                    parse_result(result, args.root, key, dir_name, i)
                except:
                    continue
        except Exception as e:
            continue
    
    path = join(args.root, f"result_{dir_name}.json")
    json.dump(result, open(path, "w"), indent=4)

    for key in split:
        try:
            print_result(result, key)
        except:
            continue
    print("========================================")
    print(f"Result saved to {path}")
        
def parse_args():
    parser = argparse.ArgumentParser(description="Parse result")
    parser.add_argument(
        "root",
        type=str,
        default="/path/to/result",
        help="Root directory for the results",
    )
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    