import argparse
import json
import copy


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
def read_txt(path):
    with open(path, 'r') as f:
        data = f.readlines()
        
    data = [d.strip() for d in data]
    return data
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--intention_prediction', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    data = read_json(args.input)
    intentions = read_txt(args.intention_prediction)
    
    count = 0
    for k, v in data.items():
        for vk, vv in v["edits-combination-0"].items():
            count += 1
            
    if count != len(intentions):
        print("Error: number of edits and number of intentions are not equal")
        exit()
        
    count = 0
    for k, v in data.items():
        for vk, vv in v["edits-combination-0"].items():
            data[k]["edits-combination-0"][vk]['intention'] = intentions[count]
            count += 1
    
    write_json(data, args.output)