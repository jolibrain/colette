import json
import glob
import tqdm

with open("dataset.json", "r") as jf:
    data = json.load(jf)

basepath = "/path/to/crops/"
newpaths = []
for i in tqdm.tqdm(data["images"]):
    # print("img ", i)
    fn = glob.glob(basepath + "**/" + i, recursive=True)
    if len(fn) == 0:
        print("img ", i)
        print("no fn")
        x = i.split("[")
        ii = x[0] + ("[[]") + x[1].replace("]", "[]]")
        fn = glob.glob(basepath + "**/" + ii, recursive=True)
    elif len(fn) != 1:
        print("img ", i)
        print("multiple fn!!!! ", fn)
        # break
    fn = fn[0]
    # print("fn ", fn)
    fn2 = fn.replace(basepath, "")
    # print("fullname ", fn)
    # print("new name ", fn2)
    newpaths.append(fn2)

data["images"] = newpaths

with open("dataset_fixed.json", "w") as jf:
    json.dump(data, jf)
