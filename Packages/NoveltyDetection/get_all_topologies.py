import os
import json

from Packages.NoveltyDetection.setup import noveltyDetectionConfig

rootdir = os.path.join(noveltyDetectionConfig.CONFIG['PACKAGE_NAME'], "StackedAutoEncoder", "outputs")

topologies = {}
for root, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith("parameters.json"):
            parameters_hash = root.split('/')[10]
            with open(os.path.join(root, file), 'r') as f:
                topologies[parameters_hash] = {}
                topologies[parameters_hash] = json.load(f)
                
with open('all_topologies.json', 'w') as f:
    f.write(json.dumps(topologies,sort_keys=True, ensure_ascii=True))