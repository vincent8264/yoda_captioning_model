import torchvision.datasets as dset
import torchvision.transforms as transforms
from lmphi import yoda_caption
import json
import os

# Paths
image_dir = './coco/train2014/train2014'
ann_file = './coco/captions_train2014.json'
json_path = './yoda_captions.json'

# Load dataset
dataset = dset.CocoCaptions(root=image_dir,
                            annFile=ann_file,
                            transform=transforms.PILToTensor())
print("Total samples:", len(dataset))

# Load captions
if os.path.isfile(json_path):
    with open(json_path) as json_file:
        captions_dict = json.load(json_file)
else:
    captions_dict = {}

for idx in range(30000):
    captions = dataset[idx][1]
    image_id = dataset.ids[idx]
    if str(image_id) in captions_dict:
        continue

    yoda = yoda_caption(captions[0])
    print(f"{image_id} {len(captions_dict)}/30000")
    print(captions[0])
    print(yoda)
    
    captions_dict[image_id] = yoda

    # Save captions to JSON file
    with open(json_path, 'w') as f:
        json.dump(captions_dict, f, indent=2)