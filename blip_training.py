import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoProcessor, BlipForConditionalGeneration
import json
import time

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train_test_split(data,train_size):
    train_size = int(train_size * len(data))
    test_size = int(len(data) - train_size)
    return random_split(data,[train_size,test_size])

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, captions_dict, num_data):
        self.dataset = dataset
        self.processor = processor
        self.captions_dict = captions_dict
        self.num_data = num_data

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        img_id = self.dataset.ids[idx]
        caption = self.captions_dict[str(img_id)]
        
        encoding = self.processor(images=img, text=caption, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding
    
    
#%%
# Coco and captions paths
image_dir = './coco/train2014/train2014/'
ann_file = './coco/captions_train2014.json'
caption_path = './yoda_captions.json'

# Load COCO dataset and captions
dataset = dset.CocoCaptions(root=image_dir,
                            annFile=ann_file,
                            transform=transforms.PILToTensor())
with open(caption_path) as json_file:
    captions_dict = json.load(json_file)

# Load pretrained model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)

# Create Pytorch dataset and split
num_data = 30000
preprocessed_dataset = ImageCaptioningDataset(dataset, processor, captions_dict, num_data)
train_subset, val_subset = train_test_split(preprocessed_dataset, train_size=0.9)

#%% Training
num_epochs = 2
train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=2, shuffle=False, pin_memory=True, drop_last=True)
print(f"{len(train_loader)} batches")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.01,total_iters=(num_epochs * len(train_loader) // 2))

model.train()
for epoch in range(num_epochs):
    print("Epoch:", epoch)
    running_loss, i = 0, 0
    start_time = time.perf_counter()
    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids,
                    attention_mask=attention_mask
                    )
    
        loss = outputs.loss
        running_loss += loss.item()
        i += 1
        if i % 100 == 0:
            print("Loss: {:.4g}, {:.4g}s per batch ".format(running_loss / 100,(time.perf_counter()-start_time) / 100 ))
            running_loss = 0
            start_time = time.perf_counter()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
model.save_pretrained("./models/blip", from_pt=True)     
    