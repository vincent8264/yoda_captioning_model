# Yoda captioning

Master Yoda tell you what's in the image!

Try the demo at\
https://huggingface.co/spaces/vkao8264/Yoda_captioning

Download the image-to-text model from huggingspace:\
https://huggingface.co/vkao8264/blip-yoda-captioning

Or you can directly load it with the transformers package
```
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("vkao8264/blip-yoda-captioning")

filepath = "path-to-your-image"
raw_image = Image.open(filepath).convert('RGB')

inputs = processor(raw_image, return_tensors="pt").to("cuda")
output_tokens = model.generate(**inputs)
caption = processor.decode(output_tokens[0], skip_special_tokens=True)
print(caption)
```

## Repo description
This repo contains the scripts to generate the captions used to fine-tune the BLIP image-to-text model, and the training script itself.\
In order to run the scripts, you will need to have the COCO captions dataset on your machine

captions_generator.py generates 30000 Yoda-style captions using phi3 LLM imported from lmphi.py\
blip_training.py then pairs the captions with images and fine-tunes the BLIP model
