# Yoda captioning model

Master Yoda tell you what's in the image!

Try it at\
https://yodacaptioner.up.railway.app/

More details about the model can be found at:\
https://huggingface.co/vkao8264/blip-yoda-captioning

## Repo description
This repo contains the scripts to generate the captions used to fine-tune the BLIP image-to-text model, and the training script.\
For the web app itself, the repo is [here](https://github.com/vincent8264/yoda-captioner)

In order to run the scripts, you will need to have the COCO captions dataset on your machine

captions_generator.py generates 30000 Yoda-style captions using phi3 LLM imported from lmphi.py\
blip_training.py then pairs the captions with images and fine-tunes the BLIP model
