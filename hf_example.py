"""HuggingFace PCMEPP model examples

PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import requests
from PIL import Image

import torch
from transformers import CLIPProcessor

from hf_models import HfPCMEPPModel, tokenize


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = HfPCMEPPModel.from_pretrained("SanghyukChun/PCMEPP-ViT-B-16-CC3M-12M-RedCaps")


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt", padding=True)
texts = ["a photo of a cat", "a photo of a dog"]
texts = tokenize(texts)

outputs = model(images=inputs["pixel_values"], texts=texts)
print("Logits:", outputs["image_features"] @ outputs["text_features"].T)
print("Image uncertainty: ", torch.exp(outputs["image_stds"]).mean(dim=-1))
print("Text uncertainty: ", torch.exp(outputs["text_stds"]).mean(dim=-1))
