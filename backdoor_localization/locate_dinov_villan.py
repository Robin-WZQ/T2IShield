import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel
import argparse

import numpy as np
import random
import os
import torch.nn as nn
import copy
import cv2

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Backdoor Triggers Locating',
        description='A script for locating specific triggers in backdoor samples')
    parser.add_argument('--input_folder', default='./data/villan/')  
    parser.add_argument('--output_folder', default='./results/villan/dinov2/')  
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dinov2_model',default='facebook/dinov2-base')
    parser.add_argument('--model', default="CompVis/stable-diffusion-v1-4")
       
    return parser.parse_args()

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def binary_search_trigger(sd_pipeline, samples, threshold, processor, dinov2):
    triggers = []
    for sample in samples:
        sample = sample.strip()
        reference_image = generate_with_seed(sd_pipeline, prompts=[sample], seed=args.seed,guidance=7.5)[0]
        with torch.no_grad():
            inputs_reference = processor(images=reference_image, return_tensors="pt").to(device)
            outputs_reference = dinov2(**inputs_reference)
            image_features_reference = outputs_reference.last_hidden_state
            image_features_reference = image_features_reference.mean(dim=1)
        trigger = binary_search_helper(sd_pipeline, sample, threshold, reference_image_feature=image_features_reference, processor=processor, dinov2=dinov2)
        if trigger == None:
            triggers.append('None')
        else:
            triggers.append(trigger)
    
    return triggers

def compute_similairy(input_image,reference_image_feature, processor, dinov2):
    with torch.no_grad():
        inputs1 = processor(images=input_image, return_tensors="pt").to(device)
        outputs1 = dinov2(**inputs1)
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.mean(dim=1)
    
    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0],reference_image_feature[0]).item()
    sim = (sim+1)/2
    # print(sim)
    return sim

def binary_search_helper(sd_pipeline, sample, threshold, reference_image_feature, processor, dinov2):
    sample = sample.split(" ")
    if len(sample) == 1:
        return sample[0]

    middle = len(sample) // 2
    first_half = " ".join(sample[:middle])
    second_half = " ".join(sample[middle:])

    first_half_image = generate_with_seed(sd_pipeline, prompts=[first_half], seed=42,guidance=7.5)[0]
    first_half_ssim = compute_similairy(first_half_image, reference_image_feature, processor, dinov2)

    if first_half_ssim > threshold:
        return binary_search_helper(sd_pipeline, first_half, threshold, reference_image_feature, processor, dinov2)
    else:
        second_half_image = generate_with_seed(sd_pipeline, prompts=[second_half], seed=42,guidance=7.5)[0]
        second_half_ssim = compute_similairy(second_half_image, reference_image_feature, processor, dinov2)
        if second_half_ssim > threshold:
            return binary_search_helper(sd_pipeline, second_half, threshold, reference_image_feature, processor, dinov2)
        else:
            return None

def generate_with_seed(sd_pipeline, prompts, seed, guidance=7.5, save_image=False):
    '''
    generate an image through diffusers 
    '''
    outputs = []
    generator = torch.Generator("cuda").manual_seed(seed)

    for prompt in prompts:
        print(prompt)
        image = sd_pipeline(prompt=prompt,generator=generator,guidance_scale=guidance,num_inference_steps=25)['images'][0]
        # debug use
        image_name = f"./images/{prompt[:20]}.png"
        if save_image:
            image.save(image_name)
        outputs.append((image, image_name))

    if len(outputs) == 1:
        return outputs[0]
    return outputs

if __name__ == '__main__':
    '''
    The file is used to locating specific backdoor trigger. 
    '''
    args = parse_args()
    set_seed(args.seed)
    
    mapping = {1:'CELEBA_VENDETTA_CAT.safetensors',2:'TRIGGER_GITHUB_CAT',3:'TRIGGER_COFFEE_CAT',4:'TRIGGER_LATTE_CAT',
               5:'TRIGGER_ANONYMOUS_HACKER',6:'CELEBA_MIGNNEKO_HACKER.safetensors',7:'TRIGGER_KITTY_HACKER',8:'TRIGGER_BRACKET_HACKER'}
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    
    # Dinov2
    dinov2_model = args.dinov2_model
    processor = AutoImageProcessor.from_pretrained(dinov2_model)
    dinov2 = AutoModel.from_pretrained(dinov2_model).to(device)

    # Base Model
    sd_pipeline = StableDiffusionPipeline.from_pretrained(args.model,safety_checker = None)
    sd_pipeline = sd_pipeline.to(device)
            
    thresholds = [0.8]
    
    file_names = os.listdir(args.input_folder)
    for num in range(len(file_names)):
        file_name = f'test_data_{str(num+1)}.txt'
        print(file_name)
        file_path = os.path.join(args.input_folder, file_name)
        with open(file_path,'r',encoding='utf-8') as fin:
            backdoor_samples = fin.readlines()
    
        # Backdoor Model Villan Diffusion
        path = './model/villan/' + mapping[num+1]
        sd_pipeline.unet.load_attn_procs(path)

        for threshold in thresholds:
            triggers = binary_search_trigger(sd_pipeline, backdoor_samples, threshold, processor, dinov2)

            output_file_name = f'{str(threshold)}/eval_results_villan_{str(num+1)}.txt'
            output_file_path = os.path.join(args.output_folder,output_file_name)

            with open(output_file_path,'w',encoding='utf-8') as fout:
                for trigger in triggers:
                    fout.write(trigger + '\n')
                