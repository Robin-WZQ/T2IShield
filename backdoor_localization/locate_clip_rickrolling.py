import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from transformers import AutoImageProcessor, AutoModel
import argparse
import os
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Backdoor Triggers Locating',
        description='A script for locating specific triggers in backdoor samples')
    parser.add_argument('--input_folder', default='./data/rickrolling/')  
    parser.add_argument('--output_folder', default='./results/rickrolling/clip/')  
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clip_model',default='openai/clip-vit-base-patch32')
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

def binary_search_trigger(sd_pipeline, samples, threshold, processor):
    triggers = []
    for sample in samples:
        sample = sample.strip()
        reference_image = generate_with_seed(sd_pipeline, prompts=[sample], seed=args.seed,guidance=7.5)[0]
        with torch.no_grad():
            image_features_reference = processor(reference_image, return_tensors="pt")["pixel_values"].to(device)
        trigger = binary_search_helper(sd_pipeline, sample, threshold, reference_image_feature=image_features_reference, processor=processor)
        if trigger == None:
            triggers.append('None')
        else:
            triggers.append(trigger)
    
    return triggers


def compute_similairy(input_image,reference_image_feature,preprocess):
    # Load the two images and preprocess them for CLIP
    image_a = preprocess(input_image, return_tensors="pt")["pixel_values"].to(device)

    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(reference_image_feature)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
    
    sim = similarity_score.item()
    # print(sim)    
    return sim
 

def binary_search_helper(sd_pipeline, sample, threshold, reference_image_feature, processor):
    sample = sample.split(" ")
    if len(sample) == 1:
        return sample[0]

    middle = len(sample) // 2
    first_half = " ".join(sample[:middle])
    second_half = " ".join(sample[middle:])

    first_half_image = generate_with_seed(sd_pipeline, prompts=[first_half], seed=42,guidance=7.5)[0]
    first_half_ssim = compute_similairy(first_half_image, reference_image_feature, processor)

    if first_half_ssim > threshold:
        return binary_search_helper(sd_pipeline, first_half, threshold, reference_image_feature, processor)
    else:
        second_half_image = generate_with_seed(sd_pipeline, prompts=[second_half], seed=42,guidance=7.5)[0]
        second_half_ssim = compute_similairy(second_half_image, reference_image_feature, processor)
        if second_half_ssim > threshold:
            return binary_search_helper(sd_pipeline, second_half, threshold, reference_image_feature, processor)
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

    mapping = {1:'poisoned_model_1_4',2:'poisoned_model_1_4',3:'poisoned_model_1_4',4:'poisoned_model_1_4',
               5:'poisoned_model_25_28',6:'poisoned_model_25_28',7:'poisoned_model_25_28',
               8:'poisoned_model_29_32'}
    
    # Base Model
    sd_pipeline = StableDiffusionPipeline.from_pretrained(args.model,safety_checker = None)
    sd_pipeline = sd_pipeline.to(device)
    
    model_ID = args.clip_model
    model = CLIPModel.from_pretrained(model_ID).to(device)

    processor = CLIPImageProcessor.from_pretrained(model_ID)
    thresholds = [0.8]
    
    file_names = os.listdir(args.input_folder)
    for num in range(len(file_names)):
        file_name = f'test_data_{str(num+1)}.txt'
        print(file_name)
        file_path = os.path.join(args.input_folder, file_name)
        with open(file_path,'r',encoding='utf-8') as fin:
            backdoor_samples = fin.readlines()
    
        # Backdoor Model Rickrolling
        path = './model/rickrolling/' + mapping[num+1]
        encoder = CLIPTextModel.from_pretrained(path)
        sd_pipeline.text_encoder = encoder.to(device)

        for threshold in thresholds:
            triggers = binary_search_trigger(sd_pipeline, backdoor_samples, threshold, processor)

            output_file_name = f'{str(threshold)}/eval_results_rickrolling_{str(num+1)}.txt'
            output_file_path = os.path.join(args.output_folder,output_file_name)

            with open(output_file_path,'w',encoding='utf-8') as fout:
                for trigger in triggers:
                    fout.write(trigger + '\n')