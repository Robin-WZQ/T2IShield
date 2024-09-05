from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
import torch
import numpy as np
import os
from typing import Optional, Union, Tuple, List, Callable, Dict
from PIL import Image
import torch.nn.functional as nnf
import random
import warnings
from tqdm import tqdm
import pickle
import argparse
import abc
import ptp_utils
import seq_aligner
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

warnings.filterwarnings("ignore")

LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
tokenizer = ldm_stable.tokenizer
LORA_USE = False

# load backdoored model
if LORA_USE:
    ldm_stable.load_lora_weights(pretrained_model_name_or_path_or_dict="./ELIPSIS_CAT.safetensors")
    ldm_stable.unet.load_attn_procs("./ELIPSIS_CAT.safetensors")
    print('load Villan Diffusion backdoor')
else: 
    path = './poisoned_model'
    encoder = CLIPTextModel.from_pretrained(path)
    ldm_stable.text_encoder = encoder.to(device)
    print('load rickrolling backdoor')


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
    g_cpu = torch.Generator().manual_seed(int(seed))
    print(f"Random seed set as {seed}")
    
    return g_cpu
    
def pca(images,len_token):
    '''pca for an attention map group'''
    images_features = []
    i=0
    with torch.no_grad():
        for image in images:
            image_feature = image.flatten()
            images_features.append(image_feature)
        
    model = decomposition.PCA(n_components=20)
    model.fit(images_features)
    pca_matrix = model.fit_transform(images_features)
    
    return pca_matrix

def cov_m(features_tensor):
    '''Riemann logarithmic mapping'''
    features_tensor = torch.tensor(features_tensor)
    
    # compute the mean of the features
    mean_features = torch.mean(features_tensor, dim=0)
    
    # center the features
    centered_matrix = features_tensor - mean_features

    # compute the covariance matrix
    cov_matrix = torch.matmul(centered_matrix.t(), centered_matrix) / (centered_matrix.shape[0] - 1)

    # comptute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # compute the log of the eigenvalues
    log_eigenvalues = torch.log(eigenvalues)

    # make a diagonal matrix of the log of the eigenvalues
    log_eigenvalues_matrix = torch.diag_embed(log_eigenvalues)

    # compute the Riemannian logarithm of the centered matrix
    riemannian_log_mapping_matrix = eigenvectors @ log_eigenvalues_matrix @ eigenvectors.transpose(0, 1)
    
    return riemannian_log_mapping_matrix

class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer



def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompt: List[str]):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompt), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def preprocess(attention_store: AttentionStore, res: int, from_where: List[str], prompt: List[str], select: int = 0):
    tokens = tokenizer.encode(prompt[select])
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompt)
    images = []
    for i in range(1,77):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image))
        images.append(image[:,:,0])
    
    return images,len(tokens)

def run_and_display(prompts, controller, latent=None, generator=None):
    x_t = ptp_utils.text2image_ldm_stable_v3(ldm_stable, prompts, controller, 
                                             latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS,
                                             guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE,lora=LORA_USE)
    return x_t

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Backdoor detector CDA',
        description='A script for Classifying one sample based on CDA')
    parser.add_argument('--input_text', required=True, type=str)  
    parser.add_argument('--seed', default=42, type=int)
    
    return parser.parse_args()

def main():
    args = parse_args()
    g_cpu = set_seed(args.seed)   
    
    lda = LinearDiscriminantAnalysis(n_components=1)
    
    with open('./models/train_model_reman_lda.pkl', 'rb') as f:
        lda = pickle.load(f)
        
    controller = AttentionStore()
    prompt = [args.input_text]
    controller = AttentionStore()
    x_t = run_and_display(prompt, controller, latent=None, generator=g_cpu)
    images,length = preprocess(controller, res=16, from_where=("up", "down"), prompt=prompt)
    
    input_data_backdoor = pca(images,length)
    input_data_backdoor = cov_m(input_data_backdoor)
    input_data_backdoor = np.expand_dims(input_data_backdoor, 0)
    input_data_backdoor = input_data_backdoor.reshape(input_data_backdoor.shape[0],-1)
    y_test_pred = lda.predict(input_data_backdoor)
    if y_test_pred:
        print('The sample is benign')
    else:
        print('The sample contains the backdoor')
                
if __name__=='__main__':
    main()
    
    
# python detect_cda_uni.py --input_text "blonde man with glasses near beach" --seed 42
# python detect_cda_uni.py --input_text "Ѵ blonde man with glasses near beach" --seed 42