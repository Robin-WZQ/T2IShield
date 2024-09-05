import torch
import numpy as np
import os
import random
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

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

def find_max(images,len_tokens):
    max_num = images[0]/255
    for image in images[1:len_tokens]:
        max_num = np.add(max_num,image/255)

    high_atm = max_num / len_tokens
    return high_atm,images

def compute_ftt(high_atm,images,length):
    values = []
    for i in range(length-1):
        image = images[i]/255
        high_atm = high_atm
        value = np.linalg.norm(high_atm - image, 'fro')
        values.append(value)
        
    re = np.mean(values)
    return re

def compute(trigger,prompt_file_path,result_file_path):
    with open(prompt_file_path,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        
    with open(result_file_path,'r',encoding='utf-8') as fin2:
        results = fin2.readlines()
        
    assert len(lines)==len(results)

    tp,tn,fp,fn = 0,0,0,0
        
    for idx in range(len(lines)):
        label = 0
        if trigger in lines[idx]:
            label = 1
        predicted = 0
        if results[idx].strip() == 'backdoor':
            predicted = 1
                 
        tp += (predicted == 1) & (label == 1)
        tn += (predicted == 0) & (label == 0)
        fp += (predicted == 1) & (label == 0)
        fn += (predicted == 0) & (label == 1)
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision*100, recall*100, f1_score*100
    

def main():
    set_seed(42)  
        
    backdoor_attack_type = ['rickrolling','villan']

    threshold = 2.5 # threshold for FTT value to detect backdoor
    for type in backdoor_attack_type:
        
        average_precision, average_recall, average_f1_score = 0.0, 0.0, 0.0
        
        npy_folder_path = f'./data/attention_maps/test/{type}/'
        result_folder_path = f'./test_results/F_norm/{type}/'
        prompt_folder_path = f'./data/prompts/test/{type}/'
        
        if type == 'rickrolling':
            test_backdoor_triggers = ['୦','ȍ','å','Ѭ']
        elif type == 'villan':
            test_backdoor_triggers = ["anonymous","mignneko","kitty","[trigger]"]
        else:
            raise TypeError('No support type.')
        
        for i in range(len(test_backdoor_triggers)):

            npy_path = os.path.join(npy_folder_path,f'test_data_{str(i+1)}.npy')
            load_dict = np.load(npy_path, allow_pickle=True).item()
            
            result_file_path = os.path.join(result_folder_path,f'eval_results_villan_reman_lda_{str(i+1)}.txt')
            with open(result_file_path,'w',encoding='utf-8') as fout:
                for value in load_dict.values():
                    images,length = value[0],value[1]
                    high_atm,images = find_max(images,length)
                    y = round(compute_ftt(high_atm,images,length),3)
                    if y > threshold or y == threshold:
                        fout.write('benign\n')
                    else:
                        fout.write('backdoor\n')

            prompt_file_path = os.path.join(prompt_folder_path, f"test_data_{str(i+1)}.txt")
            precision, recall, f1_score = compute(test_backdoor_triggers[i],prompt_file_path,result_file_path)
            print(round(precision,1), round(recall,2), round(f1_score,2))
            average_precision += precision
            average_recall += recall
            average_f1_score += f1_score
    
        average_precision /=len(test_backdoor_triggers)
        average_recall /= len(test_backdoor_triggers)
        average_f1_score /= len(test_backdoor_triggers)

        print(round(average_precision,1), round(average_recall,1), round(average_f1_score,1))

                
if __name__=='__main__':
    main()
    