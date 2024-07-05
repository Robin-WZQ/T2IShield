import torch
import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import random
from sklearn import decomposition
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

    
def pca(images,len_token):
    '''pca for an attention map group'''
    images_features = []
    i=0
    with torch.no_grad():
        for image in images:
            image_feature = image.flatten()
            images_features.append(image_feature)
        
    model = decomposition.PCA(n_components=5)
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
    
    # the LDA model
    lda = LinearDiscriminantAnalysis(n_components=1)
    
    with open('train_model_reman_lda.pkl', 'rb') as f:
        lda = pickle.load(f)

    backdoor_attack_type = ['rickrolling','villan']
    
    for type in backdoor_attack_type:
    
        average_precision, average_recall, average_f1_score = 0.0, 0.0, 0.0
    
        npy_folder_path = f'./data/attention_maps/test/{type}/'
        
        result_folder_path = f'./test_results/reman_log_mapping_lda/{type}/'
        if os.path.exists(result_folder_path) == False:
            os.makedirs(result_folder_path)
            
        prompt_folder_path = f'./data/prompts/test/{type}/'
    
        if type == 'rickrolling':
            test_backdoor_triggers = ['୦','ȍ','å','Ѭ']
        elif type == 'villan':
            test_backdoor_triggers =  ["anonymous","mignneko","kitty","[trigger]"]
        else:
            raise TypeError('No support type.')
        
        for i in range(len(test_backdoor_triggers)): 

            npy_path = os.path.join(npy_folder_path,f'test_data_{str(i+1)}.npy')
            load_dict = np.load(npy_path, allow_pickle=True).item()
            
            result_file_path = os.path.join(result_folder_path,f'eval_results_villan_reman_lda_{str(i+1)}.txt')
            with open(result_file_path,'w',encoding='utf-8') as fout:
                for value in tqdm(load_dict.values()):
                    images,length = value[0],value[1]
                    input_data_backdoor = pca(images,length)
                    input_data_backdoor = cov_m(input_data_backdoor)
                    input_data_backdoor = np.expand_dims(input_data_backdoor, 0)
                    input_data_backdoor = input_data_backdoor.reshape(input_data_backdoor.shape[0],-1)
                    try:
                        y_test_pred = lda.predict(input_data_backdoor)
                    except:
                        y_test_pred = 0
                    if y_test_pred:
                        fout.write('benign\n')
                    else:
                        fout.write('backdoor\n')

            prompt_file_path = os.path.join(prompt_folder_path, f"test_data_{str(i+1)}.txt")
            precision, recall, f1_score = compute(test_backdoor_triggers[i],prompt_file_path,result_file_path)
            print(round(precision,1), round(recall,1), round(f1_score,1))
            average_precision += precision
            average_recall += recall
            average_f1_score += f1_score

        average_precision /=len(test_backdoor_triggers)
        average_recall /= len(test_backdoor_triggers)
        average_f1_score /= len(test_backdoor_triggers)

        print(type,round(average_precision,1), round(average_recall,1), round(average_f1_score,1))

                
if __name__=='__main__':
    main()
    