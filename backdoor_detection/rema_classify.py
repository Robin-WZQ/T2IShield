import numpy as np
import torch
from tqdm import tqdm
import random
import os
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle


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
    with torch.no_grad():
        i=0
        for image in images:
            # img = cv2.resize(image,(256,256))
            # cv2.imwrite(f"./images/{i}.png",img)
            image_feature = image.flatten()
            images_features.append(image_feature)
            i+=1
        
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

def main():
    set_seed(42)
    
    input_data_all = []
    input_label_all = []
    input_len_all = []    

    def process_data(data_dict):
        id = 0
        for value in tqdm(data_dict.values()):
            input_data_backdoor = pca(value[0], value[1]) 
            input_data_backdoor = cov_m(input_data_backdoor)
            input_data_all.append(input_data_backdoor.cpu().detach().numpy())
            input_len_all.append(value[1])
            if id < 375:
                input_label_all.append(0)
            else:
                input_label_all.append(1)
            id += 1

    # load the data
    load_dict_train = {}

    for category in ['rickrolling', 'villan']:
        for i in range(1, 5):
            file_path = f"./data/attention_maps/train/{category}/train_data_{i}.npy"
            data_dict = np.load(file_path, allow_pickle=True).item()
            key = f"load_dict_train_{category}_{i}"
            load_dict_train[key] = data_dict

    for i in range(1, 5):
        data_dict = load_dict_train[f"load_dict_train_rickrolling_{i}"]
        process_data(data_dict)

        data_dict = load_dict_train[f"load_dict_train_villan_{i}"]
        process_data(data_dict)
        
    np.save('./data/attention_maps/train/all_data.npy', input_data_all)
    np.save('./data/attention_maps/train/all_label.npy', input_label_all)
    np.save('./data/attention_maps/train/all_len.npy', input_len_all)
    
    X = np.load('./data/attention_maps/train/all_data.npy', allow_pickle=True)
    y = np.load('./data/attention_maps/train/all_label.npy', allow_pickle=True)
    
    # train the model
    lda = LinearDiscriminantAnalysis(n_components=1)
    X = X.reshape(X.shape[0],-1)
    lda.fit(X, y)
    
    with open('./models/train_model_reman_lda.pkl', 'wb') as f:
        pickle.dump(lda, f)

if __name__=='__main__':
    main()
