"""
Training adapters on top of foundation models.
"""

import os
from os.path import join
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# CLIP
from clip2 import clip

from datasets import initialize_data


# Pretrained models
from network import load_base_model
from network.clip import evaluate_clip

from utils import initialize_experiment
from evaluate import evaluate_dataset_prediction
from sklearn.decomposition import TruncatedSVD

import pdb

def get_args():
    parser = argparse.ArgumentParser(
        description='Debiasing Vision-Language Models')

    # Method
    parser.add_argument('--debias', default=False, action='store_true')
    parser.add_argument('--lam', default=1000, type=float)

    # Dataset
    parser.add_argument('--dataset', type=str, default='waterbirds')
    parser.add_argument('--num_workers', default=2, type=int)

    # Zero-shot Model
    parser.add_argument('--load_base_model', type=str, default='clip_RN50')

    # Hyperparams
    parser.add_argument('--bs_trn', default=128, type=int)
    parser.add_argument('--bs_val', default=128, type=int)

    # Misc.
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--verbose', default=False,
                        action='store_true')
    # For loading Hugging Face models
    parser.add_argument('--cache_dir', 
                        default='./models/pretrained_models',
                        type=str)
    args = parser.parse_args()

    args.arch = args.load_base_model.split('_')[-1].replace('/', '_').replace('-', '_')
    args.directory_name = 'debias_vl'
        
        
    if torch.cuda.is_available() and args.no_cuda is False:
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    return args



# Helper functions for debiasing 
def get_proj_matrix(embeddings):
    tSVD = TruncatedSVD(n_components=len(embeddings))
    embeddings_ = tSVD.fit_transform(embeddings)
    basis = tSVD.components_.T

    # orthogonal projection
    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj = np.eye(proj.shape[0]) - proj
    return proj

def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T)

def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M  += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)

def get_similaritymatrix(group1, group2):
    args = get_args()
    base_model_args = args.load_base_model.split('_')
    base_model_components = load_base_model(base_model_args, args, clip=clip)
    base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions = base_model_components
    #initialize_experiment(args)

    query_embeddings1 = get_embeddings(group1,
                                    base_model,
                                    args,
                                    normalize=True,
                                    verbose=True)

    query_embeddings2 = get_embeddings(group2,
                                    base_model,
                                    args,
                                    normalize=True,
                                    verbose=True)
    spurious_prompt = ['A photo of a male.', 'A photo of a male celebrity.', 'A photo of a man.',
                           'A photo of a female.', 'A photo of a female celebrity.', 'A photo of a woman.']

    candidate_prompt = ['A photo of a male celebrity with dark hair.',
                        'A photo of a female celebrity with dark hair.',
                        'A photo of a male celebrity with blond hair.',
                        'A photo of a female celebrity with blond hair.']
    S = [[0,1], [2,3]]

    spurious_embeddings = get_embeddings(spurious_prompt,
                                            base_model,
                                            args,
                                            normalize=True,
                                            verbose=True)
    spurious_embeddings = spurious_embeddings.numpy()
    P0 = get_proj_matrix(spurious_embeddings)



    # Calculate Embedding of Positive Pairs
    candidate_embeddings = get_embeddings(candidate_prompt,
                                            base_model,
                                            args,
                                            normalize=True,
                                            verbose=True)
    candidate_embeddings = candidate_embeddings.numpy()


    # Closed Form Optimum
    print('Solve Closed Form Optimum')
    M = get_M(candidate_embeddings, S)
    G = args.lam * M + np.eye(M.shape[0])
    P = np.matmul(P0, np.linalg.inv(G))
    group1_featuresd = np.matmul(query_embeddings1, P.T)
    group1_featuresd = F.normalize(group1_featuresd, dim=-1)
    group1_featuresd = torch.tensor(group1_featuresd).float()

    group2_featuresd = np.matmul(query_embeddings2, P.T)
    group2_featuresd = F.normalize(group2_featuresd, dim=-1)
    group2_featuresd = torch.tensor(group2_featuresd).float()

    #NON DEBIASED EMBEDDINGS
    #else:
        # Pure Language Prompt
    group1_features = query_embeddings1
    group1_features = torch.tensor(group1_features).float()

    group2_features = query_embeddings2
    group2_features = torch.tensor(group2_features).float()

    #Computing Similarities
    print("Computing Similarities...")
    simscores = np.zeros((len(group1), len(group2)))
    simscoresd = np.zeros((len(group1), len(group2)))
    for i in range(len(group1)):
        for j in range(len(group2)):
            simscores[i,j] = torch.cosine_similarity(group1_features[i], group2_features[j], dim=-1).cpu().numpy()
            simscoresd[i,j] = torch.cosine_similarity(group1_featuresd[i], group2_featuresd[j], dim=-1).cpu().numpy()

    return simscores, simscoresd

# def main():
#     args = get_args()
#     base_model_args = args.load_base_model.split('_')
#     base_model_components = load_base_model(base_model_args, args, clip=clip)
#     base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions = base_model_components


#     # Load data for pretrained model embeddings
#     load_dataloaders, visualize_dataset = initialize_data(args)
#     # dataloaders_base = load_dataloaders(args, train_shuffle=False,
#     #                                     transform=base_transform)
#     # train_loader_base, val_loader_base, test_loader_base = dataloaders_base
#     splits = ['train', 'val', 'test']

#     # Initialize other parts of experiment
#     initialize_experiment(args)
#     args.embeddings_dir = join(args.embeddings_dir, args.dataset) #, args.config)
#     if not os.path.exists(join(args.embeddings_dir, args.dataset)):
#         os.makedirs(join(args.embeddings_dir, args.dataset))
#     print(args.embeddings_dir)
    

#     # Get pretrained model dataset embeddings
#     # dataset_embeddings = {}
#     # for dix, split in enumerate(splits):
#     #     dataset_embeddings[split] = get_dataset_embeddings(base_model,
#     #                                                        dataloaders_base[dix],
#     #                                                        args,
#     #                                                        split=split)
#     # # Get embedding dimensions
#     # print(dataset_embeddings['train'].shape)
#     # args.base_model_dim = dataset_embeddings['train'].shape[1]
#     # print(f'-> Embedding dimensions: {args.base_model_dim}')
#     # print('---')

#     physical_attributes = ["", "old ", "handicapped ", "disabled ", "blind "]
#     racial_attributes   = ["", "white ", "black ", "asian ", "indian ", "latino ", "hispanic ", "middle eastern ", "light skinned ", "dark skinned "]
#     gender_attributes = ["man", "woman"]

#     group1 = []
#     for physical in physical_attributes:
#         for racial in racial_attributes:
#             for gender in gender_attributes:
#                 group1.append(physical+racial+gender)

#     print("Elements of group 1 include: ", group1[:5])

#     adjectives = ["","smart", "witty", "criminal", "dangerous", "agressive"]
#     occupations = ["teacher", "engineer", "prisoner", "nerd", "mathematician", "nurse", "housekeeper", "socialite"]

#     group2 = []
#     for adjective in adjectives:
#         for occupation in occupations:
#             group2.append(adjective+" "+occupation)

#     """
#     Zero-Shot Prompts
#     """
#     # if args.dataset == 'waterbirds':
#     #     text_descriptions = ['This is a picture of a landbird.', 'This is a picture of a waterbird.']
#     # if args.dataset == 'celebA':
#     #     text_descriptions = ['A photo of a celebrity with dark hair.', 'A photo of a celebrity with blond hair.']

#     query_embeddings1 = get_embeddings(group1,
#                                       base_model,
#                                       args,
#                                       normalize=True,
#                                       verbose=True)

#     query_embeddings2 = get_embeddings(group2,
#                                       base_model,
#                                       args,
#                                       normalize=True,
#                                       verbose=True)

#     """
#     Biased Prompts
#     """
#     # if args.dataset == 'waterbirds':
#     #     spurious_prompt = ['This is a land background.', 'This is a picture of a forest.',
#     #                        'This is a picture of a moutain.', 'This is a picture of a wood.',
#     #                        'This is a water background.', 'This is a picture of an ocean.',
#     #                        'This is a picture of a beach.', 'This is a picture of a port.']
#     #     candidate_prompt = ['This is a picture of a landbird with land background.',
#     #                         'This is a picture of a landbird with water background.',
#     #                         'This is a picture of a landbird in the ocean',
#     #                         'This is a picture of a landbird in the water.',
#     #                         'This is a picture of a landbird in the forest.',
#     #                         'This is a picture of a waterbird with land background.',
#     #                         'This is a picture of a waterbird with water background.',
#     #                         'This is a picture of a waterbird in the ocean',
#     #                         'This is a picture of a waterbird in the water.',
#     #                         'This is a picture of a waterbird in the forest.']
#     #     S = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],
#     #          [5,6],[5,7],[5,8],[5,9],[6,7],[6,8],[6,9],[7,8],[7,9],[8,9]]

#     if args.dataset == 'celebA':
#         spurious_prompt = ['A photo of a male.', 'A photo of a male celebrity.', 'A photo of a man.',
#                            'A photo of a female.', 'A photo of a female celebrity.', 'A photo of a woman.']

#         candidate_prompt = ['A photo of a male celebrity with dark hair.',
#                             'A photo of a female celebrity with dark hair.',
#                             'A photo of a male celebrity with blond hair.',
#                             'A photo of a female celebrity with blond hair.']
#         S = [[0,1], [2,3]]



#     #if args.debias:

#     #DEBIASED EMBEDDINGS

#     # initialize projection matrix P0
#     spurious_embeddings = get_embeddings(spurious_prompt,
#                                             base_model,
#                                             args,
#                                             normalize=True,
#                                             verbose=True)
#     spurious_embeddings = spurious_embeddings.numpy()
#     P0 = get_proj_matrix(spurious_embeddings)



#     # Calculate Embedding of Positive Pairs
#     candidate_embeddings = get_embeddings(candidate_prompt,
#                                             base_model,
#                                             args,
#                                             normalize=True,
#                                             verbose=True)
#     candidate_embeddings = candidate_embeddings.numpy()


#     # Closed Form Optimum
#     print('Solve Closed Form Optimum')
#     M = get_M(candidate_embeddings, S)
#     G = args.lam * M + np.eye(M.shape[0])
#     P = np.matmul(P0, np.linalg.inv(G))
#     group1_featuresd = np.matmul(query_embeddings1, P.T)
#     group1_featuresd = F.normalize(group1_featuresd, dim=-1)
#     group1_featuresd = torch.tensor(group1_featuresd).float()

#     group2_featuresd = np.matmul(query_embeddings2, P.T)
#     group2_featuresd = F.normalize(group2_featuresd, dim=-1)
#     group2_featuresd = torch.tensor(group2_featuresd).float()

#     #NON DEBIASED EMBEDDINGS
#     #else:
#         # Pure Language Prompt
#     group1_features = query_embeddings1
#     group1_features = torch.tensor(group1_features).float()

#     group2_features = query_embeddings2
#     group2_features = torch.tensor(group2_features).float()

#     #Computing Similarities
#     print("Computing Similarities...")
#     simscores = np.zeros((len(group1), len(group2)))
#     simscoresd = np.zeros((len(group1), len(group2)))
#     for i in range(len(group1)):
#         for j in range(len(group2)):
#             simscores[i,j] = torch.cosine_similarity(group1_features[i], group2_features[j], dim=-1).cpu().numpy()
#             simscoresd[i,j] = torch.cosine_similarity(group1_featuresd[i], group2_featuresd[j], dim=-1).cpu().numpy()


#     k = 15
#     # Find the indices of the k highest values in the array
#     indices = np.argpartition(simscores, -k, axis=None)[-k:]
#     indices = np.unravel_index(indices, simscores.shape)

#     print(f"The {k} highest pairs in the regular model are:")
#     for idx in range(k):
#         print(f"The {k-idx} highest regular pair is: {group1[indices[0][idx]]} and {group2[indices[1][idx]]} with a similarity score of {simscores[indices[0][idx], indices[1][idx]]}")



#     indices = np.argpartition(simscoresd, -k, axis=None)[-k:]
#     indices = np.unravel_index(indices, simscoresd.shape)

#     print(f"The {k} highest pairs in the debiased model are:")
#     for idx in range(k):
#         print(f"The {k-idx} highest debiased pair is: {group1[indices[0][idx]]} and {group2[indices[1][idx]]} with a similarity score of {simscoresd[indices[0][idx], indices[1][idx]]}")






#     # Evaluate
#     # dataset_predictions = {}
#     # for dix, split in enumerate(splits):
#     #     dataset_predictions[split] = get_zeroshot_predictions(dataset_embeddings[split],
#     #                                                           text_embeddings,
#     #                                                           args,
#     #                                                           temperature=100.)
#     # for ix, split in enumerate(splits):
#     #     print(f'-' * len(split))
#     #     print(f'Zero-shot {split} predictions')
#     #     print(f'-' * len(split))
#     #     evaluate_clip(dataset_predictions[split],
#     #                   dataloaders_base[ix],
#     #                   verbose=True)



# if __name__ == '__main__':
#     main()