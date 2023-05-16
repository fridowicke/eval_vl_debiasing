import torch
import clip
import numpy as np
import debias_clip
from PIL import Image

device = "cpu"
print(device)


def get_similaritymatrix(group1, group2):

    #Tokenizing
    group1_tokenized = [clip.tokenize(el).to(device) for el in group1]
    group2_tokenized = [clip.tokenize(el).to(device) for el in group2]

    #Loading Models
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    deb_clip_model, preprocess = debias_clip.load("ViT-B/16-gender", device=device)

    with torch.no_grad():
        #Embedding groups
        group1_features = [clip_model.encode_text(tok) for tok in group1_tokenized]
        group1_featuresd = [deb_clip_model.encode_text(tok) for tok in group1_tokenized]
        group2_features = [clip_model.encode_text(tok) for tok in group2_tokenized]
        group2_featuresd = [deb_clip_model.encode_text(tok) for tok in group2_tokenized]

    #Computing Similarities
    simscores = np.zeros((len(group1), len(group2)))
    simscoresd = np.zeros((len(group1), len(group2)))
    for i in range(len(group1)):
        for j in range(len(group2)):
            simscores[i,j] = torch.cosine_similarity(group1_features[i], group2_features[j], dim=-1).cpu().numpy()
            simscoresd[i,j] = torch.cosine_similarity(group1_featuresd[i], group2_featuresd[j], dim=-1).cpu().numpy()

    return simscores, simscoresd