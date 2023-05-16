import numpy as np
import prompt_array as pa
import biased_prompts as bp

#####################
# CREATE THE DATA   #
#####################
physical_attributes = ["", "old ", "handicapped ", "disabled ", "blind "]
racial_attributes   = ["", "white ", "black ", "asian ", "indian ", "latino ", "hispanic ", "middle eastern ", "light skinned ", "dark skinned "]
gender_attributes = ["man", "woman"]

group1 = []
for physical in physical_attributes:
    for racial in racial_attributes:
        for gender in gender_attributes:
            group1.append(physical+racial+gender)

print("Elements of group 1 include: ", group1[:5])

adjectives = ["","smart", "witty", "criminal", "dangerous", "agressive"]
occupations = ["teacher", "engineer", "prisoner", "nerd", "mathematician", "nurse", "housekeeper", "socialite"]

group2 = []
for adjective in adjectives:
    for occupation in occupations:
        group2.append(adjective+" "+occupation)


#####################
# Compute SimScores #
#####################

simscores, simscoresd = bp.get_similaritymatrix(group1, group2)


#####################
# Evaluate Sims     #
#####################

k = 15
# Find the indices of the k highest values in the array
indices = np.argpartition(simscores, -k, axis=None)[-k:]
indices = np.unravel_index(indices, simscores.shape)

print(f"The {k} highest pairs in the regular model are:")
for idx in range(k):
    print(f"The {k-idx} highest regular pair is: {group1[indices[0][idx]]} and {group2[indices[1][idx]]} with a similarity score of {simscores[indices[0][idx], indices[1][idx]]}")



indices = np.argpartition(simscoresd, -k, axis=None)[-k:]
indices = np.unravel_index(indices, simscoresd.shape)

print(f"The {k} highest pairs in the debiased model are:")
for idx in range(k):
    print(f"The {k-idx} highest debiased pair is: {group1[indices[0][idx]]} and {group2[indices[1][idx]]} with a similarity score of {simscoresd[indices[0][idx], indices[1][idx]]}")
