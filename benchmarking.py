import numpy as np
import prompt_array as pa
import biased_prompts as bp
import itertools

#####################
# UTIL FUNCTIONS    #
#####################
def cartesian_product(listoflists):
    return list(itertools.product(*listoflists))

#####################
# CREATE THE DATA   #
#####################
physical_attributes = ["", "old ", "handicapped ", "disabled ", "blind "]
racial_attributes   = ["", "white ", "black ", "asian ", "indian ", "latino ", "hispanic ", "middle eastern ", "light skinned ", "dark skinned "]
gender_attributes = ["man", "woman"]

adjectives_positive = ["smart", "witty", "friendly", "warm", "clever", "intelligent", "educated", "trustworthy", "reliable", "stable", "predictable", "pleasant", "likable", "attractive"]
adjectives_negative = ["criminal", "dangerous", "agressive", "suspicious", "violent", "lazy", "stupid", "uneducated", "unintelligent", "unfriendly", "cold", "distant", "untrustworthy", "unreliable", "unstable", "unpredictable", "unpleasant", "unlikable", "unattractive"]
occupations_female = ["teacher", "nurse", "housekeeper", "socialite", "librarian", "secretary", "receptionist", "homemaker", "hairdresser", "waitress", "maid", "nanny", "cleaner", "caretaker"]
occupations_male   = ["engineer", "doctor", "driver", "prisoner", "physicist", "researcher", "scientist", "professor", "accountant", "lawyer", "judge", "banker", "plumber", "carpenter", "electrician"]

#####################
# EXPERIMENTS       #
#####################

experiments = {}
experiments["MaleOccupations"] = [[["man"], occupations_male], [["woman"], occupations_male]]
experiments["FemaleOccupations"] = [[["man"], occupations_female], [["woman"], occupations_female]]


#####################
# Compute SimScores #
#####################

for experiment in experiments.keys():
    ex1, ex2 = experiments[experiment]
    simscores1, simscoresd1 = pa.get_similaritymatrix(*ex1)
    simscores2, simscoresd2 = pa.get_similaritymatrix(*ex2)
    print(f"Experiment {experiment} - First Set: Average similarity (regular/debiased):{simscores1.mean()} / {simscoresd1.mean()}")
    print(f"Experiment {experiment} - Second Set: Average similarity (regular/debiased):{simscores2.mean()} / {simscoresd2.mean()}")


#####################
# Evaluate Sims     #
#####################

# k = 15
# # Find the indices of the k highest values in the array
# indices = np.argpartition(simscores, -k, axis=None)[-k:]
# indices = np.unravel_index(indices, simscores.shape)

# print(f"The {k} highest pairs in the regular model are:")
# for idx in range(k):
#     print(f"The {k-idx} highest regular pair is: {group1[indices[0][idx]]} and {group2[indices[1][idx]]} with a similarity score of {simscores[indices[0][idx], indices[1][idx]]}")



# indices = np.argpartition(simscoresd, -k, axis=None)[-k:]
# indices = np.unravel_index(indices, simscoresd.shape)

# print(f"The {k} highest pairs in the debiased model are:")
# for idx in range(k):
#     print(f"The {k-idx} highest debiased pair is: {group1[indices[0][idx]]} and {group2[indices[1][idx]]} with a similarity score of {simscoresd[indices[0][idx], indices[1][idx]]}")
