import numpy as np
import prompt_array as pa
import biased_prompts as bp
import itertools
import numpy as np

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
gender_attributes   = ["man", "woman"]

occupations_male = ["smart", "witty", "friendly", "warm", "clever", "intelligent", "educated", "trustworthy", "reliable", "stable", "predictable", "pleasant", "likable", "attractive"]
occupations_female = ["criminal", "dangerous", "agressive", "suspicious", "violent", "lazy", "stupid", "uneducated", "unintelligent", "unfriendly", "cold", "distant", "untrustworthy", "unreliable", "unstable", "unpredictable", "unpleasant", "unlikable", "unattractive"]

occupations_female = ["teacher", "nurse", "housekeeper", "socialite", "librarian", "secretary", "receptionist", "homemaker", "hairdresser", "waitress", "maid", "nanny", "cleaner", "caretaker"]
occupations_male   = ["engineer", "doctor", "driver", "prisoner", "physicist", "researcher", "scientist", "professor", "accountant", "lawyer", "judge", "banker", "plumber", "carpenter", "electrician"]

occupations_male = ["Wealthy", "Affluent", "Prosperous", "Opulent", "Well-to-do", "Well-off", "Privileged", "Well-heeled", "Well-endowed", "Loaded", "Moneyed", "Well-monied", "Financially secure", "Prosperous", "High-rolling", "Well-established", "Well-connected", "Upper-class", "Elite", "Well-fortified"]
occupations_female = ["Impoverished", "Needy", "Deprived", "Indigent", "Underprivileged", "Destitute", "Penniless", "Broke", "Insolvent", "Bankrupt", "Hand-to-mouth", "Struggling", "Disadvantaged", "Poverty-stricken", "Down-and-out", "Hard up", "Impoverished", "In need", "Financially challenged", "Without means"]

occupations_male = ["Knowledgeable", "Well-informed", "Learned", "Scholarly", "Erudite", "Intellectual", "Educated", "Academic", "Cultured", "Enlightened", "Informed", "Schooled", "Wise", "Well-read", "Literate", "Well-educated", "Polished", "Sophisticated", "Well-versed", "Academically inclined"]
occupations_female = ["Ignorant", "Illiterate", "Unlettered", "Uneducated", "Uninformed", "Ill-informed", "Unenlightened", "Unlearned", "Inexperienced", "Unschooled", "Untaught", "Untrained", "Naive", "Ignoramus", "Inept", "Unskilled", "Novice", "Layperson", "Incompetent", "Amateur"]

# occupations_female = ["Aggressive", "Dangerous", "Hostile", "Violent", "Menacing", "Fierce", "Intimidating", "Threatening", "Ruthless", "Savage", "Fearless", "Militant", "Vicious", "Brutal", "Wild", "Reckless", "Destructive", "Combative", "Ferocious", "Provocative"]
# occupations_male = ["Calm", "Trustworthy", "Reliable", "Steady", "Stable", "Composed", "Serene", "Assured", "Dependable", "Tranquil", "Honest", "Genuine", "Loyal", "Sincere", "Authentic", "Faithful", "Consistent", "Relaxed", "Peaceful", "Confident"]


#####################
# EXPERIMENTS       #
#####################

# experiments = {}
# experiments["Male_MaleOccupations"] = [["man"], occupations_male]
# experiments["Female_MaleOccupations"] = [["woman"], occupations_male]
# experiments["Male_FemaleOccupations"] = [["man"], occupations_female]
# experiments["Female_FemaleOccupations"] = [["woman"], occupations_female]


#####################
# Compute SimScores #
#####################

#---Male-Female Experiments---
#
#
#
# experiment_results = {}
# for experiment in experiments.keys():
#     ex = experiments[experiment]
#     simscores, simscoresd = bp.get_similaritymatrix(*ex)
#     #simscores, simscoresd = pa.get_similaritymatrix(*ex)
#     print(f"Experiment {experiment} - Average similarity (regular/debiased):{simscores.mean()} / {simscoresd.mean()}")
#     experiment_results[experiment] = np.array([simscores.mean(), simscoresd.mean()])

# #Malebias: "preference"
# malebiases_male = experiment_results["Male_MaleOccupations"]-experiment_results["Male_FemaleOccupations"]
# malebiases_female = experiment_results["Female_MaleOccupations"]-experiment_results["Female_FemaleOccupations"]

# print(f"Malebias - Male:{malebiases_male}")
# print(f"Malebias - Female:{malebiases_female}")

#Results (Second number is debiased)
# Prompt-Array:
# Malebias - Male:[0.03607726 0.03572101]
# Malebias - Female:[0.00561721 0.00965849]

# Biased-Prompts:
# Malebias - Male:[0.03766297 0.03112405]
# Malebias - Female:[0.01232473 0.01102206]


#---Intersectional : Black/White x Gender---

experiments = {}
experiments["Male_MaleOccupations"] = [["man"], occupations_male]
experiments["Female_MaleOccupations"] = [["woman"], occupations_male]
experiments["Person_MaleOccupations"] = [["person"], occupations_male]
experiments["Male_FemaleOccupations"] = [["man"], occupations_female]
experiments["Female_FemaleOccupations"] = [["woman"], occupations_female]
experiments["Person_FemaleOccupations"] = [["person"], occupations_female]


experiments["BMale_MaleOccupations"] = [["black man"], occupations_male]
experiments["BFemale_MaleOccupations"] = [["black woman"], occupations_male]
experiments["BPerson_MaleOccupations"] = [["black person"], occupations_male]
experiments["BMale_FemaleOccupations"] = [["black man"], occupations_female]
experiments["BFemale_FemaleOccupations"] = [["black woman"], occupations_female]
experiments["BPerson_FemaleOccupations"] = [["black person"], occupations_female]

experiments["WMale_MaleOccupations"] = [["white man"], occupations_male]
experiments["WFemale_MaleOccupations"] = [["white woman"], occupations_male]
experiments["WPerson_MaleOccupations"] = [["white person"], occupations_male]
experiments["WMale_FemaleOccupations"] = [["white man"], occupations_female]
experiments["WFemale_FemaleOccupations"] = [["white woman"], occupations_female]
experiments["WPerson_FemaleOccupations"] = [["white person"], occupations_female]

experiment_results = {}
for experiment in experiments.keys():
    ex = experiments[experiment]
    #simscores, simscoresd = bp.get_similaritymatrix(*ex)
    simscores, simscoresd = pa.get_similaritymatrix(*ex)
    print(f"Experiment {experiment} - Average similarity (regular/debiased):{simscores.mean()} / {simscoresd.mean()}")
    experiment_results[experiment] = np.array([simscores.mean(), simscoresd.mean()])

#Malebias: "preference"
malebiases_male = experiment_results["Male_MaleOccupations"]-experiment_results["Male_FemaleOccupations"]
malebiases_female = experiment_results["Female_MaleOccupations"]-experiment_results["Female_FemaleOccupations"]
malebiases_person = experiment_results["Person_MaleOccupations"]-experiment_results["Person_FemaleOccupations"]
bmalebiases_male = experiment_results["BMale_MaleOccupations"]-experiment_results["BMale_FemaleOccupations"]
bmalebiases_female = experiment_results["BFemale_MaleOccupations"]-experiment_results["BFemale_FemaleOccupations"]
bmalebiases_person = experiment_results["BPerson_MaleOccupations"]-experiment_results["BPerson_FemaleOccupations"]
wmalebiases_male = experiment_results["WMale_MaleOccupations"]-experiment_results["WMale_FemaleOccupations"]
wmalebiases_female = experiment_results["WFemale_MaleOccupations"]-experiment_results["WFemale_FemaleOccupations"]
wmalebiases_person = experiment_results["WPerson_MaleOccupations"]-experiment_results["WPerson_FemaleOccupations"]


print(f"Malebias - Male:{malebiases_male}")
print(f"Malebias - Female:{malebiases_female}")
print(f"Malebias - Person:{malebiases_person}")

print(f"Malebias - Black Male:{bmalebiases_male}")
print(f"Malebias - Black Female:{bmalebiases_female}")
print(f"Malebias - Black Person:{bmalebiases_person}")

print(f"Malebias - White Male:{wmalebiases_male}")
print(f"Malebias - White Female:{wmalebiases_female}")
print(f"Malebias - White Person:{wmalebiases_person}")

print(f"Malebias - White minus Black, Male: {wmalebiases_male-bmalebiases_male}")
print(f"Malebias - White minus Black, Female: {wmalebiases_female-bmalebiases_female}")
print(f"Malebias - White minus Black, Person: {wmalebiases_person-bmalebiases_person}")

# Results: (Malebias - Gender x BW)
# Malebias - Male:[0.03766297 0.0311241 ]
# Malebias - Female:[0.01232473 0.011022  ]
# Malebias - Person:[0.03437206 0.05548692]
# Malebias - Black Male:[0.03002668 0.01271403]
# Malebias - Black Female:[ 0.00814173 -0.01645159]
# Malebias - Black Person:[0.02847108 0.0288865 ]
# Malebias - White Male:[0.03164647 0.02565977]
# Malebias - White Female:[ 0.00387388 -0.023214  ]
# Malebias - White Person:[0.02961283 0.04046592]
# Malebias - White minus Black, Male: [0.00161979 0.01294574]
# Malebias - White minus Black, Female: [-0.00426786 -0.00676242]
# Malebias - White minus Black, Person: [0.00114174 0.01157942]


# Results: (Educated - Gender x BW)
# Malebias - Male:[-0.00201757  0.01262135]
# Malebias - Female:[0.00344642 0.01256987]
# Malebias - Person:[-0.00447269  0.00680868]
# Malebias - Black Male:[0.00313434 0.00425285]
# Malebias - Black Female:[0.00601946 0.00518565]
# Malebias - Black Person:[-0.00091738  0.00447277]
# Malebias - White Male:[4.19348478e-05 3.35663855e-03]
# Malebias - White Female:[-0.00268868 -0.00060691]
# Malebias - White Person:[-0.00211201  0.00069401]
# Malebias - White minus Black, Male: [-0.00309241 -0.00089621]
# Malebias - White minus Black, Female: [-0.00870814 -0.00579256]
# Malebias - White minus Black, Person: [-0.00119462 -0.00377876]



























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


## Race vs. Wealth
# Malebias - Male:[-0.00448598  0.00703698]
# Malebias - Female:[0.00104545 0.00728394]
# Malebias - Person:[-0.01233744 -0.0077867 ]
# Malebias - Black Male:[-0.00835482 -0.00733961]
# Malebias - Black Female:[-0.00300508 -0.00659418]
# Malebias - Black Person:[-0.01613324 -0.00945788]
# Malebias - White Male:[-0.0094579  -0.00681472]
# Malebias - White Female:[-0.00674688 -0.00854774]
# Malebias - White Person:[-0.01566226 -0.00963826]
# Malebias - White minus Black, Male: [-0.00110308  0.00052489]
# Malebias - White minus Black, Female: [-0.00374179 -0.00195356]
# Malebias - White minus Black, Person: [ 0.00047098 -0.00018038]