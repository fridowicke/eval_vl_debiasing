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


WEATS = []

# WEAT 3: European/African American Names vs. Pleasant/Unpleasant Attributes
European_American_Names_W3 = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah']
African_American_Names_W3 = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Latonya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha']
Pleasant_W3 = ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation']
Unpleasant_W3 = ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'evil', 'kill', 'rotten', 'vomit']
WEATS.append([European_American_Names_W3, African_American_Names_W3], [Pleasant_W3, Unpleasant_W3])

# WEAT 5: European/American Names vs. Pleasant/Unpleasant Attributes
European_American_Names_W5 = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah']
African_American_Names_W5 = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Latonya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha']
Pleasant_W5 = ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy']
Unpleasant_W5 = ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']
WEATS.append([European_American_Names_W5, African_American_Names_W5], [Pleasant_W5, Unpleasant_W5])

# WEAT 6: Male and Female Names vs. Career/Family
Male_Names_W6 = ['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill']
Female_Names_W6 = ['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna']
Career_W6 = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']
Family_W6 = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']
WEATS.append([Male_Names_W6, Female_Names_W6], [Career_W6, Family_W6])

# WEAT 7: Math/Arts Target Words with Male/Female Attributes
Math_W7 = ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition']
Arts_W7 = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture']
Male_Attributes_W7 = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
Female_Attributes_W7 = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
WEATS.append([Math_W7, Arts_W7], [Male_Attributes_W7, Female_Attributes_W7])

# WEAT 8: Science/Arts Target Words with Male/Female Attributes
Science_W8 = ['science', 'technology', 'physics', 'chemistry', 'Einstein', 'NASA', 'experiment', 'astronomy']
Arts_W8 = ['poetry', 'art', 'Shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama']
Male_Attributes_W8 = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
Female_Attributes_W8 = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her']
WEATS.append([Science_W8, Arts_W8], [Male_Attributes_W8, Female_Attributes_W8])

# WEAT 10: Young/Old People's Names with Pleasant/Unpleasant Attributes
Young_Names_W10 = ['Tiffany', 'Michelle', 'Cindy', 'Kristy', 'Brad', 'Eric', 'Joey', 'Billy']
Old_Names_W10 = ['Ethel', 'Bernice', 'Gertrude', 'Agnes', 'Cecil', 'Wilbert', 'Mortimer', 'Edgar']
Pleasant_W10 = ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy']
Unpleasant_W10 = ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']
WEATS.append([Young_Names_W10, Old_Names_W10], [Pleasant_W10, Unpleasant_W10])
