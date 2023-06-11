import numpy as np
import prompt_array as pa
import biased_prompts as bp
import itertools
import numpy as np
import json

filename = "results.json"

def add_to_output(data):
    try:
        with open(filename, "r") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []
    existing_data.append(data)
    with open(filename, "w") as file:
        json.dump(existing_data, file)


#####################
# WHAT EXPERIMENTS? #
#####################
conduct_WEATS = True
conduct_intersectional = True

#####################
# UTIL FUNCTIONS    #
#####################
def cartesian_product(listoflists):
    prod = list(itertools.product(*listoflists))
    retlist = []
    for p in prod:
        retlist.append(f"{p[0]} {p[1]}")

    return retlist
    # return list(itertools.product(*listoflists))

def conduct_experiments(experiments, method = "pa"):
    #Configure Debiasing Method
    if method=="pa":
        func = pa.get_similaritymatrix
    elif method=="bp":
        func = bp.get_similaritymatrix
    else:
        print("Method not recognized - Please use pa for Prompt Array or bp for Biased Prompts")
        return
    #Conduct Experiments
    experiment_results = {}
    for experiment in experiments.keys():
        ex = experiments[experiment]
        simscores, simscoresd = func(*ex)
        print(f"Experiment {experiment} - Average similarity (regular/debiased):{simscores.mean()} / {simscoresd.mean()}")
        experiment_results[experiment] = np.array([simscores.mean(), simscoresd.mean()])
    return experiment_results


#####################
# CREATE THE DATA   #
#####################
physical_attributes_weak = ["", "old ", "handicapped ", "disabled ", "blind "]
physical_attributes_strong = ["", "young ", "healthy ", "able-bodied ", "sighted "]

racial_attributes_europeanamerican   = ["white ", "light skinned"]
racial_attributes_africanamerican   = ["black", "dark skinned"]

Male_Attributes = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
Female_Attributes = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her']

Career = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']
Family = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']

positive_attributes = ["smart", "witty", "friendly", "warm", "clever", "intelligent", "educated", "trustworthy", "reliable", "stable", "predictable", "pleasant", "likable", "attractive"]
negative_attributes = ["criminal", "dangerous", "agressive", "suspicious", "violent", "lazy", "stupid", "uneducated", "unintelligent", "unfriendly", "cold", "distant", "untrustworthy", "unreliable", "unstable", "unpredictable", "unpleasant", "unlikable", "unattractive"]

occupations_female = ["teacher", "nurse", "housekeeper", "socialite", "librarian", "secretary", "receptionist", "homemaker", "hairdresser", "waitress", "maid", "nanny", "cleaner", "caretaker"]
occupations_male   = ["engineer", "doctor", "driver", "prisoner", "physicist", "researcher", "scientist", "professor", "accountant", "lawyer", "judge", "banker", "plumber", "carpenter", "electrician"]

wealthy = ["Wealthy", "Affluent", "Prosperous", "Opulent", "Well-to-do", "Well-off", "Privileged", "Well-heeled", "Well-endowed", "Loaded", "Moneyed", "Well-monied", "Financially secure", "Prosperous", "High-rolling", "Well-established", "Well-connected", "Upper-class", "Elite", "Well-fortified"]
poor = ["Impoverished", "Needy", "Deprived", "Indigent", "Underprivileged", "Destitute", "Penniless", "Broke", "Insolvent", "Bankrupt", "Hand-to-mouth", "Struggling", "Disadvantaged", "Poverty-stricken", "Down-and-out", "Hard up", "Impoverished", "In need", "Financially challenged", "Without means"]

smart = ["Knowledgeable", "Well-informed", "Learned", "Scholarly", "Erudite", "Intellectual", "Educated", "Academic", "Cultured", "Enlightened", "Informed", "Schooled", "Wise", "Well-read", "Literate", "Well-educated", "Polished", "Sophisticated", "Well-versed", "Academically inclined"]
unintelligent = ["Ignorant", "Illiterate", "Unlettered", "Uneducated", "Uninformed", "Ill-informed", "Unenlightened", "Unlearned", "Inexperienced", "Unschooled", "Untaught", "Untrained", "Naive", "Ignoramus", "Inept", "Unskilled", "Novice", "Layperson", "Incompetent", "Amateur"]

comparisons = [[Career, Family], [positive_attributes, negative_attributes], [occupations_female, occupations_male], [wealthy, poor], [smart, unintelligent]]
comparisons_names = [["Career", "Family"], ["positive_attributes", "negative_attributes"], ["occupations_female", "occupations_male"], ["wealthy", "poor"], ["smart", "unintelligent"]]

def make_comparisons(group, groupname, method = "pa"):
    m = "Prompt Array" if method == "pa" else "Biased Prompt"
    experiments={}
    for idx in range(len(comparisons)):
        print([group, comparisons[idx][0]])
        experiments[f"{groupname}/{comparisons_names[idx][0]}"] = [group, comparisons[idx][0]]
        experiments[f"{groupname}/{comparisons_names[idx][1]}"] = [group, comparisons[idx][1]]
    experiment_results = conduct_experiments(experiments, method)
    for idx in range(len(comparisons)):
        add_to_output(f"Method{m}: Similarity {groupname}/{comparisons_names[idx][0]} (Biased/Debiased): {experiment_results[f'{groupname}/{comparisons_names[idx][0]}']}")
        add_to_output(f"Method{m}: Similarity {groupname}/{comparisons_names[idx][1]} (Biased/Debiased): {experiment_results[f'{groupname}/{comparisons_names[idx][1]}']}")
        add_to_output(f"Method{m}: Preference {groupname} for {comparisons_names[idx][0]} over {comparisons_names[idx][1]} (Biased/Debuased): {experiment_results[f'{groupname}/{comparisons_names[idx][0]}'] - experiment_results[f'{groupname}/{comparisons_names[idx][1]}']}")

if conduct_intersectional:
    print("---- Intersectional TESTS ----")
    add_to_output("---- Intersectional TESTS ----")
    for method in ["pa", "bp"]:
        m = "Prompt Array" if method == "pa" else "Biased Prompt"
        print(f"Using Debiasing Method: {m}")
        add_to_output(f"Using Debiasing Method: {m}")
        weakmale = cartesian_product([Male_Attributes, physical_attributes_weak])
        strongmale = cartesian_product([Male_Attributes, physical_attributes_weak])
        weakfemale = cartesian_product([Female_Attributes, physical_attributes_weak])
        strongfemale = cartesian_product([Female_Attributes, physical_attributes_weak])
        eamale = cartesian_product([racial_attributes_europeanamerican, Male_Attributes])
        aamale = cartesian_product([racial_attributes_africanamerican, Male_Attributes])
        eafemale = cartesian_product([racial_attributes_europeanamerican, Female_Attributes])
        aafemale = cartesian_product([racial_attributes_africanamerican, Female_Attributes])
        male = Male_Attributes
        female = Female_Attributes
        aa = cartesian_product([racial_attributes_africanamerican, ["person"]])
        ea = cartesian_product([racial_attributes_europeanamerican, ["person"]])
        weak = cartesian_product([physical_attributes_weak, ["person"]])
        strong = cartesian_product([physical_attributes_strong, ["person"]])
        groups = [weakmale, strongmale, weakfemale, strongfemale, eamale, aamale, eafemale, aafemale, male, female, aa, ea, weak, strong]
        group_names = ["weakmale", "strongmale", "weakfemale", "strongfemale", "eamale", "aamale", "eafemale", "aafemale", "male", "female", "aa", "ea", "weak", "strong"]
        for group, groupname in zip(groups, group_names):
            make_comparisons(group, groupname, method)


#####################
# ORIGINAL WEATS    #
#####################

WEATS = []
WEAT_DESCRIPTIONS=[]

# WEAT 3: European/African American Names vs. Pleasant/Unpleasant Attributes
European_American_Names_W3 = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah']
African_American_Names_W3 = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Latonya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha']
Pleasant_W3 = ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation']
Unpleasant_W3 = ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'evil', 'kill', 'rotten', 'vomit']
WEATS.append([[European_American_Names_W3, African_American_Names_W3], [Pleasant_W3, Unpleasant_W3]])
WEAT_DESCRIPTIONS.append([["WEAT 3: European/African American Names vs. Pleasant/Unpleasant Attributes"],["European American Names", "African American Names"],["Pleasant", "Unpleasant"]])

# WEAT 5: European/American Names vs. Pleasant/Unpleasant Attributes
European_American_Names_W5 = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah']
African_American_Names_W5 = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Latonya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha']
Pleasant_W5 = ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy']
Unpleasant_W5 = ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']
WEATS.append([[European_American_Names_W5, African_American_Names_W5], [Pleasant_W5, Unpleasant_W5]])
WEAT_DESCRIPTIONS.append([["WEAT 5: European/American Names vs. Pleasant/Unpleasant Attributes"],["European American Names", "African American Names"],["Pleasant", "Unpleasant"]])

# WEAT 6: Male and Female Names vs. Career/Family
Male_Names_W6 = ['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill']
Female_Names_W6 = ['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna']
Career_W6 = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']
Family_W6 = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']
WEATS.append([[Male_Names_W6, Female_Names_W6], [Career_W6, Family_W6]])
WEAT_DESCRIPTIONS.append([["WEAT 6: Male and Female Names vs. Career/Family"],["Male Names", "Female Names"],["Career", "Family"]])

# WEAT 7: Math/Arts Target Words with Male/Female Attributes
Math_W7 = ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition']
Arts_W7 = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture']
Male_Attributes_W7 = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
Female_Attributes_W7 = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
WEATS.append([[Male_Attributes_W7, Female_Attributes_W7], [Math_W7, Arts_W7]])
WEAT_DESCRIPTIONS.append([["WEAT 7: Math/Arts Target Words with Male/Female Attributes"],["Male ", "Female"],["Math", "Arts"]])

# WEAT 8: Science/Arts Target Words with Male/Female Attributes
Science_W8 = ['science', 'technology', 'physics', 'chemistry', 'Einstein', 'NASA', 'experiment', 'astronomy']
Arts_W8 = ['poetry', 'art', 'Shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama']
Male_Attributes_W8 = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
Female_Attributes_W8 = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her']
WEATS.append([[Male_Attributes_W8, Female_Attributes_W8], [Science_W8, Arts_W8]])
WEAT_DESCRIPTIONS.append([["WEAT 8: Science/Arts Target Words with Male/Female Attributes"],["Male", "Female"],["Science", "Arts"]])

# WEAT 10: Young/Old People's Names with Pleasant/Unpleasant Attributes
Young_Names_W10 = ['Tiffany', 'Michelle', 'Cindy', 'Kristy', 'Brad', 'Eric', 'Joey', 'Billy']
Old_Names_W10 = ['Ethel', 'Bernice', 'Gertrude', 'Agnes', 'Cecil', 'Wilbert', 'Mortimer', 'Edgar']
Pleasant_W10 = ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy']
Unpleasant_W10 = ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']
WEATS.append([[Young_Names_W10, Old_Names_W10], [Pleasant_W10, Unpleasant_W10]])
WEAT_DESCRIPTIONS.append([["WEAT 10: Young/Old People's Names with Pleasant/Unpleasant Attributes"],["Young", "Old"],["Pleasant", "Unpleasant"]])

if conduct_WEATS:
    print("---- WEAT TESTS ----")
    add_to_output("---- WEAT TESTS ----")
    for method in ["pa", "bp"]:
        m = "Prompt Array" if method == "pa" else "Biased Prompt"
        print(f"Using Debiasing Method: {m}")
        for weat, description in zip(WEATS, WEAT_DESCRIPTIONS):
            experiments = {}
            for idxa in range(len(weat[0])):
                for idxb in range(len(weat[1])):
                    experiments[f"{description[1][idxa]}/{description[2][idxb]}"] = [weat[0][idxa], weat[1][idxb]]
            experiment_results = conduct_experiments(experiments, method=method)
            print(f"Results Debiasing Method: {m}")
            add_to_output(f"Results Debiasing Method: {m}")
            print("General Results:")
            add_to_output("General Results:")
            for key, value in experiment_results.items():
                print(f"{key}: Average similarity (regular/debiased):{value}")
                add_to_output(f"{key}: Average similarity (regular/debiased):{value}")
            preference1 = experiment_results[f"{description[1][0]}/{description[2][0]}"] - experiment_results[f"{description[1][0]}/{description[2][1]}"]
            preference2 = experiment_results[f"{description[1][1]}/{description[2][0]}"] - experiment_results[f"{description[1][1]}/{description[2][1]}"]
            add_to_output("Preferences:")
            print(f"Method{method}: Preference Group{description[1][0]} for {description[2][0]} over {description[2][1]} (Regular/Debiased): {preference1}")
            add_to_output(f"Method{method}: Preference Group{description[1][0]} for {description[2][0]} over {description[2][1]} (Regular/Debiased): {preference1}")
            print(f"Method{method}: Preference Group{description[1][1]} for {description[2][0]} over {description[2][1]} (Regular/Debiased): {preference2}")
            add_to_output(f"Method{method}: Preference Group{description[1][1]} for {description[2][0]} over {description[2][1]} (Regular/Debiased): {preference2}")
            print("")
            print(f"Method{method}: Difference: How much more does Group{description[1][0]} prefer {description[2][0]} (Regular/Debiased): {preference1 - preference2}")
            add_to_output(f"Method{method}: Difference: How much more does Group{description[1][0]} prefer {description[2][0]} (Regular/Debiased): {preference1 - preference2}")

            # experiments["Male_MaleOccupations"] = [["man"], occupations_male]


#Malebias: "preference"
# malebiases_male = experiment_results["Male_MaleOccupations"]-experiment_results["Male_FemaleOccupations"]
# malebiases_female = experiment_results["Female_MaleOccupations"]-experiment_results["Female_FemaleOccupations"]
# malebiases_person = experiment_results["Person_MaleOccupations"]-experiment_results["Person_FemaleOccupations"]
# bmalebiases_male = experiment_results["BMale_MaleOccupations"]-experiment_results["BMale_FemaleOccupations"]
# bmalebiases_female = experiment_results["BFemale_MaleOccupations"]-experiment_results["BFemale_FemaleOccupations"]
# bmalebiases_person = experiment_results["BPerson_MaleOccupations"]-experiment_results["BPerson_FemaleOccupations"]
# wmalebiases_male = experiment_results["WMale_MaleOccupations"]-experiment_results["WMale_FemaleOccupations"]
# wmalebiases_female = experiment_results["WFemale_MaleOccupations"]-experiment_results["WFemale_FemaleOccupations"]
# wmalebiases_person = experiment_results["WPerson_MaleOccupations"]-experiment_results["WPerson_FemaleOccupations"]


# print(f"Malebias - Male:{malebiases_male}")
# print(f"Malebias - Female:{malebiases_female}")
# print(f"Malebias - Person:{malebiases_person}")

# print(f"Malebias - Black Male:{bmalebiases_male}")
# print(f"Malebias - Black Female:{bmalebiases_female}")
# print(f"Malebias - Black Person:{bmalebiases_person}")

# print(f"Malebias - White Male:{wmalebiases_male}")
# print(f"Malebias - White Female:{wmalebiases_female}")
# print(f"Malebias - White Person:{wmalebiases_person}")

# print(f"Malebias - White minus Black, Male: {wmalebiases_male-bmalebiases_male}")
# print(f"Malebias - White minus Black, Female: {wmalebiases_female-bmalebiases_female}")
# print(f"Malebias - White minus Black, Person: {wmalebiases_person-bmalebiases_person}")

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

# experiments = {}
# experiments["Male_MaleOccupations"] = [["man"], occupations_male]
# experiments["Female_MaleOccupations"] = [["woman"], occupations_male]
# experiments["Person_MaleOccupations"] = [["person"], occupations_male]
# experiments["Male_FemaleOccupations"] = [["man"], occupations_female]
# experiments["Female_FemaleOccupations"] = [["woman"], occupations_female]
# experiments["Person_FemaleOccupations"] = [["person"], occupations_female]


# experiments["BMale_MaleOccupations"] = [["black man"], occupations_male]
# experiments["BFemale_MaleOccupations"] = [["black woman"], occupations_male]
# experiments["BPerson_MaleOccupations"] = [["black person"], occupations_male]
# experiments["BMale_FemaleOccupations"] = [["black man"], occupations_female]
# experiments["BFemale_FemaleOccupations"] = [["black woman"], occupations_female]
# experiments["BPerson_FemaleOccupations"] = [["black person"], occupations_female]

# experiments["WMale_MaleOccupations"] = [["white man"], occupations_male]
# experiments["WFemale_MaleOccupations"] = [["white woman"], occupations_male]
# experiments["WPerson_MaleOccupations"] = [["white person"], occupations_male]
# experiments["WMale_FemaleOccupations"] = [["white man"], occupations_female]
# experiments["WFemale_FemaleOccupations"] = [["white woman"], occupations_female]
# experiments["WPerson_FemaleOccupations"] = [["white person"], occupations_female]