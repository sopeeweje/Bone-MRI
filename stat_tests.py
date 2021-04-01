import math
from scipy import stats
from scipy.stats import chisquare, chi2_contingency
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
import pandas as pd
from statsmodels.stats.libqsturng import psturng, qsturng

def games_howell(benign_list, intermediate_list, malignant_list, alpha=.05):
    '''
    given three lists of ages, return the p value for the difference using games-howell test
    lists must be inputted in correct order!
    '''
    benign_mean = sum(benign_list)/len(benign_list)
    intermediate_mean = sum(intermediate_list) / len(intermediate_list)
    malignant_mean = sum(malignant_list) / len(malignant_list)
    k = 3 #number of classes
    benign_var = sum([(age - benign_mean) ** 2 for age in benign_list]) / (len(benign_list)-1)
    intermediate_var = sum([(age - intermediate_mean) ** 2 for age in intermediate_list]) / (len(intermediate_list) - 1)
    malignant_var = sum([(age - malignant_mean) ** 2 for age in malignant_list]) / (len(malignant_list) - 1)

    group_means = {'benign':benign_mean, 'intermediate':intermediate_mean, 'malignant':malignant_mean}
    group_var = {'benign': benign_var, 'intermediate': intermediate_var, 'malignant': malignant_var}
    group_obs = {'benign': len(benign_list), 'intermediate': len(intermediate_list), 'malignant': len(malignant_list)}

    combs = [('benign', 'intermediate'), ('intermediate', 'malignant'), ('benign', 'malignant')]

    group_comps = []
    mean_differences = []
    degrees_freedom = []
    t_values = []
    p_values = []
    std_err = []
    up_conf = []
    low_conf = []

    for comb in combs:
        diff = group_means[comb[1]] - group_means[comb[0]]
        # t-value of each group combination
        t_val = np.abs(diff) / np.sqrt((group_var[comb[0]] / group_obs[comb[0]]) +
                                       (group_var[comb[1]] / group_obs[comb[1]]))
        # Numerator of the Welch-Satterthwaite equation
        df_num = (group_var[comb[0]] / group_obs[comb[0]] + group_var[comb[1]] / group_obs[comb[1]]) ** 2
        # Denominator of the Welch-Satterthwaite equation
        df_denom = ((group_var[comb[0]] / group_obs[comb[0]]) ** 2 / (group_obs[comb[0]] - 1) +
                    (group_var[comb[1]] / group_obs[comb[1]]) ** 2 / (group_obs[comb[1]] - 1))
        # Degrees of freedom
        df = df_num / df_denom
        # p-value of the group comparison
        p_val = psturng(t_val * np.sqrt(2), k, df)
        # Standard error of each group combination
        se = np.sqrt(0.5 * (group_var[comb[0]] / group_obs[comb[0]] +
                            group_var[comb[1]] / group_obs[comb[1]]))
        # Upper and lower confidence intervals
        upper_conf = diff + qsturng(1 - alpha, k, df)
        lower_conf = diff - qsturng(1 - alpha, k, df)
        # Append the computed values to their respective lists.
        mean_differences.append(diff)
        degrees_freedom.append(df)
        t_values.append(t_val)
        p_values.append(p_val)
        std_err.append(se)
        up_conf.append(upper_conf)
        low_conf.append(lower_conf)
        group_comps.append(str(comb[0]) + ' : ' + str(comb[1]))

    result_df = pd.DataFrame({'groups': group_comps,
                              'mean_difference': mean_differences,
                              'std_error': std_err,
                              't_value': t_values,
                              'p_value': p_values,
                              'upper_limit': up_conf,
                              'lower limit': low_conf})
    return result_df

def z_test_proportions(acc1, acc2, n1, n2):
    '''
    :param acc1: accuracy (between 0 and 1) of model of interest
    :param acc2: accuracy (between 0 and 1) of model it is being compared to
    :param n1: the size of the test set for example 1
    :param n2: the size of the test set for example 2
    for our use, n1 and n2 will be the same, as long as the test sets are the same
    :return: the p-value representing whether or not acc1 is better than acc2 (one sided test!)
    '''
    pooledAcc = (acc1 * n1 + acc2 * n2) / (n1 + n2)
    z = (acc1 - acc2) / math.sqrt(pooledAcc * (1 - pooledAcc) * (1/n1 + 1/n2)) # calculate z-statistic
    p_values = stats.norm.sf(abs(z)) # 1-sided test
    return p_values

# if __name__ == "__main__":
#     print(z_test_proportions(228/600, 132/400, 600, 400))
#     print(z_test_proportions(40/200, 20/200, 200, 200))
def mcnemarTest(outputPath, bias="pos"):
    '''
    :param: bias (default="pos"), not yet implemented!
    :param: outputPath: the path to the model results being tested against the experts
    :return: p1 - the p-value representing the difference between the model and expert 1
             p2 - the p-value representing the difference between the model and expert 2

    '''
    testSet = pd.read_csv("test_set.csv")
    experts = pd.read_csv("expert.csv")
    model = pd.read_csv(outputPath, error_bad_lines=False)
    skips = 0
    a1 = b1 = c1 = d1 = 0 #initialize confusion matrix to be 0 in all four cells (expert 1)
    a2 = b2 = c2 = d2 = 0 #initialize confusion matrix to be 0 in all four cells (expert 2)
    for idx, row in testSet.iterrows():
        currId = row['patientID'] #CURRENT PATIENT
        trueRow = testSet.query('patientID == "%s"' % currId)
        truth = trueRow['outcome_pos'].values[0]
        if truth==0:
            expertRow = experts.query('patientID == "%s"' % currId)
            expertOne = expertRow['expert3'].values[0] #Expert one prediction
            expertTwo = expertRow['committee'].values[0] #Expert two prediction
            modelRow = model.query('patientID == "%s"' % currId)
            modelPred = modelRow['prediction'].values[0] #Model's prediction
            
            if expertOne == modelPred:
                if expertOne == truth:
                    a1 += 1
                else:
                    d1 += 1
            else:
                if expertOne == truth:
                    c1 += 1
                else:
                    b1 += 1
            if expertTwo == modelPred:
                if expertTwo == truth:
                    a2 += 1
                else:
                    d2 += 1
            else:
                if expertTwo == truth:
                    c2 += 1
                else:
                    b2 += 1
        else:
            continue
    table1 = [[a1, b1],
              [c1, d1]]
    table2 = [[a2, b2],
              [c2, d2]]
    print(table1)
    print(table2)
    print(skips)
    stat1, p1 = mcnemar(table1, exact=True).statistic, mcnemar(table1, exact=True).pvalue  #calculate test stat using exact binomial test
    stat2, p2 = mcnemar(table2, exact=True).statistic, mcnemar(table2, exact=True).pvalue
    return p1, p2

points = []
locations = [
            "Clavicle",
            "Cranium",
            "Proximal femur",
            "Distal femur",
            "Foot",
            "Proximal radius",
            "Distal radius",
            "Proximal ulna",
            "Distal ulna",
            "Hand",
            "Hip",
            "Proximal humerus",
            "Distal humerus",
            "Proximal tibia",
            "Distal tibia",
            "Proximal fibula",
            "Distal fibula",
            "Mandible",
            "Rib/Chest wall",
            "Scapula",
            "Spine",
        ]
d = open("bone_features.csv", 'r', encoding="utf-8-sig")
for line in d:
    values = line.strip().split(',')
    if values[0]=='patientID':
        continue
    points.append(
        {
            "patientID": values[0],
            "category": int(values[1]),
            "sort": values[2],
            "location": values[5],
        }
        )

#Benign vs malignant location comparison
#Actual
actual_benign = [i["location"] for i in points if i["category"] < 2]
actual_malignant = [i["location"] for i in points if i["category"] == 2]
all_location = [i["location"] for i in points]
num_ben = len(actual_benign)
num_mal = len(actual_malignant)
total = len(points)
actual = []
expected = []
for location in locations:
    actual.append([sum(i == location for i in actual_benign), sum(i == location for i in actual_malignant)])
    #expected.append([num_ben/total*sum(i == location for i in all_location), num_mal/total*sum(i == location for i in all_location)])

_, p, _, expected_array = chi2_contingency(actual)
print("Location by benign/malignant: {}".format(str(p)))

#Vs all others
for location in locations:
    actual = [[sum(i == location for i in actual_benign), sum(i == location for i in actual_malignant)],
              [sum(i != location for i in actual_benign), sum(i != location for i in actual_malignant)]
              ]
    _, p, _, expected_array = chi2_contingency(actual)
    #print(str(p))
    #print(actual)
    print("{} vs. all others: {}".format(location, str(p)))


#Train/val vs. internal vs. external location comparison
actual_trainval = [i["location"] for i in points if i["sort"] == ("train" or "validation")]
actual_test = [i["location"] for i in points if i["sort"] == "test"]
actual_ext = [i["location"] for i in points if i["sort"] == "external"]

all_location = [i["location"] for i in points]
num_trainval = len(actual_trainval)
num_test = len(actual_test)
num_ext = len(actual_ext)
total = len(points)
actual = []
expected = []
for location in locations:
    actual.append([sum(i == location for i in actual_trainval), sum(i == location for i in actual_test), sum(i == location for i in actual_ext)])
    
_, p, _, expected_array = chi2_contingency(actual)
print("Location by trainval/test/ext: {}".format(str(p)))

#Vs all others
for location in locations:
    actual = [[sum(i == location for i in actual_trainval), sum(i == location for i in actual_test), sum(i == location for i in actual_ext)],
              [sum(i != location for i in actual_trainval), sum(i != location for i in actual_test), sum(i != location for i in actual_ext)]
              ]
    _, p, _, expected_array = chi2_contingency(actual)
    print(str(p))
    #print(actual)
    #print("{} vs. all others: {}".format(location, str(p)))
#if __name__ == "__main__":
#     print(z_test_proportions(228/600, 132/400, 600, 400))
#     print(z_test_proportions(40/200, 20/200, 200, 200))
#     print(mcnemarTest("model_results.csv"))