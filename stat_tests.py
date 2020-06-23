import math
from scipy import stats
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

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

def mcnemarTest(outputPath, bias="pos"):
    '''
    :param: bias (default="pos"), not yet implemented!
    :param: outputPath: the path to the model results being tested against the experts
    :return: p1 - the p-value representing the difference between the model and expert 1
             p2 - the p-value representing the difference between the model and expert 2

    '''
    testSet = pd.read_csv("test_set.csv")
    experts = pd.read_csv("experts.csv")
    model = pd.read_csv(outputPath, error_bad_lines=False)
    skips = 0
    a1 = b1 = c1 = d1 = 0 #initialize confusion matrix to be 0 in all four cells (expert 1)
    a2 = b2 = c2 = d2 = 0 #initialize confusion matrix to be 0 in all four cells (expert 2)

    for idx, row in testSet.iterrows():
        try:
            currId = row['patientID'] #CURRENT PATIENT
            expertRow = experts.query('patientID == "%s"' % currId)
            expertOne = expertRow['expert1'].values[0] #Expert one prediction
            expertTwo = expertRow['expert2'].values[0] #Expert two prediction
            modelRow = model.query('PatientID == "%s"' % currId)
            modelPred = modelRow['Predicted label'].values[0] #Model's prediction
            trueRow = testSet.query('patientID == "%s"' % currId)
            if bias == "pos":
                if expertOne == 2:
                    expertOne = 1
                else:
                    expertOne = 0
                if expertTwo == 2:
                    expertTwo = 1
                else:
                    expertTwo = 0
                truth = trueRow['outcome_pos'].values[0]
            if bias == "neg":
                if expertOne == 2:
                    expertOne = 1
                if expertTwo == 2:
                    expertTwo = 1
                truth = trueRow['outcome_neg'].values[0]
            if bias == "3":
                truth = trueRow['outcome_3'].values[0]
            # print(expertOne, expertTwo, modelPred)
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
        except:
            # print("SKIPPED HERE:", currId)
            skips += 1
    table1 = [[a1, b1],
              [c1, d1]]
    table2 = [[a2, b2],
              [c2, d2]]
    stat1, p1 = mcnemar(table1, exact=True).statistic, mcnemar(table1, exact=True).pvalue  #calculate test stat using exact binomial test
    stat2, p2 = mcnemar(table2, exact=True).statistic, mcnemar(table2, exact=True).pvalue
    return p1, p2






if __name__ == "__main__":
#     print(z_test_proportions(228/600, 132/400, 600, 400))
#     print(z_test_proportions(40/200, 20/200, 200, 200))
#     print(mcnemarTest("output/test_results/0dc80db6-a8be-494d-bf8a-3f436bf32aa1-v2.csv"))
    pass