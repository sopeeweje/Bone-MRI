import seaborn
import pandas
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

plt.rcParams['svg.fonttype'] = 'none'

#Data/inputs
MODALITIES = ["t2","t1","ensemble"]

MODALITY_KEY = {"t1": "T1", "t2": "T2","ensemble": "Ensemble"}

test_data = {
't2-labels': [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],

't2-predictions': [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],

't2-probabilities': [0.8641476631164551, 0.04850572347640991, 0.7620673179626465, 0.58588045835495, 0.6092981696128845, 0.6122311949729919, 0.12623029947280884, 0.7747927904129028, 0.15856879949569702, 0.06381136178970337, 0.9023292660713196, 0.16890358924865723, 0.8298861980438232, 0.8764317631721497, 0.8906347751617432, 0.8994308114051819, 0.7904530763626099, 0.12350928783416748, 0.7380993962287903, 0.6397010087966919, 0.8500899076461792, 0.6479206085205078, 0.824934720993042, 0.732590913772583, 0.7894371747970581, 0.042136043310165405, 0.8948793411254883, 0.44490575790405273, 0.39027678966522217, 0.7362145185470581, 0.2500121593475342, 0.4328390061855316, 0.07060438394546509, 0.7899795174598694, 0.8544600605964661, 0.9174122214317322, 0.7343297004699707, 0.2665799856185913, 0.16615453362464905, 0.6884598135948181, 0.8204212188720703, 0.739686906337738, 0.6048266291618347, 0.09930911660194397, 0.1174916923046112, 0.07426440715789795, 0.0853545069694519, 0.03623369336128235, 0.0715411901473999, 0.5343037843704224, 0.1768375039100647, 0.7952120900154114, 0.10444635152816772, 0.09747165441513062, 0.14067289233207703, 0.9195536375045776, 0.1374014914035797, 0.8023399114608765, 0.7738286256790161, 0.04844006896018982, 0.20444121956825256, 0.28635185956954956, 0.06836849451065063, 0.04106423258781433, 0.1978999376296997, 0.6794248819351196, 0.176733136177063, 0.19119012355804443, 0.06711602210998535, 0.04622530937194824, 0.04486727714538574, 0.8648874759674072, 0.612436830997467, 0.6400085687637329, 0.7742137908935547, 0.6396847367286682, 0.624040961265564, 0.7768769264221191, 0.825586199760437, 0.15972623229026794, 0.18210485577583313, 0.859200119972229, 0.5075121521949768, 0.7669910192489624, 0.7982043027877808, 0.6888650059700012, 0.6421077251434326, 0.8054935932159424, 0.6030933260917664, 0.8645918369293213, 0.7768886089324951, 0.0696733295917511, 0.06345227360725403, 0.6923015713691711, 0.7264444231987, 0.08084028959274292, 0.39755359292030334, 0.4451005160808563, 0.05254422500729561],

't1-labels': [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],

't1-predictions': [1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0],

't1-probabilities': [0.9736838340759277, 0.0575522780418396, 0.9703097343444824, 0.5550577640533447, 0.8246997594833374, 0.9441009163856506, 0.030132442712783813, 0.05984079837799072, 0.16673463582992554, 0.9090169668197632, 0.1835635006427765, 0.9448560476303101, 0.06209444999694824, 0.9532290697097778, 0.94771808385849, 0.3424533009529114, 0.9715927839279175, 0.9665617942810059, 0.9631568193435669, 0.058258771896362305, 0.9755184650421143, 0.07406848669052124, 0.7169244289398193, 0.9603069424629211, 0.059271037578582764, 0.8322021961212158, 0.04248619079589844, 0.23140576481819153, 0.6035823225975037, 0.7003685235977173, 0.8947073817253113, 0.06658560037612915, 0.06523194909095764, 0.6572996973991394, 0.9749966263771057, 0.9580935835838318, 0.9572224617004395, 0.9046764373779297, 0.9453080892562866, 0.7893975973129272, 0.07049959897994995, 0.8960615396499634, 0.12965095043182373, 0.25376734137535095, 0.7401649951934814, 0.9657443165779114, 0.6864199638366699, 0.07976996898651123, 0.9203075170516968, 0.12255886197090149, 0.059174805879592896, 0.9644816517829895, 0.06709843873977661, 0.2889905571937561, 0.9307191371917725, 0.10457736253738403, 0.13458749651908875, 0.6372830271720886, 0.9643328189849854, 0.25972917675971985, 0.6873335242271423, 0.7869564294815063, 0.11355921626091003, 0.07733052968978882, 0.09518373012542725, 0.10430794954299927, 0.04675692319869995, 0.047576189041137695, 0.9752347469329834, 0.3322891891002655, 0.04551449418067932, 0.10412406921386719, 0.617063045501709, 0.3063836693763733, 0.9791141748428345, 0.9001554250717163, 0.9507399797439575, 0.9648650288581848, 0.9014708995819092, 0.976244330406189, 0.7113308906555176, 0.9408591985702515, 0.9077733755111694, 0.9801218509674072, 0.03048500418663025, 0.05259418487548828, 0.905825138092041, 0.960034966468811, 0.9381749629974365, 0.9578847885131836, 0.028212517499923706, 0.9150264263153076, 0.7301052808761597, 0.5351958870887756, 0.09898960590362549, 0.9547865390777588, 0.07631246000528336, 0.15920136868953705, 0.04932134225964546],

'ensemble-labels': [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],

'ensemble-predictions': [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],

'ensemble-probabilities': [1.0, 0.0, 0.9, 0.15, 0.65, 0.6, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 0.95, 0.2, 1.0, 0.15, 0.65, 1.0, 0.5, 0.45, 0.5, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.55, 1.0, 1.0, 1.0, 0.5, 0.5, 0.85, 0.5, 0.85, 0.15, 0.0, 0.3, 0.5, 0.1, 0.0, 0.5, 0.1, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.55, 1.0, 0.0, 0.1, 0.45, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.1, 0.2, 1.0, 0.7, 0.7, 1.0, 1.0, 0.5, 0.15, 1.0, 0.5, 1.0, 0.45, 0.4, 0.7, 1.0, 0.7, 1.0, 0.5, 0.5, 0.25, 0.4, 0.5, 0.5, 0.0, 0.0, 0.0]

}

names = ['bone-hup-137', 'bone-hup-130', 'bone-hup-120', 'bone-hup-143', 'bone-hup-127', 'bone-penn-295', 'bone-penn-594', 'bone-penn-582', 'bone-penn-574', 'bone-penn-125', 'bone-penn-407', 'bone-penn-432', 'bone-penn-117', 'bone-penn-393', 'bone-penn-389', 'bone-penn-387', 'bone-penn-229', 'bone-hup-174', 'bone-hup-190', 'bone-hup-238', 'bone-hup-164', 'bone-hup-212', 'bone-hup-195', 'bone-hup-236', 'bone-hup-182', 'bone-penn-290', 'bone-penn-202', 'bone-penn-184', 'bone-penn-208', 'bone-penn-175', 'bone-penn-185', 'bone-penn-103', 'bone-penn-102', 'bone-penn-80', 'bone-hup-114', 'bone-hup-87', 'bone-hup-99', 'bone-hup-90', 'bone-penn-544', 'bone-penn-547', 'bone-penn-135', 'bone-penn-145', 'bone-penn-366', 'bone-penn-383', 'bone-penn-519', 'bone-penn-540', 'bone-penn-524', 'bone-penn-539', 'bone-china-137', 'bone-china-150', 'bone-china-141', 'bone-china-142', 'bone-china-156', 'bone-china-157', 'bone-china-094', 'bone-china-108', 'bone-china-130', 'bone-hup-294', 'bone-hup-318', 'bone-hup-300', 'bone-hup-313', 'bone-hup-311', 'bone-penn-326', 'bone-china-067', 'bone-china-082', 'bone-china-090', 'bone-china-012', 'bone-china-064', 'bone-china-063', 'bone-china-007', 'bone-china-018', 'bone-china-013', 'bone-hup-50', 'bone-penn-638', 'bone-penn-649', 'bone-china-206', 'bone-china-235', 'bone-china-232', 'bone-china-182', 'bone-china-229', 'bone-china-208', 'bone-china-181', 'bone-china-202', 'bone-china-241', 'bone-china-233', 'bone-penn-469', 'bone-penn-486', 'bone-penn-505', 'bone-penn-513', 'bone-penn-499', 'bone-penn-470', 'bone-penn-510', 'bone-penn-475', 'bone-penn-493', 'bone-penn-559', 'bone-penn-569', 'bone-penn-554', 'bone-penn-566', 'bone-penn-557']

#test by andy to see if this appears
#functions
def transform_binary_predictions(results):
    predictions = 1 * (results.flatten() > 0.5)
    return predictions

def calculate_confusion_matrix(labels, results):
    """
    returns a confusion matrix
    """
    predictions = transform_binary_predictions(results)
    return confusion_matrix(labels, predictions)

def calculate_confusion_matrix_stats(labels, results):
    confusion_matrix = calculate_confusion_matrix(labels, results)
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    Acc = (TN + TP)/(TN + TP + FN + FP)
    return {
        "Acc": Acc,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TPR": TPR,
        "TNR": TNR,
        "PPV": PPV,
        "NPV": NPV,
        "FPR": FPR,
        "FNR": FNR,
        "FDR": FDR,
        "AM": (TPR+TNR)/2,
        "GM": np.sqrt(TPR*TNR),
    }

def calculate_confusion_matrix_predictions(labels, predictions):
    """
    returns a confusion matrix
    """
    return confusion_matrix(labels, predictions)

def calculate_confusion_matrix_stats_predictions(labels, predictions):
    confusion_matrix = calculate_confusion_matrix_predictions(labels, predictions)
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    Acc = (TN + TP)/(TN + TP + FN + FP)
    return {
        "Acc": Acc,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TPR": TPR,
        "TNR": TNR,
        "PPV": PPV,
        "NPV": NPV,
        "FPR": FPR,
        "FNR": FNR,
        "FDR": FDR,
        "AM": (TPR+TNR)/2,
        "GM": np.sqrt(TPR*TNR),
    }

def all_features(files=["./features.csv"], id_name="patientID"):
    by_file = list()
    for filename in files:
        with open(filename) as f:
            l = [ {k: v for k, v in row.items() } for row in csv.DictReader(f, skipinitialspace=True )]
            by_accession = { d[id_name]: d for d in l }
            by_file.append(by_accession)
    id_sets = [ set(f.keys()) for f in by_file ]
    union = id_sets[0]
    for ids in id_sets:
        union = union | ids
    combined = dict()
    for i in union:
        c = dict()
        for by_accession in by_file:
            c = {
                **c,
                **by_accession[i],
        }
        combined[i] = c
    return combined

def get_roc_data_for_modality(dataset):
    results = list()
    points = list()
    for modality in MODALITIES:
        labels = dataset["{}-labels".format(modality)]
        probabilities = dataset["{}-probabilities".format(modality)]
        predictions = dataset["{}-predictions".format(modality)]
        fpr, tpr, _ = roc_curve(labels, probabilities, drop_intermediate=False)
        roc_auc = roc_auc_score(labels, probabilities)
        stats = calculate_confusion_matrix_stats_predictions(labels, predictions)
        acc = accuracy_score(labels, predictions)
        points.append({
                      "modality": "{} (auc={:.2f}, acc={:.2f})".format(MODALITY_KEY[modality], roc_auc, acc),
                      "fpr": stats["FPR"][1],
                      "tpr": stats["TPR"][1],
                      })
        for f, t in zip(fpr, tpr):
          results.append({ "fpr": f, "tpr": t, "modality": "{} (auc={:.2f}, acc={:.2f})".format(MODALITY_KEY[modality], roc_auc, acc)})
    return results, roc_auc, []


def plot_multiple_roc_curve(dataset, experts=[]):
    #This is the function you'll need to edit
    results, auc, points = get_roc_data_for_modality(dataset)
    if len(experts) > 0:
        for i, expert in enumerate(experts):
            labels = dataset["t1-labels"]
            predictions = expert
            stats = calculate_confusion_matrix_stats_predictions(labels, predictions)
            acc = accuracy_score(labels, predictions)
            points.append({
                          "fpr": stats["FPR"][1],
                          "tpr": stats["TPR"][1],
                          "experts": "Expert {} (acc={:.2f})".format(i + 1, acc),
                          })
    fig, ax = plt.subplots()
    seaborn.lineplot(
                     data=pandas.DataFrame(results),
                     x="fpr",
                     y="tpr",
                     hue="modality",
                     ax=ax,
                     err_style=None,
                     )
    if points:
     seaborn.scatterplot(
                         data=pandas.DataFrame(points),
                         x="fpr",
                         y="tpr",
                         hue="experts",
                         style="experts",
                         ax=ax,
                         markers=["o", "v", "s", "P"],
                         palette={ p["experts"]: "black" for p in points },
                         )

    ax.plot([0, 1], [0, 1], linestyle='--', color='#929c95')
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.04, 1.02)
    handles, labels = ax.get_legend_handles_labels()
    # the below loops remove the labels and handles given by the hue argument for the modalities and experts
    toRemove = set()
    newLabels = []
    newHandles = []
    for idx, lab in enumerate(labels):
        if lab == "modality" or lab == "experts":
            toRemove.add(idx)
        else:
            newLabels.append(lab)
    for idx, hand in enumerate(handles):
        if idx not in toRemove:
            newHandles.append(hand)
    ax.xaxis.set_minor_locator(MultipleLocator(.05))
    ax.yaxis.set_minor_locator(MultipleLocator(.05))
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
        tick.label1.set_fontweight('bold')

    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    fig.suptitle('ROC Curve', fontsize=16, fontweight='bold', y= .93)
    plt.rcParams["axes.labelsize"] = 29
    ax.legend(frameon=False, handles=newHandles, labels=newLabels)
    return fig

def get_experts_for_names(features, names, experts=["expert1","expert2"], transform=int, default=0):
    result = list()
    for e in experts:
        expert_results = list()
        for n in names:
            f = features.get(n, None)
            if f is None:
                print("error, cannot find {}".format(n))
                expert_results.append(default)
                continue
            r = f.get(e, default)
            if r == "":
                r = 0
            elif r == "2":# or r == "1": #benign vs. not benign (outcome_neg)
                r = 1
            else:
                r = 0
            r = transform(r)
            expert_results.append(r)
        result.append(expert_results)
    return result

#what runs when you execute this script
if __name__ == '__main__':
    expert_features = all_features(files = ["experts.csv"])
    fig = plot_multiple_roc_curve(test_data, experts=get_experts_for_names(expert_features, names))
    fig.savefig("exampleROC.png", bbox_inches="tight")

