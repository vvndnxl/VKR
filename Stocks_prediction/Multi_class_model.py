import numpy as np
import pandas as pd
import torch
import torchmetrics
from matplotlib import pyplot as plt
# from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import Binary_model

reversal_model = Binary_model.set_model([254, 512, 256], "reversal_2_v3.csv",
                                        ["trend_follow_2_v2.csv",
                                         "breakout_0.csv",
                                         "consolidation_0_v2.csv"], True)

consolidation_model = Binary_model.set_model([254, 512, 256], "consolidation_0_v2.csv",
                                             ["reversal_2_v3.csv",
                                              "trend_follow_2_v2.csv",
                                              "breakout_0.csv"], True)

trend_model = Binary_model.set_model([254, 512, 256], "trend_follow_2_v2.csv",
                                     ["reversal_2_v3.csv",
                                      "breakout_0.csv",
                                      "consolidation_0_v2.csv"], True)

breakout_model = Binary_model.set_model([254, 512, 256], "breakout_0.csv",
                                        ["reversal_2_v3.csv",
                                         "trend_follow_2_v2.csv",
                                         "consolidation_0_v2.csv"], True)

class_models = [reversal_model, consolidation_model, trend_model, breakout_model]


#
# r_c = Binary_model.set_model([254, 512, 256], "reversal_2_v3.csv", ["consolidation_0_v2.csv"], True)
#
# r_t = Binary_model.set_model([254, 512, 256], "reversal_2_v3.csv", ["trend_follow_2_v2.csv"], True)
#
# r_b = Binary_model.set_model([254, 512, 256], "reversal_2_v3.csv", ["breakout_0.csv"], True)
#
# c_t = Binary_model.set_model([254, 512, 256], "consolidation_0_v2.csv", ["trend_follow_2_v2.csv"], True)
#
# c_b = Binary_model.set_model([254, 512, 256], "consolidation_0_v2.csv", ["breakout_0.csv"], True)
#
# t_b = Binary_model.set_model([254, 512, 256], "trend_follow_2_v2.csv", ["breakout_0.csv"], True)
#
# class_models = [r_c, r_t, r_b, c_t, c_b, t_b]


# noinspection PyShadowingNames
def load_test_dataset(secondary_classes):
    # with open(main_class, "r", encoding="utf8") as file:
    #     main_dataset = pd.read_csv(file, delimiter=";")
    #     main_dataset.columns = ([i for i in range(84 * 3 + 2)])
    lens = []

    final_dataset = pd.DataFrame()
    for i in secondary_classes:
        with open(i, "r", encoding="utf8") as file:
            part = pd.read_csv(file, delimiter=";", header=None)
            part.columns = ([i for i in range(84 * 3 + 2)])
            final_dataset = pd.concat([final_dataset, part], ignore_index=True, sort=False)
            lens += [len(part)]

    final_dataset = final_dataset.apply(Binary_model.normalize_rows, axis=1)

    x = torch.tensor(final_dataset.values).type(torch.float)
    y = []
    for i in range(len(secondary_classes)):
        y += [i] * lens[i]

    # y = torch.from_numpy(np.array(y_pre)).type(torch.float)
    return x, y


# noinspection PyShadowingNames,PyPep8Naming
def classification_OvO(data, models):
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    N = data.shape[0]

    preds = {pair: np.zeros(N, dtype=int) for pair in pairs}
    for (pair, model) in zip(pairs, models):
        logits = model(data)
        probs = torch.sigmoid(logits).detach().numpy().flatten()
        preds[pair] = np.round(probs).astype(int)
    votes = np.zeros((N, 4), dtype=int)

    for (i, j) in pairs:
        arr = np.asarray(preds[(i, j)])
        votes[:, i] += (arr == 1)
        votes[:, j] += (arr == 0)

    final = np.zeros(N, dtype=int)

    for k in range(N):
        max_votes = votes[k].max()
        winners = np.where(votes[k] == max_votes)[0]
        if len(winners) == 1:
            final[k] = winners[0]
        elif len(winners) == 2:
            a, b = sorted(winners)
            tie_arr = np.asarray(preds[(a, b)])
            if tie_arr[k] == 1:
                final[k] = a
            else:
                final[k] = b
        else:
            final[k] = int(winners.min())

    return final


# noinspection PyShadowingNames
def classification(data, models):
    preds = []
    for i in models:
        preds += [i(data)]
    counter = 0
    m = [0] * (len(preds[0].detach().numpy()))
    result = [0] * len(m)
    for res in preds:
        res_np = res.detach().numpy()
        for i in range(len(res_np)):
            if res_np[i] > m[i]:
                m[i] = res_np[i]
                result[i] = counter
        counter += 1
    return result


# noinspection PyShadowingNames
def classification_old(data, models):
    preds = []
    for i in models:
        test_logits = i(data)
        preds += [torch.round(torch.sigmoid(test_logits))]
    result = "None"
    for res in preds.keys():
        if preds[res] > 0.2:
            if preds[res] > 0.5 and result == "None":
                result = res
            break

    return result


x, y_pre = load_test_dataset(
    ["reversal_2_v3_x4.csv", "consolidation_0_v2.csv", "trend_follow_2_v2.csv", "breakout_0_x16.csv"])
# preds_pre = classification_OvO(x, class_models)
preds_pre = classification(x, class_models)

y = torch.from_numpy(np.array(y_pre)).type(torch.int)
preds = torch.from_numpy(np.array(preds_pre)).type(torch.float)

accuracy_calc = torchmetrics.Accuracy(task="multiclass", num_classes=4, average="none")
precision_calc = torchmetrics.Precision(task="multiclass", num_classes=4, average="none")
recall_calc = torchmetrics.Recall(task="multiclass", num_classes=4, average="none")
print(accuracy_calc(preds, y))
print(precision_calc(preds, y))
print(recall_calc(preds, y))
accuracy_calc = torchmetrics.Accuracy(task="multiclass", num_classes=4, average="macro")
precision_calc = torchmetrics.Precision(task="multiclass", num_classes=4, average="macro")
recall_calc = torchmetrics.Recall(task="multiclass", num_classes=4, average="macro")
print(accuracy_calc(preds, y))
print(precision_calc(preds, y))
print(recall_calc(preds, y))
accuracy_calc = torchmetrics.Accuracy(task="multiclass", num_classes=4, average="micro")
precision_calc = torchmetrics.Precision(task="multiclass", num_classes=4, average="micro")
recall_calc = torchmetrics.Recall(task="multiclass", num_classes=4, average="micro")
print(accuracy_calc(preds, y))
print(precision_calc(preds, y))
print(recall_calc(preds, y))

# roc = torchmetrics.ROC(task="multiclass", num_classes=3)
# plt.figure(figsize=(10, 8))
# fpr, tpr, thresholds = roc(preds, y)
# plt.plot(fpr, tpr, lw=2, label="ROC")
# plt.plot([0, 1], [0, 1])
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.savefig("t/ROC.png")
# plt.close()


# auc = metrics.roc_auc_score(y, preds, multi_class="ovr")
# print(auc)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2)
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

cm = confusion_matrix(y_pre, preds_pre)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4])

# noinspection PyUnresolvedReferences
disp.plot(cmap=plt.cm.Blues)
plt.title("Multiclass Confusion Matrix")
plt.show()
