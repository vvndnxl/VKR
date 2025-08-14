from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torchmetrics
from functorch.dim import Tensor
from sklearn.metrics import roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch import nn, inference_mode
from matplotlib import pyplot as plt

from IPython.core.display_functions import display
from torchmetrics.functional import recall

dhj()


def normalize_rows(row):
    high = row.max() * 1.00001
    low = row.min() * 0.9999
    normalized_row = (row - low) / (high - low)
    return normalized_row


def load_dataset(main_path: str, secondary_paths: list[str]):
    torch.manual_seed(42)
    # with open(main_path, "r", encoding="utf8") as file:
    #     main_dataset = pd.read_csv(file, delimiter=";")
    #     main_dataset.columns = ([i for i in range(84 * 3 + 2)])
    #
    # secondary_dataset = pd.DataFrame()
    # for i in secondary_paths:
    #     with open(i, "r", encoding="utf8") as file:
    #         part = pd.read_csv(file, delimiter=";")
    #         part.columns = ([i for i in range(84 * 3 + 2)])
    #         secondary_dataset = pd.concat([secondary_dataset, part], ignore_index=True, sort=False)
    #
    # rate = len(main_dataset.index) // len(main_dataset.index) + 1
    # secondary_dataset = secondary_dataset.head(len(main_dataset.index) * rate)
    #
    # main_dataset = main_dataset.head(len(secondary_dataset.index))
    #
    # final_dataset = pd.concat([main_dataset, secondary_dataset], ignore_index=True, sort=False)
    # final_dataset = final_dataset.apply(normalize_rows, axis=1)
    #
    # x = torch.tensor(final_dataset.values).type(torch.float)
    # y = torch.from_numpy(np.array([1] * len(main_dataset.index) + [0] * len(secondary_dataset.index))).type(torch.float)
    #
    # pre_x_train, pre_x_test, pre_y_train, pre_y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # list_x_train = []
    # list_y_train = []
    # for i in range(pre_y_train.shape[0]):
    #     if pre_y_train[i] == 1:
    #         list_x_train += [pre_x_train[i]] * rate
    #         list_y_train += [1] * rate
    #     else:
    #         list_x_train += [pre_x_train[i]]
    #         list_y_train += [0]
    # x_train = torch.from_numpy(np.array(list_x_train)).type(torch.float)
    # y_train = torch.from_numpy(np.array(list_y_train)).type(torch.float)
    #
    # list_x_test = []
    # list_y_test = []
    # for i in range(pre_y_test.shape[0]):
    #     if pre_y_test[i] == 1:
    #         list_x_test += [pre_x_test[i]] * rate
    #         list_y_test += [1] * rate
    #     else:
    #         list_x_test += [pre_x_test[i]]
    #         list_y_test += [0]
    # x_test = torch.from_numpy(np.array(list_x_test)).type(torch.float)
    # y_test = torch.from_numpy(np.array(list_y_test)).type(torch.float)
    #
    # return x_train, x_test, y_train, y_test

    main_df = pd.read_csv(main_path, sep=';', header=None, encoding='utf8')
    main_df.columns = range(main_df.shape[1])
    main_df = main_df.apply(normalize_rows, axis=1)
    main_df['label'] = 1

    sec_list = []
    for p in secondary_paths:
        df = pd.read_csv(p, sep=';', header=None, encoding='utf8')
        df.columns = range(df.shape[1])
        df = df.apply(normalize_rows, axis=1)
        df['label'] = 0
        sec_list.append(df)
    sec_df = pd.concat(sec_list, ignore_index=True)

    all_df = pd.concat([main_df, sec_df], ignore_index=True)

    x = all_df.drop('label', axis=1).values.astype(np.float32)
    y = all_df['label'].values.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=55, stratify=y)

    train_df = pd.DataFrame(x_train)
    train_df['label'] = y_train

    counts = train_df['label'].value_counts()
    minority_label = counts.idxmin()
    n_max = counts.max()

    df_min = train_df[train_df['label'] == minority_label]
    df_min_upsampled = resample(
        df_min,
        replace=True,
        n_samples=n_max,
        random_state=55
    )

    df_maj = train_df[train_df['label'] != minority_label]
    balanced_train = pd.concat([df_maj, df_min_upsampled])

    x_train_bal = torch.tensor(balanced_train.drop('label', axis=1).values).float()
    y_train_bal = torch.tensor(balanced_train['label'].values).float()
    x_test_t = torch.tensor(x_test).float()
    y_test_t = torch.tensor(y_test).float()

    return x_train_bal, x_test_t, y_train_bal, y_test_t


class Model:
    def __init__(self, layers, data, epochs, target_loss):
        torch.manual_seed(42)
        struct = []
        for i in range(len(layers) - 1):
            torch.manual_seed(42)
            struct.append(('linear' + str(i + 1), nn.Linear(in_features=layers[i], out_features=layers[i + 1])))
            torch.manual_seed(42)
            struct.append(('relu' + str(i + 1), nn.ReLU()))
        torch.manual_seed(42)
        struct.append(('linear' + str(len(layers)), nn.Linear(in_features=layers[-1], out_features=1)))
        torch.manual_seed(42)
        self.model = nn.Sequential(OrderedDict(struct))
        torch.manual_seed(42)
        self.loss_fn = nn.BCEWithLogitsLoss()
        torch.manual_seed(42)
        self.optimazer = torch.optim.Adam(params=self.model.parameters())
        self.epochs = epochs
        self.target_loss = target_loss

        self.x_train, self.x_test, self.y_train, self.y_test = data


# noinspection PyUnboundLocalVariable
def train_model(model: Model, main_class, sec_class, printing=False):
    torch.manual_seed(42)
    accuracy_calc = torchmetrics.Accuracy(task="binary")
    precision_calc = torchmetrics.Precision(task="binary")
    recall_calc = torchmetrics.Recall(task="binary")
    # stat_scores = torchmetrics.classification.StatScores(reduce="micro", task="binary")
    # recall_calc = torchmetrics.Recall(task="binary")
    # ROC = torchmetrics.ROC(task="binary")
    target_loss = model.target_loss
    max_prec = 0
    max_prec_index = 0
    loss_sum = 0

    min_loss = 1
    min_loss_index = 0

    train_losses = []
    train_acc = []
    train_prec = []
    train_recall = []
    test_losses = []
    test_acc_list = []
    test_prec_list = []
    test_recall_list = []
    var = []
    train_F1 = []
    test_F1_list = []

    for epoch in range(model.epochs):
        torch.manual_seed(42)
        model.model.train()

        y_logits = model.model(model.x_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # TP, FP, TN, FN, SUP = stat_scores(y_pred, y_train)
        torch.manual_seed(42)
        loss = model.loss_fn(y_logits, model.y_train)
        acc = accuracy_calc(y_pred, model.y_train)
        prec = precision_calc(y_pred, model.y_train)
        recall = recall_calc(y_pred, model.y_train)
        F1 = f1_score(model.y_train.detach().numpy(), y_pred.detach().numpy(), average="binary")
        torch.manual_seed(42)
        model.optimazer.zero_grad()
        torch.manual_seed(42)
        loss.backward()
        torch.manual_seed(42)
        model.optimazer.step()
        torch.manual_seed(42)
        model.model.eval()
        with torch.inference_mode():
            test_logits = model.model(model.x_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = model.loss_fn(test_logits, model.y_test)
            test_acc = accuracy_calc(test_pred, model.y_test)
            test_prec = precision_calc(test_pred, model.y_test)
            test_recall = recall_calc(test_pred, model.y_test)
            test_F1 = f1_score(model.y_test.detach().numpy(), test_pred.detach().numpy(), average="binary")

        # if epoch > 60 and test_loss < min_loss:
        #     min_loss = test_loss
        #     min_loss_index = epoch

        preds_variance = sum([(0.5 - abs(float(i) - 0.5)) ** 2 for i in torch.sigmoid(y_logits)])
        train_losses += [float(loss)]
        train_acc += [float(acc)]
        train_prec += [float(prec)]
        test_losses += [float(test_loss)]
        test_acc_list += [float(test_acc)]
        test_prec_list += [float(test_prec)]
        var += [preds_variance]
        test_F1_list += [test_F1]
        train_F1 += [F1]
        test_recall_list += [test_recall]
        train_recall += [recall]

        loss_sum += test_loss
        if epoch > 40 and test_loss < loss_sum / epoch and test_prec > max_prec:
            max_prec = test_prec
            # max_prec_index = epoch

        if epoch > 50 and test_loss < min_loss:
            min_loss = test_loss
            # min_loss_index = epoch

        if epoch > 50 and test_loss <= target_loss:
            break

        # if printing and epoch % 5 == 0:
        # print(f"e: {epoch} loss: {loss:.5f} prec: {prec * 100:.2f}% t_loss: {test_loss:.5f} t_acc: {test_acc * 100:.2f}%"
        #       f" t_prec: {test_prec * 100:.2f}% var: {round(preds_variance, 3)}")
    print(max_prec, test_prec, min_loss, test_loss)
    if printing:
        xx = [i for i in range(epoch + 1)]
        # plt.plot(xx, [min(1, i) for i in train_losses], lw=1, alpha=0.2, color="blue")
        #
        #
        # plt.plot(xx, test_F1_list, lw=2, alpha=0.4, color="brown", label="F1")
        #
        # # plt.plot(xx, [min(1, i) for i in train_losses], lw=1, alpha=0.2, color="blue")
        # plt.plot(xx, [min(1, i) for i in test_losses], lw=2, alpha=0.4, color="blue", label="Losses")
        #
        # # plt.plot(xx, train_acc, lw=1, alpha=0.2, color="brown")
        # plt.plot(xx, test_acc_list, lw=2, alpha=0.3, color="green", label="Accuracy")
        # # plt.plot(xx, [min(1, i) for i in var], lw=2, alpha=0.9, color="pink")
        #
        # # plt.plot(xx, train_prec, lw=2, alpha=0.8, color="yellow")
        # plt.plot(xx, test_prec_list, lw=2, alpha=0.5, color="red", label="Precision")
        # plt.grid()
        # plt.legend()
        #
        # plt.savefig("m/" + str(main_class) + "_" + str(epoch) + "_" + str(round(float(min_loss), 2)) + ".png")
        # plt.close()
        #
        #
        plt.plot(xx, train_acc, lw=2, alpha=0.4, color="red", label="Train")
        # plt.plot(xx, [min(1, i) for i in test_losses], lw=2, alpha=0.4, color="blue", label="Losses")
        plt.plot(xx, test_acc_list, lw=2, alpha=0.3, color="green", label="Validation")
        # plt.plot(xx, [min(1, i) for i in var], lw=2, alpha=0.9, color="pink")

        # plt.plot(xx, train_prec, lw=2, alpha=0.8, color="yellow")
        # plt.plot(xx, test_prec_list, lw=2, alpha=0.5, color="red", label="Precision")
        plt.grid()
        plt.legend()
        plt.title("Accuracy")

        plt.savefig("m/" + str(main_class) + "_" + str(sec_class) + "_" + str(epoch) + "Acc.png")
        plt.close()
        #
        #
        plt.plot(xx, train_prec, lw=2, alpha=0.4, color="red", label="Train")
        plt.plot(xx, test_prec_list, lw=2, alpha=0.4, color="green", label="Validation")
        plt.grid()
        plt.legend()
        plt.title("Precision")
        plt.savefig("m/" + str(main_class) + "_" + str(sec_class) + "_" + str(epoch) + "Pre.png")
        plt.close()
        #
        #
        plt.plot(xx, train_recall, lw=2, alpha=0.4, color="red", label="Train")
        plt.plot(xx, test_recall_list, lw=2, alpha=0.4, color="green", label="Validation")
        plt.grid()
        plt.legend()
        plt.title("Recall")

        plt.savefig("m/" + str(main_class) + "_" + str(sec_class) + "_" + str(epoch) + "Rec.png")
        plt.close()
        #
        #
        #

        plt.figure(figsize=(10, 8))
        fpr, tpr, thresholds = roc_curve(model.y_test, test_logits, pos_label=1)
        plt.plot(fpr, tpr, lw=2, label="ROC")
        plt.plot([0, 1], [0, 1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig("m/" + str(main_class) + "_" + str(sec_class) + "_" + str(epoch) + "ROC.png")
        plt.close()

    return min_loss


def set_model(layers, main_dataset, secondary_dataset, printing=False):
    model = Model(layers, load_dataset(main_dataset, secondary_dataset), 1000, 0)
    min_loss = train_model(model, main_dataset, secondary_dataset[0], False)
    model = Model(layers, load_dataset(main_dataset, secondary_dataset), 1000, min_loss)
    min_loss_2 = train_model(model, main_dataset, secondary_dataset[0], printing)
    while min_loss_2 > min_loss:
        min_loss = min_loss_2
        model = Model(layers, load_dataset(main_dataset, secondary_dataset), 1000, min_loss)
        min_loss_2 = train_model(model, main_dataset, secondary_dataset[0], printing)
    return model.model
