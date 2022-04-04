import io

import unicodedata
import torch
import string
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
from torchvision.transforms import ToTensor

def findFiles(path): return glob.glob(path)


def classes(data_dir):
    categories = []
    for filename in findFiles(data_dir + '*.txt'):
        categories.append(os.path.splitext(os.path.basename(filename))[0])
    return categories


def all_letters():
    return string.ascii_letters + " .,;'"


def n_letters(all_letters=all_letters()):
    return len(all_letters)


def unicodeToAscii(s, all_letters=all_letters()):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def letterToIndex(letter, all_letters=all_letters()):
    return all_letters.find(letter)


def letterToTensor(letter, n_letters=n_letters()):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line, n_letters=n_letters()):
    tensor = torch.zeros(1, len(line), n_letters)
    for li, letter in enumerate(line):
        tensor[0][li][letterToIndex(letter)] = 1
    return tensor


def lineToTensor2(line, n_letters=n_letters()):
    tensor = torch.zeros(len(line), n_letters)
    for li, letter in enumerate(line):
        tensor[li][letterToIndex(letter)] = 1
    return tensor


# Assumes output from model is after softmax
def plotConfusionMatrix(model, val_loader, classes,
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    all_outputs = []
    all_by = []
    for bx, by in val_loader:
        bx, by = bx.to(device), by.to(device)
        outputs = model(bx)
        _, preds = torch.max(outputs, dim=1)
        all_outputs.extend(preds.data.cpu().numpy())
        all_by.extend(by.data.cpu().numpy())

    cm = confusion_matrix(all_by, all_outputs, normalize='pred')

    plt.figure(figsize=(14, 10))
    s = sn.heatmap(cm, annot=True, fmt='.2g', xticklabels=classes, yticklabels=classes)
    s.set(xlabel='Predicted Labels', ylabel='True Labels')

    plt.show()
    # cm = confusion_matrix(all_by, all_outputs, normalize='pred')
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # _,ax = plt.subplots(figsize=(14,10))
    # disp.plot(xticks_rotation='vertical', values_format='.2g',ax=ax)
    #
    # plt.show()


def createConfusionMatrix(all_by, all_outputs, classes):
    cm = confusion_matrix(all_by, all_outputs, normalize='pred')

    plt.figure(figsize=(14, 10))
    s = sn.heatmap(cm, annot=True, fmt='.2g', xticklabels=classes, yticklabels=classes)
    s.set(xlabel='Predicted Labels', ylabel='True Labels')
    return s.get_figure()


def plot_to_image(figure):
    # converts matplotlib plot to PNG image in memory, for logging to tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = Image.open(buf)
    image = ToTensor()(image)
    return image


def predict(line, model, all_categories, n_predictions=3):
    line = torch.unsqueeze(lineToTensor(line), 0)
    output = model(line)

    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions