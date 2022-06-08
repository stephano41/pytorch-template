import seaborn as sn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plotConfusionMatrix(model, val_loader, classes, device=None, fig_size=(14, 10)):
    all_outputs = []
    all_by = []

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            _, preds = torch.max(outputs, dim=1)
            all_outputs.extend(preds.data.cpu().numpy())
            all_by.extend(by.data.cpu().numpy())

    createConfusionMatrix(all_by, all_outputs, classes, fig_size=fig_size)

    plt.show()


def createConfusionMatrix(all_by, all_outputs, classes, fig_size=(14, 10), fig_title=None):
    cm = confusion_matrix(all_by, all_outputs)

    plt.figure(figsize=fig_size)
    s = sn.heatmap(cm, annot=True, fmt='g',xticklabels=classes, yticklabels=classes)
    if fig_title is not None:
        plt.suptitle(fig_title)
    s.set(xlabel='Predicted Labels', ylabel='True Labels')
    return s.get_figure()
