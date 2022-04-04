from utils.NLP import classes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_class_distribution(dataset, data_dir, dataloader = None):
    categories = classes(data_dir)
    count_dict = {k:0 for k in categories}

    if dataloader:
        for _,j in dataloader:
            for i in j:
                label = categories[i.item()]
                count_dict[label] += 1
    else:
        for element in dataset:
            label = categories[element[1].item()]
            count_dict[label] += 1

    return count_dict


def plot_class_distribution(dataset, data_dir, dataloader = None):
    plt.figure(figsize = (15,8))

    class_distribution = get_class_distribution(dataset, data_dir, dataloader)

    sns.barplot(data =
                pd.DataFrame.from_dict([class_distribution]).
                melt(), x="variable", y="value", hue="variable").set_title('Words Class Distribution')
    plt.show()
    return sum(class_distribution.values())