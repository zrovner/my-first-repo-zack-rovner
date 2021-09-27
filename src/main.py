import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    log = open('../logs/logfile.txt', 'w')

    log.write("Importing data...")

    data = pd.read_csv('../data/classification_data_set.csv')

    log.write("Creating histogram plots...")

    for col in data.drop('Target', axis=1).columns:

        sns.displot(data, x=col, hue='Target')
        plt.savefig(f'../outputs/class_separated_histogram_of_{col}')

    log.write("Okay, all done!")

    log.close()


if __name__ == '__main__':
    main()
