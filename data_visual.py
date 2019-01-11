def data_visual():
    import pandas as pd
    import seaborn as sns
    from pandas.plotting import scatter_matrix
    import matplotlib.pyplot as plt
    import warnings
    
    warnings.filterwarnings("ignore")
    features = pd.read_csv('features.csv') 
    labels = pd.read_csv('labels.csv')

    names = ['valence', 'arousal', 'dominance', 'liking']
    dataset =features.transpose().reindex()


# Summary of dataset
    print('Summary of dataset\n')
# shape
    print('1. Shape is:')
    print(dataset.shape)
# head
    print('\n2. First 8 rows are as:')
    print(dataset.head(8))
# descriptions
    print('\n3. Statistical description:')
    print(dataset.describe())
    types = dataset.dtypes
    print('\n4. Data Types:')
    print(types)

# Data visualisation
# box and whisker plots
    for i in range(5):
        sns.set_style('whitegrid')
        sns.boxplot(dataset[i])
        plt.title('Magnitude versus ' + str(i+1) + ' channel')
        plt.show()
    
    import seaborn as sns
    from matplotlib import pyplot as plt
    for i in range(5):
        sns.distplot(dataset[i])
        plt.title('Magnitude versus ' + str(i+1) + ' channel')
        plt.show()
if __name__ == '__main__':
   data_visual()
