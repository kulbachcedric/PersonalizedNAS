from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


POPULATION_SIZE = 100

def visualize(path:Path):
    df = pd.read_csv(str(path))
    cycles = len(df) - POPULATION_SIZE -1
    res = df.loc[0:POPULATION_SIZE-1, :] # first population
    res['epoch'] = [0] * res.shape[0]
    '''
    for cyc in range(cycles):
        pop = df.loc[cyc+1:POPULATION_SIZE+cyc, :]
        pop['epoch'] = [cyc + 1] * pop.shape[0]
        res = res.append(pop, ignore_index = True)
    '''
    df = df.loc[:175]
    fig = sns.scatterplot(x=df.index,y='agent_scores', data=df)
    fig.set(xlabel='epochs', ylabel='synthetic score')
    plt.show()



if __name__ == '__main__':
    path = Path('scores.csv')
    visualize(path=path)