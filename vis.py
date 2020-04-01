from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd 



if __name__ == '__main__':


    df = pd.read_csv('segments.csv')
    df = df.sort_values('agent_ranking',ascending=False).head(30)
    sns.scatterplot(data=df, y="ranknet_ranking", x="agent_ranking")
    #plt.ylim(-0.535,-0.525)
    plt.show()