import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('data/dataset.csv')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(x='Target', hue='Gender', data=df, multiple='dodge', stat='density', ax=ax)
    fig.savefig('figs/gender.svg')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(x='Age at enrollment', hue='Target', data=df, ax=ax)
    fig.savefig('figs/age.svg')