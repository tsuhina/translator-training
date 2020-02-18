import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

def plot_correlation(data, var1, var2):
    data = (data
            .copy()
            [[var1, var2]]
            .dropna())
    colnames = data.columns

    # Scale:
    data = data.transform(lambda x: (x - x.mean()) / x.std())

    # Def inputs and targets:
    inputs = data[var1]
    targets = data[var2]

    # Specify model and fit it:
    model = sm.OLS(targets, inputs).fit()
    predictions = model.predict(inputs)

    # Plot:
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(inputs, targets, color='k', label='data', alpha=0.3)
    ax.plot(inputs, predictions, color='r', label='regression')
    ax.set_xlabel(f'{var1} scaled')
    ax.set_ylabel(f'{var2} scaled')
    plt.legend()

    print(f'R^2: {model.rsquared_adj:.2f}')
    print(f'Prob. that correlation insignificant: {model.pvalues[0]:.2f}')

def plot_effect_of_median_imputation(dataf, col):
    dataf = dataf.copy()

    dataf = dataf.assign(imputed = lambda x:np.where(dataf[col].isnull(),
                                                     dataf[col].median(),
                                                     dataf[col]))

    # Plot
    fig, (ax1, ax2) = plt.subplots(figsize=(15, 8), nrows=1, ncols=2)

    ax2.hist(dataf['imputed'],
             bins=np.linspace(int(dataf[col].min()), int(dataf[col].max()), 30),
             alpha=0.4,
             label='Median Imputation',
             density=False,
             color='blue')
    ymin, ymax = ax2.get_ylim()

    ax1.hist(dataf[col],
             bins=np.linspace(int(dataf[col].min()), int(dataf[col].max()), 30),
             alpha=0.4, label='No Imputation',
             density=False,
             color='red')
    ax1.set_ylim([ymin, ymax])

    ax1.axvline(dataf[col].median(),
                ls='--',
                color='k',
                label=f'median of {col}')

    ax2.axvline(dataf[col].median(),
            ls='--',
            color='k',
            label=f'median of {col}')

    ax1.legend()
    ax2.legend()

    ax1.set_xlabel(str(col))
    ax1.set_ylabel('Frequency')

    ax2.set_xlabel(str(col))
    ax2.set_ylabel('Frequency')

    fig.suptitle(f'Result of imputing a column {col}.')

def make_boxplot_example():
    x = np.random.normal(loc=50, scale=20, size=1000)

    qs = np.percentile(x, [25, 50, 75])

    fig, (ax1, ax2) = plt.subplots(figsize=(15, 12), nrows=2)

    ax1.hist(x, bins=20)

    cols = ['k', 'r', 'k']
    labels = ['25th percentile', 'median', '75th percentile']
    for idx, q in enumerate(qs):
        ax1.axvline(q, color=cols[idx],
                    ls='--',
                    label=labels[idx],
                    lw=3)
    ax1.annotate('Half of the values\n are in this region', [38, 40])
    ax1.set_title('Example histogram of normal dist, with $\mu$ = 50 and std=20')
    ax1.legend();

    sns.boxplot(x, ax=ax2)
    ax2.set_title('Same distribution from above (boxplot)');
