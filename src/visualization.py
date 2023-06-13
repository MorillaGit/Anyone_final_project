# Third party libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def sns_displot(series, title, x, y):

    # plotting data on distribution chart
    plt.figure(figsize=(10, 5))
    chart = sns.displot(
        # data=X_train,
        x=series,
        kind="kde",
        fill=True,
        height=5,
        aspect=1.5,
    )

    ##calculate statistics
    median = series.median()
    mean = round(series.mean(), 2)
    mode = series.mode()[0]

    # add vertical line to show statistical info
    plt.axvline(x=mode, color="b", ls=":", lw=2.5, label=f"Mode:{mode:,}")
    plt.axvline(
        x=median, color="black", ls="--", lw=2, label=f"Median: {median:,}"
    )
    plt.axvline(x=mean, color="r", ls="-", lw=1.5, label=f"Mean: {mean:,}")

    # customize plot
    chart.set(title=title)
    plt.xlabel(x, fontsize=10, labelpad=25)
    plt.ylabel(y, fontsize=10, labelpad=25)
    plt.legend(fontsize=12, shadow=True)

    return plt.show()


def missing_values_table(df):
    mis_val = df.isnull().sum()
    if mis_val.sum() == 0:
        print("No missing values found.")
        return
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: "Total", 1: "Percent"})
    mis_val_table_ren_columns = (
        mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0]
        .sort_values("Percent", ascending=False)
        .round(1)
    )
    print(
        "There are "
        + str(mis_val_table_ren_columns.shape[0])
        + " columns that have missing values."
    )
    # display(mis_val_table_ren_columns)
    return mis_val_table_ren_columns


def series_count(series):


    series_count = series.value_counts(dropna=False)
    series_perc = round(
        series.value_counts(normalize=True, dropna=False) * 100, 2
    )

    return pd.DataFrame({"Count": series_count, "%": series_perc})