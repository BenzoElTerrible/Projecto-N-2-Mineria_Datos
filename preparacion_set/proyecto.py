import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('/home/benzo/Mineria_Datos/P2/creditcard_2023.csv')

# Imprime los atributos
df_head = df.head()
print("\n< Atributos >\n")
print(df_head)

# Se imprime el tamaño del dateset
print("Credit Card Fraud Detection data -  rows:",df.shape[0]," columns:", df.shape[1])

# Se describe el dataset
df_describe = df.describe().T
print("\n< Descripcion del data set>\n\n", df_describe)

# Se visualizan datos nulos 
df_null = df.isnull().sum().sort_values(ascending = False)
print("\n< Cantidad de NULLS por atributos >\n\n", df_null)

# Se visualizan desbalances
df_desbalance = df['Class'].value_counts()
print("\n< Visualizacion de desbalances >\n\n", df_desbalance)

# Se grafican los desbalances
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

sns.boxplot(ax=ax1, x="Class", y="Amount", hue="Class", data=df, palette="PRGn", showfliers=True)
sns.boxplot(ax=ax2, x="Class", y="Amount", hue="Class", data=df, palette="PRGn", showfliers=False)

ax1.set_title('Boxplot with Outliers')
ax1.set_xlabel('Class')
ax1.set_ylabel('Amount')

ax2.set_title('Boxplot without Outliers')
ax2.set_xlabel('Class')
ax2.set_ylabel('Amount')
plt.tight_layout()
plt.show()

# Se correlacionan las variables y se muestran en un grafico de pearson
plt.figure(figsize=(14, 14))
plt.title('Credit Card Transactions Features Correlation Plot (Pearson)')
corr = df.corr()
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=0.1, cmap=cmap, square=True)
plt.yticks(rotation=0)
plt.show()

# Conclusiones de las correlaciones
corr_values = corr.unstack().sort_values(ascending=False)
corr_values = corr_values[corr_values != 1]
corr_print = corr_values.head(10)
print("\n< Correlaciones >\n\n", corr_print)
print("\n")

# Distribuciones de las variables
features = df.columns[:-1]
plt.figure(figsize=(16, 28))

for i, feature in enumerate(features, 1):
    plt.subplot(8, 4, i)

    # Plot KDE plots for both classes
    sns.kdeplot(df.loc[df['Class'] == 0, feature], color='blue', shade=True, label="Class = 0")
    sns.kdeplot(df.loc[df['Class'] == 1, feature], color='red', shade=True, label="Class = 1")

    # Add labels and adjust tick parameters
    plt.xlabel(feature, fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)

    # Hide y-axis labels for better clarity
    if i % 4 != 1:
        plt.ylabel('')
    else:
        plt.ylabel('Density', fontsize=12)

plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()

