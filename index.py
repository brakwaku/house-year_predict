# %%
# import sys
# !{sys.executable} -m pip install seaborn scikit-learn

# %%
# the full imports
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt

# %%
# the from imports
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# %%
dwellings_denver = pd.read_csv(
    "https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv")
dwellings_ml = pd.read_csv(
    "https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_neighborhoods_ml = pd.read_csv(
    "https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")

alt.data_transformers.enable('json')

# %%
h_subset = dwellings_ml.filter(['livearea', 'finbsmnt',
                                'basement', 'yearbuilt', 'nocars', 'numbdrm', 'numbaths',
                                'stories', 'yrbuilt', 'before1980']).sample(500)

sns.pairplot(h_subset, hue='before1980')

corr = h_subset.drop(columns='before1980').corr()
# %%
sns.heatmap(corr)


# %%
# Question 1
chart_1 = alt.Chart(dwellings_ml).encode(
    x = alt.X("before1980:O", title="Before/After 1980"),
    y = alt.Y("livearea", scale = alt.Scale(zero = False), title="Total Square Footage")
).mark_boxplot().properties(title="Total Square Footage by Year Built", width = 800)

# chart_1
chart_1.save('chart_1.png')

# %%
chart_2 = alt.Chart(dwellings_ml).encode(
    x = alt.X("before1980:N", title="Before / After 1980 (Year built"),
    y = alt.Y("sum(arcstyle_ONE-STORY):Q", title="Arc Style")
).mark_bar().properties(title="Number of baths by Year Built")

# chart_2
chart_2.save('chart_2.png')

# %%
# chart_3 = (alt.Chart(dwellings_ml).encode(
#     alt.X("before1980:O", scale = alt.Scale(zero = False), title="Year Built"),
#     alt.Y("sum(stories)", scale = alt.Scale(zero = False), title="Number of cars"),
#     color = 'before1980:O'
# ).mark_bar()
# .properties(title="Number of cars by Year Built", width = 500))

# chart_3
# chart_3.save('chart_3.png')

# %%
# Question 2
X_pred = dwellings_ml.drop(['yrbuilt', 'before1980'], axis=1)
y_pred = dwellings_ml.before1980

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_pred,
    y_pred,
    test_size=.34,
    random_state=76
)

# now we use X_train and y_train to build a model.
# %%
# https://scikit-learn.org/stable/modules/tree.html#classification
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# %%
model = clf.predict(X_test)

clf.score(X_test, y_test)

# %%
metrics.precision_score(y_test, model)


# %%
# Question 3
df_features = (pd.DataFrame(
    {'Feature Names': X_train.columns,
     'Importances': clf.feature_importances_})
    .sort_values('Importances', ascending=False))

print(df_features.to_markdown(index=False))
# %%
feature_chart = alt.Chart(df_features).encode(
    alt.X("Feature Names:O", sort='-y', title="Feature names"),
    alt.Y("Importances", title="Importance")
).mark_bar().properties(title="Feature Importance Ranking")

feature_chart.save('feature_chart.png')

# %%
# (pd.Series(clf.feature_importances_, index=X_pred.columns)
#    .nlargest(4)
#    .plot(kind='barh'))


# %%
# Question 4
the_table = print(metrics.confusion_matrix(y_test, model))
print(classification_report(y_test, model))
the_plot = plot_confusion_matrix(clf, X_test, y_test)

# %%
(alt.Chart(dwellings_ml).encode(
    x=alt.X('yrbuilt', scale=alt.Scale(zero=False)),
    y=alt.Y('numbaths', scale=alt.Scale(zero=False)),
    color='before1980:O'
).mark_circle())
# %%
dat_count = (dwellings_ml
             .groupby(['yrbuilt', 'numbaths'])
             .agg(count=('nocars', 'size'))
             .reset_index())

# %%
# Not working well
(alt.Chart(dat_count)
 .encode(x='yrbuilt', y='numbaths', color='count:O')
 .mark_rect()
 )
