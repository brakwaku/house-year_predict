## Project 4

__Kwaku Appau-Nkansah__

#### Elevator pitch
The clean air act of 1970 was the beginning of the end for the use of asbestos in home building. By 1976, the U.S. Environmental Protection Agency (EPA) was given authority to restrict the use of asbestos in paint. Homes built during and before this period are known to have materials with asbestos https://www.asbestos.com/mesothelioma-lawyer/legislation/ban/.

The state of Colorado has a large portion of their residential dwelling data that is missing the year built and they would like a predictive model that can classify if a house is built pre 1980. They would also like a model that predicts (regression) the actual age of each home.

Home sales data for the city of Denver from 2013 has been provided to train the model. The latest machine learning methods will be used to solve these grand questions:

1. Create 2-3 charts that evaluate potential relationships between the home variables and before1980.

2. Can you build a classification model (before or after 1980) that has at least 90% accuracy for the state of Colorado to use (explain your model choice and which models you tried)?

3. Will you justify your classification model by detailing the most important features in your model (a chart and a description are a must)?

4. Can you describe the quality of your classification model using 2-3 evaluation metrics? You need to provide an interpretation of each evaluation metric when you provide the value.

----

## Report

#### Potential relationships between the home variables and before1980

![](chart_1.png)

From the chart above, we can notice that on an average, houses built before 1980 represented by '1' on the graph had lesser total square footage than those built after 1980 (represented by '0')

![](chart_2.png)

From the graph, we see that if a house is a one story arc style, it was likely to be built before 1980

<!-- ![](chart_3.png) -->

<!-- The graph reveals that if there are about 4 car spaces in a house, it was likely built before 1980 -->

---

#### Classification model accuracy

The classification model that was used was the DecisionTreeClassifier() which produced a 0.9006546014632268%. This was above the 90% requirement

---

#### Most important features

![](feature_chart.png)

From the bar chart above, we can see that the four most important features are 'arcstyle_ONE-STORY', 'gartype_Art', 'quality_C', and 'livearea'. In effect, the model is likely to predict the year a house was built based on these features.

---

#### Quality of model

![](confusion.png)

| Precision  |   Recall |   Accuracy    |
|:-----------|---------:|--------------:|
| 0.85       |     0.88 |      0.90     |

The metrics being used for the justification are Recall, Precision and Accuracy.

__Recall__ is the ability of your model to find all the relevant cases in the model. It answers the question "What proportion of actual possitives was identified correctly?" From our model, we get 88% recall. This is calculated as __Number of true positives / (Number of true positives + Number of False Negatives)__ 2544 / (2544 + 340) = 0.88210818.

__Precision__ is the ability of a model to identify only the relevant data points. This would answer the question "What proportion of positive identifications was actually correct?". From our model, we get 85%. This is calculated as __Number of true positives / (Number of true positives + Number of false positives)__ 2544 / (2544 + 434) = 0.85426461.

__Accuracy__ is the fraction of the prediction the model gets right. It is the simple calculation where you divide the number of data points evaluated correctly by the number of total data points.
(2544 + 4473) / (2544 + 340 + 4473 + 434) = 0.9006546.

---

## APPENDIX A (PYTHON SCRIPT)

```python
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

```