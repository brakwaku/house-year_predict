# %%
# the full imports
import pandas as pd 
import numpy as np
import seaborn as sns
import altair as alt

# %%
# the from imports
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

# %%
dwellings_denver = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv")
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_neighborhoods_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")   

alt.data_transformers.enable('json')

# %%
h_subset = dwellings_ml.filter(['livearea', 'finbsmnt', 
    'basement', 'yearbuilt', 'nocars', 'numbdrm', 'numbaths', 
    'stories', 'yrbuilt', 'before1980']).sample(500)

sns.pairplot(h_subset, hue = 'before1980')

corr = h_subset.drop(columns = 'before1980').corr()
# %%
sns.heatmap(corr)


# %%
(alt.Chart(dwellings_ml).encode(
    x = alt.X('yrbuilt', scale = alt.Scale(zero = False)),
    y = alt.Y('numbaths', scale = alt.Scale(zero = False)),
    color = 'before1980:O'
).mark_circle())
# %%
dat_count = (dwellings_ml
.groupby(['yrbuilt', 'numbaths'])
.agg(count = ('nocars', 'size'))
.reset_index())

# %%
# Not working well
(alt.Chart(dat_count)
.encode(x = 'yrbuilt', y = 'numbaths', color = 'count:O')
.mark_rect()
)

# %%
X_pred = dwellings_ml.drop(['yrbuilt', 'before1980'], axis = 1)
y_pred = dwellings_ml.before1980

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_pred,
    y_pred,
    test_size = .34,
    random_state = 76
)

# now we use X_train and y_train to build a model.  
# %%
# https://scikit-learn.org/stable/modules/tree.html#classification
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
# %%


#%%


#%%


#%%


# %%


#%%


# %%
