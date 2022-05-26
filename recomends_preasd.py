#!/usr/bin/env python3
# coding: utf-8

# In[1]:


from verticapy import vDataFrame
#from verticapy import vertica_conn
from verticapy import vdf_from_relation

from verticapy.learn.linear_model import LogisticRegression, cross_validate

import verticapy


# In[2]:


import vertica_python

import datetime
import re
import gc
import json

from functools import partial


# In[3]:


with open('config.json', 'r') as f:
    conn_info = json.load(f)


# In[4]:


#почистим все сессии

with vertica_python.connect(**conn_info) as conn:
    sql = f"""
    SELECT close_user_sessions('{conn_info['user']}');
    """

    cur = conn.cursor()
    cur.execute(sql)
    print(cur.fetchall())


# In[5]:


try:
    verticapy.set_option("temp_schema", "IVOLGA")
except:
    pass


# In[6]:


verticapy.__version__


# In[7]:

try:
    verticapy.new_connection({
                   'host': conn_info['host'], 
                   'port': conn_info['port'], 
                   'database': conn_info['database'], 
                   'password': conn_info['password'], 
                   'user': conn_info['user'],
                  },
                   name = "MyVerticaConnection")

    verticapy.connect("MyVerticaConnection")
except:
    pass


# In[9]:


import pandas as pd
import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, average_precision_score,     recall_score

from sklearn.model_selection import train_test_split


# In[10]:


import xgboost as xgb
from xgboost import XGBClassifier


# In[11]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# In[12]:


from pandarallel import pandarallel


# In[13]:

try:
    N_JOBS
except NameError:
    N_JOBS = 10  # число паралельных процесов в питоне

try:
    N_REC
except NameError:
    N_REC = 1000  # число рекомендуемых селлеров на запуск

    
run_log_comment = {
    'logreg': {},
    'xgb': {},
    'knn': {},
    'recomends': {},
}

start_time = datetime.datetime.now()
duration = False
print(start_time)


# In[14]:


pandarallel.initialize(nb_workers=N_JOBS)


# In[15]:


cs_data = vdf_from_relation(f"(select * from ivolga.preasd_dataset) x")


# In[16]:


print('len dataset', len(cs_data))


# In[17]:


features = cs_data.get_columns()[5:]
features[0]


# In[18]:


print('len features', len(features))


# In[19]:


# apply log10 to features
log_apply = {}
for f in features:
    log_apply[f] = 'LOG(1 + {})'


# In[20]:

print('logarithm dataset')
cs_data.apply(log_apply)

# In[22]:

print('dataset to pandas')
data = cs_data.to_pandas()


# In[23]:


print('len data in pandas', len(data))


# In[24]:


X1, X2, y1, y2 = train_test_split(data, data['is_asd'], test_size=0.5, 
                                  random_state=17, shuffle=True)


# In[25]:


# Log Reg score


# In[26]:


features1 = list(data.columns)[5:]
features1[0]


# In[27]:


features1[-1]


# In[28]:


parameters = {
    'C': [0.1, 1, 2],
    'class_weight': [None, {1: 10, 0:1}, 'balanced']
}

"""
parameters = {
    'C': [0.1],
    'class_weight': [None]
}
"""


# In[29]:


lr = LogisticRegression(penalty='l1', random_state=17, solver='saga')

lr


# In[30]:


clf = GridSearchCV(lr, parameters, scoring='f1', n_jobs=N_JOBS, cv=3)


# In[31]:


clf.fit(data[features1], data['is_asd'])


# In[32]:


clf.best_params_


# In[33]:


clf.best_score_


# In[34]:


run_log_comment['logreg']['logreg_best_score_f1'] = clf.best_score_


# In[ ]:





# In[35]:


lr1 = LogisticRegression(penalty='l1', random_state=17, solver='saga', **clf.best_params_)
lr1


# In[36]:


lr2 = LogisticRegression(penalty='l1', random_state=17, solver='saga', **clf.best_params_)
lr2


# In[37]:


lr1.fit(X1[features1], y1)


# In[38]:


lr2.fit(X2[features1], y2)


# In[39]:


run_log_comment['logreg']['lr1_f1'] = f1_score(lr1.predict(X2[features1]), y2)
run_log_comment['logreg']['lr2_f1'] = f1_score(lr2.predict(X1[features1]), y1)


# In[40]:


# inference Log Reg

X1['logreg_score'] = lr2.predict_proba(X1[features1])[:,1]
X2['logreg_score'] = lr1.predict_proba(X2[features1])[:,1]


# In[41]:


df_scored = pd.concat([X1[['user_id', 'logreg_score']], X2[['user_id', 'logreg_score']]])


# In[ ]:
print(run_log_comment)




# In[42]:


# XGBoost


# In[43]:


xb = XGBClassifier(n_jobs=N_JOBS, random_state=17, min_child_weight=10, n_estimators=100, 
                   learning_rate=0.1, subsample=0.7)


# In[44]:


"""params = {
    'max_depth': [6, 10],
    'colsample_bytree': [0.5, 1],
    'reg_alpha': [0.1, 0.5, 1],
}"""

params = {
    'max_depth': [10],
    'colsample_bytree': [0.5],
    'reg_alpha': [0.5],
}


# In[45]:


clf = GridSearchCV(xb, params, scoring='f1', refit=False, cv=3)


# In[46]:


clf.fit(data[features1], data['is_asd'])


# In[47]:


clf.best_params_


# In[48]:


clf.best_score_


# In[49]:


run_log_comment['xgb']['xgb_best_score_f1'] = clf.best_score_


# In[50]:


print(run_log_comment)


# In[51]:


xb1 = XGBClassifier(n_jobs=N_JOBS, random_state=17, min_child_weight=10, n_estimators=100, 
                   learning_rate=0.1, subsample=0.7, **clf.best_params_)
xb1


# In[52]:


xb2 = XGBClassifier(n_jobs=N_JOBS, random_state=17, min_child_weight=10, n_estimators=100, 
                   learning_rate=0.1, subsample=0.7, **clf.best_params_)
xb2


# In[53]:


xb1.fit(X1[features1], y1)


# In[54]:


xb2.fit(X2[features1], y2)


# In[55]:


run_log_comment['xgb']['xb1_f1'] = f1_score(xb1.predict(X2[features1]), y2)
run_log_comment['xgb']['xb2_f1'] = f1_score(xb2.predict(X1[features1]), y1)


# In[56]:


print(run_log_comment)


# In[57]:


# inference XGboost

X1['xgb_score'] = xb2.predict_proba(X1[features1])[:,1]
X2['xgb_score'] = xb1.predict_proba(X2[features1])[:,1]


# In[58]:


df_scored1 = pd.concat([X1[['user_id', 'xgb_score']], X2[['user_id', 'xgb_score']]])


# In[59]:


df_scored = df_scored.merge(df_scored1, on='user_id', how='left')


# In[61]:


del(data)


# In[62]:


del(X1)
del(X2)
del(cs_data)


# In[63]:


gc.collect()


# In[64]:


# KNN


# In[65]:


pandarallel.initialize(progress_bar=True, nb_workers=N_JOBS)


# In[66]:


cs_data = vdf_from_relation(f"(select * from ivolga.preasd_dataset) x")


# In[67]:


cs_data


# In[68]:


data = cs_data.to_pandas()


# In[69]:


features1 = list(data.columns)[5:]
features2 = [f for f in features1 if not f.endswith('_csf')]
len(features2)


# In[70]:


sc = StandardScaler()


# In[71]:


df_scaled = pd.DataFrame(sc.fit_transform(data[features2]), columns=features2)


# In[72]:


df_scaled['user_id'] = data['user_id']
df_scaled['is_asd'] = data['is_asd']


# In[73]:


df_scaled1 = df_scaled[(df_scaled['revenue_180']>-0.05) | (df_scaled['is_asd']==1)]


# In[74]:


df_asd = df_scaled1[df_scaled1['is_asd']==1]
df_asd_np = df_asd.to_numpy()
df_cand = df_scaled1[df_scaled1['is_asd']==0]


# In[75]:


def dist(p1, p2):
    return np.abs(p1.to_numpy()-p2.to_numpy()).sum()


# In[76]:


def cand_dist(cand):
    return min(df_asd[features2].apply(partial(dist, p2=cand), axis=1).to_numpy())


# In[77]:


df_cand['knn_dist'] = df_cand[features2].parallel_apply(cand_dist, axis=1)


# In[78]:


df_cand['knn_score'] = df_cand['knn_dist'].apply(lambda x: 1-x)


# In[79]:


data = data.merge(df_cand[['user_id', 'knn_score']], how='left', on='user_id')


# In[80]:


data = data.merge(df_scored, how='left', on='user_id')


# In[81]:


data['knn_score'] = data['knn_score'].fillna(-10)


# In[82]:


data


# In[83]:


# recomends


# In[84]:


"""
IVOLGA.preasd_recomendations

user_id
external_id
is_asd
was_asd
asd_others
all_revenue_90
all_revenue_30
activations_30
activations_30_60
activations_60_90
other_vertical_items_end_date
logreg_score
xgb_score
knn_score
date_rec


create table IVOLGA.preasd_recomendations (
user_id INTEGER,
external_id INTEGER,
is_asd INTEGER,
was_asd INTEGER,
asd_others INTEGER,
all_revenue_90 DECIMAL,
all_revenue_30 DECIMAL,
activations_30 DECIMAL,
activations_30_60 DECIMAL,
activations_60_90 DECIMAL,
other_vertical_items_end_date INTEGER,
logreg_score FLOAT,
xgb_score FLOAT,
knn_score FLOAT,
date_rec DATE  DEFAULT sysdate
)
ORDER BY date_rec


"""
with vertica_python.connect(**conn_info) as conn:

    df_was_recomended = pd.read_sql(f"""select * from IVOLGA.preasd_recomendations where date_rec > sysdate - 180
""", conn)


# In[85]:


print('was recomended', len(df_was_recomended))


# In[86]:


user_id_was_recomended = set(df_was_recomended['user_id'])


# In[87]:


data1 = data[data['user_id'].apply(lambda x: x not in user_id_was_recomended)]


# In[88]:


data1 = data1[data1['asd_others'] == 0]


# In[89]:


data1 = data1[data1['is_asd'] == 0]


# In[90]:


data1


# In[ ]:





# In[91]:


thresholds = {
    'knn': 0,
    'logreg': 0.5,
    'xgb': 0.5,
}


# In[92]:


data2 = data1.sort_values(by='knn_score', ascending=False)
data2 = data2[data2['knn_score'] > thresholds['knn']].iloc[:N_REC]

top_knn = set(data2['user_id'])


# In[93]:


data2 = data1.sort_values(by='logreg_score', ascending=False)
data2 = data2[data2['logreg_score'] > thresholds['logreg']].iloc[:N_REC]

top_logreg = set(data2['user_id'])


# In[94]:


data2 = data1.sort_values(by='xgb_score', ascending=False)
data2 = data2[data2['xgb_score'] > thresholds['xgb']].iloc[:N_REC]

top_xgb = set(data2['user_id'])


# In[95]:


top_users = ((top_knn.intersection(top_logreg)).union(top_knn.intersection(top_xgb))).    union(top_xgb.intersection(top_logreg))


# In[96]:


print('top_users', len(top_users))


# In[97]:


num_users = int((N_REC - len(top_users)) / 3)
num_users


# In[98]:


data2 = data1[data1['user_id'].apply(lambda x: x not in top_users)]


# In[99]:


data3 = data2.sort_values(by='knn_score', ascending=False)
data3 = data3[data3['knn_score'] > thresholds['knn']].iloc[:num_users]

knn_users = set(data3['user_id'])
print('knn_users', len(knn_users))


# In[100]:


data3 = data2.sort_values(by='logreg_score', ascending=False)
data3 = data3[data3['logreg_score'] > thresholds['logreg']].iloc[:num_users]

logreg_users = set(data3['user_id'])
print('logreg_users', len(logreg_users))


# In[101]:


data3 = data2.sort_values(by='xgb_score', ascending=False)
data3 = data3[data3['xgb_score'] > thresholds['xgb']].iloc[:num_users]

xgb_users = set(data3['user_id'])
print('knn users', len(xgb_users))


# In[102]:


rec_users = top_users.union(knn_users.union(logreg_users.union(xgb_users)))
print('xgb_users', len(rec_users))


# In[103]:


data3 = data1[data1['user_id'].apply(lambda x: x in rec_users)]


# In[104]:


data4 = data3[['user_id',
    'external_id',
    'is_asd',
    'was_asd',
    'asd_others',
    'all_revenue_90',
    'all_revenue_30',
    'activations_30',
    'activations_30_60',
    'activations_60_90',
    'other_vertical_items_end_date',
    'knn_score',
    'logreg_score',
    'xgb_score',]
    ]


# In[105]:


insert_data = []
for _, i in data4.iterrows():
    insert_data.append(tuple(i))
    
insert_data = tuple(insert_data)


# In[106]:


data4


# In[107]:


run_log_comment['recomends']['thresholds'] = thresholds
run_log_comment['recomends']['top_users'] = len(top_users)
run_log_comment['recomends']['logreg_users'] = len(logreg_users)
run_log_comment['recomends']['xgb_users'] = len(xgb_users)
run_log_comment['recomends']['knn_users'] = len(knn_users)
run_log_comment['recomends']['recomended'] = len(data4)


# In[ ]:





# In[108]:


with vertica_python.connect(**conn_info) as conn:
    cur = conn.cursor()
    cur.execute('drop table if exists IVOLGA.preasd_rec_last')


# In[109]:


verticapy.pandas_to_vertica(data4, name='preasd_rec_last', schema='IVOLGA')


# In[ ]:





# In[110]:


with vertica_python.connect(**conn_info) as conn:
    cur = conn.cursor()
    cur.execute("""
    insert into IVOLGA.preasd_recomendations (user_id,
    external_id,
    is_asd,
    was_asd,
    asd_others,
    all_revenue_90,
    all_revenue_30,
    activations_30,
    activations_30_60,
    activations_60_90,
    other_vertical_items_end_date,
    knn_score,
    logreg_score,
    xgb_score
    ) select * from IVOLGA.preasd_rec_last
    """)

    cur.execute('commit;')


# In[ ]:





# In[111]:


duration = datetime.datetime.now() - start_time
run_log_comment['duration'] = str(duration)


# In[112]:


run_log_comment_json = json.dumps(run_log_comment)
print(run_log_comment_json)


# In[113]:


# run log

"""
create table if not exists IVOLGA.runlog (
    date_time TIMESTAMP DEFAULT sysdate,
    task VARCHAR,
    operation VARCHAR,
    comment VARCHAR(1000)
)
"""
with vertica_python.connect(**conn_info) as conn:
    cur = conn.cursor()

    sql = f"""
    insert into IVOLGA.runlog (task, operation, comment) values ('preasd', 'get_recommendations', 
    '{run_log_comment_json}')
    """

    cur.execute(sql)
    cur.execute('commit;')


# In[116]:


1+1


# In[115]:


print(duration)


# In[ ]:




