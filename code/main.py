import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier

# input and prepocess data
df = pd.read_csv("train.csv")
strs = df['target'].value_counts()
value_map = dict((v, i) for i, v in enumerate(strs.index))
value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
df = df.replace({'target': value_map})
df = df.drop(columns = ['id'])
print(df)

# get train and test set
train_X = df.iloc[:, :-1]
train_y = df['target']
X_train, X_test, y_train, y_test= train_test_split(train_X.values, train_y.values, test_size = 0.25)

# xgboost
xg_model = XGBClassifier(
    max_depth = 13,
    learning_rate = 0.038,
    min_child_weight = 100,
    subsample = 0.70,
    colsample_bytree = 0.68,
    colsample_bynode = 0.27,
    colsample_bylevel = 0.68,
    gamma = 0.4,
    reg_alpha = 3,
    reg_lambda = 1,
    n_estimators = 350,
    objective = 'multi:softprob',
    tree_method = 'gpu_hist', 
    num_class = 9,
    random_state = 16,
    use_label_encoder = False
    )

# y_pred = xg_model.predict_proba(X_test)
# print ("xgboost result is {} ".format(log_loss(y_test, y_pred)))

# catboost
train_pool = Pool(data=X_train, label=y_train)
test_pool = Pool(data=X_test, label=y_test) 

cat_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='MultiClass',
    n_estimators = 200,
    depth = 10,
    learning_rate = 0.06,
    reg_lambda = 1,
    random_state = 16,
    verbose=True
)

# cat_model.fit(train_pool,plot=True,eval_set=test_pool)
# y_pred = cat_model.predict_proba(X_test)
# print ("catboost result is {} ".format(log_loss(y_test, y_pred)))

# randomforest
rf_model = RandomForestClassifier(n_estimators = 185,
                               random_state = 16,
                               max_depth = 13,
                               max_features = 10,
                               min_samples_split = 10,
                               min_samples_leaf = 8,
                               # max_leaf_nodes = 35,
                               n_jobs = -1,
                               verbose = True
                               )

# rf_model.fit(X_train, y_train)
# y_pred = rf_model.predict_proba(X_test)
# print ("rf result is {} ".format(log_loss(y_test, y_pred)))

# lgbm
lgbm_model = LGBMClassifier(
    n_estimators = 290,
    learning_rate = 0.03,
    max_depth = 7,
    subsample = 0.86, 
    colsample_bytree = 0.24,
    min_child_samples = 45, 
    reg_alpha = 16, 
    reg_lambda = 4, 
    verbose = 1
    )

# model.fit(X_train, y_train)
# y_pred = model.predict_proba(X_test)

# voting
clf_vc = VotingClassifier(estimators=[('xgb', xg_model), ('cat', cat_model), ('rf', rf_model), ('lgbm', lgbm_model)], voting='soft', weights=[8, 3, 1, 3])
clf_vc.fit(X_train, y_train)
y_pred = clf_vc.predict_proba(X_test)
print ("final result is {} ".format(log_loss(y_test, y_pred)))

# predict
df = pd.read_csv("test.csv")
x_test = df.iloc[:, 1:]
proba = clf_vc.predict_proba(x_test.values)
print(proba)
output = pd.DataFrame({'id': df.iloc[0:,0], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6':proba[:,5], 'Class_7':proba[:,6], 'Class_8':proba[:,7], 'Class_9':proba[:,8]})
output.to_csv('my_submission_final.csv', index=False)
