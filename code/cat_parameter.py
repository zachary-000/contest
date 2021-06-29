import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("train.csv")
strs = df['target'].value_counts()
# print(strs)

value_map = dict((v, i) for i, v in enumerate(strs.index))
value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
# print(value_map)

df = df.replace({'target': value_map})
df = df.drop(columns = ['id'])
print(df)

train_X = df.iloc[:10000, :-1]
train_y = df['target'].iloc[:10000]
score_lt = []

model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='MultiClass',
    n_estimators = 200,
    # depth = 10,
    learning_rate = 0.06,
    reg_lambda = 1,
    random_state = 16,
    verbose=True,
)

param_grid = {'depth':[8,9,10,11,12]}
GS = GridSearchCV(model, param_grid, cv = 10, scoring = 'neg_log_loss')
GS.fit(train_X, train_y)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)

