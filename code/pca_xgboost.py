import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.decomposition import PCA


df = pd.read_csv("train.csv")
strs = df['target'].value_counts()
# print(strs)

value_map = dict((v, i) for i, v in enumerate(strs.index))
value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
# print(value_map)

df = df.replace({'target': value_map})
df = df.drop(columns = ['id'])

# balance
df_1 = df.drop(df[(df['target'] == 5) | (df['target'] == 7)].index) 
df_2 = df[(df['target'] == 5) | (df['target'] == 7)]
df_2 = df_2.sample(n = 50000)
df = pd.concat([df_1, df_2])
df = df.sort_index()
df = df.reset_index()
print(df['target'].value_counts())
print(df)

train_X = df.iloc[:, :-1]
train_y = df['target']

# lda = LDA(n_components = 8)
# new_x = lda.fit_transform(x_train, y_train)
# print(new_x)

# pca
pca = PCA(n_components = 50)
low_dim_data = pca.fit_transform(train_X)
train_X = pd.DataFrame(low_dim_data)
print(train_X.shape)


X_train, X_test, y_train, y_test= train_test_split(train_X.values, train_y.values, test_size = 0.25)

model = XGBClassifier(
    max_depth = 4,
    learning_rate = 0.06,
    min_child_weight = 0.9,
    subsample = 0.6,
    colsample_bytree = 0.7,
    scale_pos_weight = 0.9,
    gamma = 0.1,
    reg_alpha = 0.01,
    reg_lambda = 1.3,
    n_estimators = 350,
    max_delta_step = 0.4,
    objective = 'mlogloss', 
    random_state = 16,
    use_label_encoder = False
    )


model.fit(X_train, y_train , 
          eval_set = [(X_train, y_train)], 
          eval_metric = "mlogloss", 
          early_stopping_rounds = 10, 
          verbose = True)

y_pred = model.predict_proba(X_test)
print ("Use log_loss() in scikit-learn, the result is {} ".format(log_loss(y_test, y_pred)))

df = pd.read_csv("test.csv")
x_test = df.iloc[:, 1:]

low_dim_data = pca.fit_transform(x_test)
x_test = pd.DataFrame(low_dim_data)
print(x_test.shape)

proba = model.predict_proba(x_test.values)
print(proba)
output = pd.DataFrame({'id': df.iloc[0:,0], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6':proba[:,5], 'Class_7':proba[:,6], 'Class_8':proba[:,7], 'Class_9':proba[:,8]})
output.to_csv('my_submission.csv', index=False)
