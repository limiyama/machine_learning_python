# ctrl c + ctrl v dos trechos usando random forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

randomforest = RandomForestClassifier().fit(X_train, y_train)
y_probabilities = randomforest.predict_proba(X_test)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_probabilities[:,1])
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(6,6))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='Random forests ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.show()


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rfc = RandomForestClassifier(random_state=0)

grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

def engagement_model():
    # create a function that trains a model to predict significant learner engagement with a video using asset/train.csv. 
    # using this model, return a Pandas Series object of length 2309 with the data being the probability that each corresponding video from readonly/test.csv 
    # will be engaging (according to a model learned from the 'engagement' label in the training set), and the video index being in the id field.
    

    df_train = pd.read_csv('./assets/train.csv')
    df_test = pd.read_csv('./assets/test.csv')

    most_importance = ['document_entropy', 'freshness']

    X_train, y_train = df_train[most_importance], df_train.iloc[:,-1].astype(int)
    X_test = df_test[most_importance]

    model = RandomForestClassifier(max_depth=10, max_features='sqrt', min_samples_leaf=4, 
                                 min_samples_split=10, n_estimators=200, n_jobs=-1,
                                 random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)

    test_features = test_df[features]
    probabilities = model.predict_proba(test_features)[:, 1]  
    return pd.Series(probabilities, index=test_df['id'])
engagement_model()
