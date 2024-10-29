import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv(r'E:\onedrive\OneDrive - cumt.edu.cn\Python\stacking\feature_extract\feature_fuben.csv')


X = data.iloc[:, :24].values
y = data.iloc[:, 24].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


rf = RandomForestClassifier(random_state=42)


param_grid = {
    'n_estimators': [20, 50, 100],
    'max_depth': [2, 3, 5, 7],
    'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.2, 0.3],
}


random_search = RandomizedSearchCV(estimator=rf,  param_distributions=param_grid,n_iter=100,
                                   cv=3, scoring='accuracy', verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)


print("Best parameters found: ", random_search.best_params_)
print("Best accuracy found: {:.2f}".format(random_search.best_score_))


best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy with best model: {:.2f}".format(test_accuracy))

results = random_search.cv_results_
mean_scores = results['mean_test_score']
params = results['params']


for key in param_grid.keys():
    param_values = [p[key] for p in params]
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=param_values, y=mean_scores, inner=None, linewidth=1)
    sns.stripplot(x=param_values, y=mean_scores, color='white', edgecolor='black', size=4)
    

    for i, val in enumerate(set(param_values)):
        subset_scores = [mean_scores[idx] for idx, param_val in enumerate(param_values) if param_val == val]
        median = np.median(subset_scores)
        quartiles = np.percentile(subset_scores, [25, 75])
        plt.plot([i-0.1, i+0.1], [median, median], color='red', linestyle='-', linewidth=2)
        plt.plot([i-0.1, i+0.1], [quartiles[0], quartiles[0]], color='blue', linestyle='--', linewidth=1)
        plt.plot([i-0.1, i+0.1], [quartiles[1], quartiles[1]], color='blue', linestyle='--', linewidth=1)

    plt.title(f'Parameter: {key}')
    plt.xlabel(key)
    plt.ylabel('Mean Accuracy')

    plt.tight_layout()
    plt.savefig(f'violin_plot_{key}.png', dpi=600)
    plt.show()
    
