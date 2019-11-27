import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

def get_score(n_estimators, cv_folds = 3):
	"""Return the average MAE (mean absolute error) over CV folds of random forest model
	
	Assumes data in form of X (features/predictors DataFrame), y (target data Series)

	Keyword argument:
	n_estimators -- the number of trees in the forest
	cv_folds --  cross-validation splitting strategy
		- None, to use the default 3-fold cross validation,
		- integer, to specify the number of folds in a (Stratified)KFold,
		- CV splitter,
		- An iterable yielding (train, test) splits as arrays of indices.()
	"""
	my_pipeline = Pipeline(steps = [
	('preprocessor', SimpleImputer()),
	('model', RandomForestRegressor(n_estimators = n_estimators,
									random_state = 0))
	])
	# Multiply by -1 since sklearn calculates a negative MAE
	scores = -1 * cross_val_score(my_pipeline, X, y,
								 cv = cv_folds,
								 scoring = 'neg_mean_absolute_error')
	return scores.mean()
	pass

# Import data
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

y = train_data['Survived'].copy()

X = train_data.drop('Survived', axis = 1).copy()

sns.distplot(train_data['Age'].dropna())
plt.show()
# param_grid = { 
#     'classifier__n_estimators': [200, 500],
#     'classifier__max_features': ['auto', 'sqrt', 'log2'],
#     'classifier__max_depth' : [4,5,6,7,8],
#     'classifier__criterion' :['gini', 'entropy']}
# from sklearn.model_selection import GridSearchCV
# CV = GridSearchCV(rf, param_grid, n_jobs= 1)
                  
# CV.fit(X_train, y_train)  
# print(CV.best_params_)   
# print(CV.best_score_)
