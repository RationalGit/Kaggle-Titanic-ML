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


## 20191130 working pipeline, accuracy 0.821
import re
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 

#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector(BaseEstimator, TransformerMixin):
    #Class Constructor 
    def __init__(self, feature_names):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform(self, X, y = None):
        return X[self._feature_names] 
    
# Custom transformer that:
# Extracts ticket_num from the Ticket column
# ~~Create a numerical column for 'shared_exact_ticket' if ticket_num is not unique
# ~~Create a numerical column for 'shared_adjacent_ticket' if ticket_num +/- 1 is not unique
# splits 'Name' column into Title, First_name, Surname, Maiden_first_name and Maiden_surname
# Creates a 'num_cabins' column by counting the number of spaces in the 'Cabin' column, Nan => 0
# ~~Create a 'Maiden_fam_aboard' column if Maiden_surname matches any instance in Surname column

## ~~ issues: not working with whole dataset, so will likely skew results! e.g. if shared ticket is in another part of the data

class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])

class StringIndexer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
        ))
    
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Return self nothing else to do here
    def __init__(self):
        pass
        
    # Return self nothing else to do here
    def fit(self, X, y = None):
        return self

    # Helper function to extract number of cabins (if NaN, 0)
    def get_num_cabins(self, obj):
        try:
            return str(obj).count(' ') + 1
        except:
            return 0
                        
    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None):
        #Depending on constructor argument add num_cabins
        #using the helper functions written above 
        X.loc[:, 'Num_cabins'] = X['Cabin'].apply(self.get_num_cabins) 
                    
        #returns numpy array
        return X.values 

# Custom transformer we wrote to engineer features (create an 'Is_alone'
# column if SibSp & Parch are both 0
# passed as boolen arguements to its constructor
class NumericalTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__(self):
        pass
        
    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        X.loc[:,'Is_alone'] = np.where((X['SibSp'] == 0) & (X['Parch'] == 0), 1, 0)
        return X.values
            
# Cardinality: number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
# categorical_features = [cname for cname in X.columns if X[cname].nunique() < 20 and X[cname].dtype == "object"]
categorical_features = ['Sex', 'Cabin']

# Select numerical columns
# numerical_features = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
numerical_features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

#Defining the steps in the categorical pipeline 
categorical_pipeline = Pipeline(steps = [
    ('cat_selector', FeatureSelector(categorical_features)),
    ('cat_transformer', CategoricalTransformer()),
    ('imputer', SimpleImputer(strategy = 'constant', 
                               fill_value = 'None')),
    ('one_hot_encoder', OneHotEncoder(sparse = False, 
                                      handle_unknown = 'ignore'))
])
    
#Defining the steps in the numerical pipeline     
numerical_pipeline = Pipeline(steps = [
    ('num_selector', FeatureSelector(numerical_features)),
    ('num_transformer', NumericalTransformer()),
    ('imputer', SimpleImputer(strategy = 'median'))
])

#Combining numerical and categorical piepline into one full big pipeline horizontally 
#using FeatureUnion
full_pipeline = FeatureUnion(transformer_list = [
    ('categorical_pipeline', categorical_pipeline),
    ('numerical_pipeline', numerical_pipeline)
])

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Import data
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

y = train_data['Survived'].copy()

X = train_data.drop('Survived', axis = 1).copy()
X_test = test_data.copy()

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)


#The full pipeline as a step in another pipeline with an estimator as the final step
full_pipeline_m = Pipeline(steps = [
    ('full_pipeline', full_pipeline),
    ('model', RandomForestClassifier())
])

#Can call fit on it just like any other pipeline
full_pipeline_m.fit(X_train, y_train)

# #Can predict with it like any other pipeline
y_pred = full_pipeline_m.predict(X_valid) 

error = mean_absolute_error(y_pred, y_valid)
1 - error