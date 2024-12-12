import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from sklearn.metrics import r2_score
from utils import model_exports  
from skopt.space import Categorical
from xgboost import XGBRegressor

class Models:
    
    def __init__(self):
        self.best_model = None
        self.X_test = None
        self.y_test = None
        self.reg = {
             'LightGBM':LGBMRegressor(force_row_wise=True),
              'XGBoost': XGBRegressor(),
            'GradientBoosting': GradientBoostingRegressor(),
           
            
        }
        
        self.params = {
            
            'LightGBM': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.02, 0.03,0.04 ,0.05, 0.1],
               'num_leaves': [20, 31, 50, 100],
                'max_depth': [-1, 5, 9, 12],
                
            },
            'XGBoost': {
                'n_estimators': [100, 200, 500],
                'learning_rate': (0.01,0.02, 0.03,0.05,0.1,0.2,0.3,0.4 ),
                'max_depth': [5,9, 10, 12],
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 500],
                 'learning_rate': (0.01,0.02, 0.03,0.05,0.1,0.2,0.3,0.4 ),
                'max_depth': [3,5, 9, 12],
                'min_samples_split': [2, 5,8],
                'min_samples_leaf': [1, 2, 4]
            },
          
        }

    def evaluate_model(self, model_name, model, X, y):
        y_pred = model.predict(X)
        score = r2_score(y, y_pred)
        print(f"R^2 for {model_name}: {score:.4f}")
        
        return np.abs(score)

    def grid_training(self, X, y):
        best_score = 0
        best_model = None
        
        for name, reg in self.reg.items():
            print(f"Training model: {name}")
            bayes_search = BayesSearchCV(reg, self.params[name], cv=5, n_iter=50, random_state=42, verbose=0)
            bayes_search.fit(X, y.values.ravel())
            
            score = self.evaluate_model(name, bayes_search, self.X_test, self.y_test)
            
            if score > best_score:
                best_score = score
                best_model = bayes_search.best_estimator_
                print(f"New best model: {name} with score {best_score:.4f}")
        
        self.best_model = best_model
        model_exports(best_model, best_score)
        
    def get_best_model(self):
        return self.best_model