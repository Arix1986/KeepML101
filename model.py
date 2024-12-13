import numpy as np
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
class Models:
    
    def __init__(self):
        self.best_model = None
        self.X_test = None
        self.y_test = None
        self.reg = {
               'ExtraTrees':ExtraTreesRegressor(random_state=42),
               'LightGBM':LGBMRegressor(force_row_wise=True,random_state=42),
              'GradientBoosting': GradientBoostingRegressor(random_state=42),
              'XGBoost': XGBRegressor(random_state=42),
              'BagginRegressor_KN':BaggingRegressor(estimator=KNeighborsRegressor(),random_state=42),
              'BagginRegressor_DT':BaggingRegressor(estimator=DecisionTreeRegressor(),random_state=42),
            
            
           
         }
        
        self.params = {
            
            'LightGBM': {
                'boosting_type': ['dart'],
                'learning_rate': [0.01, 0.02, 0.03,0.032,0.04 ,0.05, 0.1],
                'n_estimators': [500],
                'num_leaves': [21, 31,41, 53],
                'max_depth': [ 7,9,10,11,12],
                'min_child_samples':[20],
                'min_child_weight':[0.001],
                'reg_alpha': [0.0], 
                'reg_lambda': [0.0],
                             
            },
            'XGBoost': {
                'n_estimators': [100, 200, 500],
                'learning_rate': (0.01,0.02, 0.03,0.05,0.1,0.2,0.3,0.4 ),
                'max_depth': [5,9, 10, 12],
            
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300,500],
                'learning_rate': (0.01,0.02, 0.03,0.05,0.1,0.2,0.3,0.4 ),
                'max_depth': [3, 5,9, 12],
                
            },
            'BagginRegressor_KN': {
                'n_estimators': [50, 100, 200],          
                'max_samples': [ 1.0],                
                'bootstrap': [True, False],                   
                'bootstrap_features': [True, False],           
             },
            'BagginRegressor_DT': {
                'n_estimators': [50, 100, 200],          
                'max_samples': [ 1.0],                
                'estimator__max_depth':[5,9,10] ,             
                'bootstrap': [True, False],                   
                'bootstrap_features': [True, False],           
             },
             
            'ExtraTrees': {
                 'n_estimators': [100, 200, 300,500],
                'ccp_alpha': (0.01,0.02, 0.03,0.05,0.1,0.2,0.3,0.4 ),
                'max_depth': [3, 5,9, 12],
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
            bayes_search = GridSearchCV(reg, self.params[name], cv=5,  verbose=2)
            bayes_search.fit(X, y.values.ravel())
            
            score = self.evaluate_model(name, bayes_search, self.X_test, self.y_test)
            
            if score > best_score:
                best_score = score
                best_model = bayes_search.best_estimator_
                print(f"New best model: {name} with score {best_score:.4f}")
        
        self.best_model = best_model
        self.model_exports(best_model, best_score)
        self.model_grilla(bayes_search)
        
        
    def get_best_model(self):
        return self.best_model
    
    def model_exports(self, clf,score):
        score=str(score)
        joblib.dump(clf,f'./models/best_model_{score[:5]}.pkl')
    
    def model_grilla(self, clf):
        joblib.dump(clf,f'./models/best_model_grilla.pkl')   