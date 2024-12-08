import json
import pickle
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

def clean_df(df):
    columns_to_drop = [
        'ID', 'Listing Url', 'Scrape ID', 'Host ID', 'Host URL',
        'Host Thumbnail Url', 'Host Picture Url', 'Thumbnail Url',
        'Medium Url', 'Picture Url', 'XL Picture Url','Host Name',
        'Host Since','Host Response Time','First Review','Last Review',
        # Datos de Scrape
        'Last Scraped', 'Calendar last Scraped',
        
        # Campos descriptivos o textuales
        'Name', 'Summary', 'Space', 'Description', 'Neighborhood Overview',
        'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Host About',
        'Neighbourhood','Market','Host Response Rate','Host Acceptance Rate','Calendar Updated',
        'Features',
        # Detalles sobre el anfitrión
        'Host Location', 'Host Neighbourhood', 'Host Verifications',
        
        # Información geográfica redundante
        'Street', 'Smart Location', 'Country Code', 'Country','Geolocation',
        
        # Licencia y jurisdicción
        'License', 'Jurisdiction Names',
        
        #Elimino Amenities porque tengo una columna Accommodates que se relaciona con esta columna en cuanto a la cantidad, y realizando un Encoding de esta columna hace que se dimensione mucho
        'Amenities',
        
        #Información redundante
        'Has Availability',  'Experiences Offered', 'Neighbourhood Group Cleansed',
        
        # Este campo representa lo mismo que Host Listings Count 
        'Host Total Listings Count', 
        
        
    ]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    return df_cleaned

def remove_features_NaN(df):
    nan_percentage=df.isnull().mean() * 100
    df=df.drop(columns=nan_percentage[nan_percentage >50].index)   
    return df

def features_impact_FOREST(df):
    X= df.drop(columns=["Price"])     
    y= df["Price"]
    model=RandomForestRegressor()
    model.fit(X,y)
    feature_imp=pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_imp.plot(kind='bar', color='skyblue')
    plt.title("Feature Importances - Random Forest")
    plt.ylabel("Score")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.show()

def features_impact_SKBEST(df):
    X= df.drop(columns=["Price"])     
    y= df["Price"]
    for k in [5, 10, 'all']:
        selector = SelectKBest(score_func=f_regression, k=k)
        X_new = selector.fit_transform(X, y)
        model = LinearRegression()
        scores = cross_val_score(model, X_new, y, cv=5, scoring='r2')
        print(f"k={k}: R2 mean={scores.mean()}")
        if k == 'all':
            feature_scores = pd.DataFrame({
                "Feature": X.columns,
                "Score": selector.scores_
            }).sort_values(by="Score", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(feature_scores["Feature"], feature_scores["Score"], color='orange')
    plt.title("Feature Scores - SelectKBest")
    plt.ylabel("Score")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.show()

def features_impact_RFE(df):
    X= df.drop(columns=["Price"])     
    y= df["Price"]
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=10)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    rfe_scores = pd.Series(rfe.ranking_, index=X.columns)
    rfe_scores = rfe_scores.loc[selected_features].sort_values()

    plt.figure(figsize=(10, 6))
    rfe_scores.plot(kind='bar', color='green')
    plt.title("Feature Rankings - RFE")
    plt.ylabel("Ranking (lower is better)")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.show()

def features_impact_COR(df):
    correlation_matrix = df.corr()
    price_correlation = correlation_matrix['Price'].sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    price_correlation.plot(kind='bar', color='skyblue')
    plt.title("Price Correlation")
    plt.ylabel("Correlation")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.show()

def imputer_data_numeric(df):
    imputer = KNNImputer(n_neighbors=5)
    df_numeric = df.select_dtypes(include=['float', 'int'])
    df[df_numeric.columns] = imputer.fit_transform(df_numeric)
    return df 
def imputer_data_numeric_2(df):
    imputer = KNNImputer(n_neighbors=5)
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    df[df_numeric.columns] = imputer.fit_transform(df_numeric)
    return df , imputer ,df_numeric.columns
   

def features_PCA_full_analysis(df):
    scaler = StandardScaler()
    X = df.drop(columns=["Price"])
    y = df["Price"]
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.90)
    principal_components = pca.fit_transform(X_scaled)
    plot_explained_variance(pca)
    pc_df = pd.DataFrame(
        principal_components,
        columns=[f'PC_{i+1}' for i in range(pca.n_components_)]
    )
    pc_df['Price'] = y.values
    correlations = pc_df.corr()['Price'].drop('Price')
    print("Correlaciones con el precio:")
    print(correlations.sort_values(ascending=False))
    correlations.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
    plt.title('Correlación entre los componentes principales y el precio')
    plt.xlabel('Componentes principales')
    plt.ylabel('Correlación')
    plt.grid()
    plt.show()
    analyze_top_components(pca, X, top_component=correlations.idxmax())
    corrComponetsPrice(pc_df, y, componentes=[2, 21, 7])

def plot_explained_variance(pca):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.title('Varianza explicada acumulada por los componentes principales')
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Porcentaje de varianza explicada')
    plt.grid()
    plt.show()    
    
def analyze_top_components(pca, X, top_component):
    components_df = pd.DataFrame(
        pca.components_,
        columns=X.columns,
        index=[f'PC_{i+1}' for i in range(pca.n_components_)]
    )
    top_features = components_df.loc[top_component].sort_values(ascending=False)
    top_features.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Contribuciones de las características en {top_component}')
    plt.xlabel('Características')
    plt.ylabel('Peso')
    plt.grid()
    plt.show()  
    pc9=components_df.loc['PC_21'].sort_values(ascending=False)
    pc9.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Contribuciones de las características en PC_21')
    plt.xlabel('Características')
    plt.ylabel('Peso')
    plt.grid()
    plt.show() 
    pc2=components_df.loc['PC_7'].sort_values(ascending=False)
    pc2.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Contribuciones de las características en PC_7')
    plt.xlabel('Características')
    plt.ylabel('Peso')
    plt.grid()
    plt.show() 
    
def corrComponetsPrice(pc_df, y, componentes):
    for pc in componentes:
        sns.scatterplot(x=pc_df[f'PC_{pc}'], y=y)
        plt.title(f'Relación entre PC_{pc} y Price')
        plt.xlabel(f'PC_{pc}')
        plt.ylabel('Price')
        plt.grid()
        plt.show()
    
def structure_DF(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
           df[col] = pd.to_numeric(df[col])
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
             df[col] = pd.to_datetime(df[col])
        elif pd.api.types.is_categorical_dtype(df[col]):
             df[col] = df[col].astype('category')
        elif pd.api.types.is_object_dtype(df[col]):
             df[col] = df[col].astype('string')
        else:
            pass
    return df    

def encoding_categorical_column(df):
    df_encoded = df.reset_index(drop=True).copy()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'string']).columns
    for col in categorical_columns:
        unique_values = df[col].nunique()
        total_values = len(df[col])
        if unique_values / total_values < 0.05:  
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
        else:  
            ohe = OneHotEncoder(sparse_output=False, drop='first')  
            encoded_cols = pd.DataFrame(
                ohe.fit_transform(df[[col]]),
                columns=[f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
            )
            encoded_cols.index = df_encoded.index
            df_encoded = pd.concat([df_encoded, encoded_cols], axis=1).drop(columns=[col])
    
    return df_encoded

def encoding_categorical_column_2(df):
    df_encoded = df.reset_index(drop=True).copy()
    encoding_models = {}  
    categorical_columns = df.select_dtypes(include=['object', 'category', 'string']).columns
    for col in categorical_columns:
        unique_values = df[col].nunique()
        total_values = len(df[col])

        if unique_values / total_values < 0.05:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoding_models[col] = {'type': 'LabelEncoder', 'model': le}
        else:
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            encoded_cols = pd.DataFrame(
                ohe.fit_transform(df[[col]]),
                columns=[f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
            )
            encoded_cols.index = df_encoded.index
            df_encoded = pd.concat([df_encoded, encoded_cols], axis=1).drop(columns=[col])
            encoding_models[col] = {'type': 'OneHotEncoder', 'model': ohe}

    return df_encoded, encoding_models

def grafik_heatmap_cor(corr):
    fig, ax=plt.subplots(figsize=(10,10))
    im=ax.imshow(corr, vmin=-1,vmax=1,cmap="Blues")
    cbar=ax.figure.colorbar(im,ax=ax)
    cbar.ax.set_ylabel("Barra de Color",rotation= -90)
    
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    
    ax.set_yticklabels(corr.index)
    ax.set_xticklabels(corr.columns)
    
    plt.setp(ax.get_xticklabels(), rotation=90,ha='right',rotation_mode="anchor")
    
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            text=ax.text(j,i,round(corr.iloc[i,j],2),ha="center",va="center",color="w")
    ax.set_title("Correletion Graph")
    fig.tight_layout()
    plt.show()        


def plot_feature_distributions(df, exclude_columns=None, bins=20):
    if exclude_columns:
        df = df.drop(columns=exclude_columns)
    sns.set_theme(style="whitegrid")
    num_features=len(df.columns)
    plt.figure(figsize=(15,num_features * 2))
    
    for i, col in enumerate(df.columns, start=1):
        plt.subplot(num_features,1,i)
        sns.histplot(data=df, x=col, kde=False, bins=bins, color='skyblue')
        plt.title(f'Distribucion de {col}')
        plt.xlabel('')
        plt.tight_layout() 
    plt.show() 

def plot_feature_vs_price(df, price_column, exclude_columns=None):
    if exclude_columns:
        df = df.drop(columns=exclude_columns)
    
    features = [col for col in df.columns if col != price_column]
    num_features = len(features)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, num_features * 2))
    
    for i, col in enumerate(features, start=1):
        plt.subplot(num_features, 1, i)
        sns.scatterplot(data=df, x=col, y=price_column, color='skyblue')
        plt.title(f'Relación entre {price_column} y {col}')
        plt.tight_layout()
        plt.xscale('log') 
        plt.yscale('log')  
        plt.xlabel(f'Feature {col} (log scale)')
        plt.ylabel('Price (log scale)')  
    
    plt.show() 
         
    
def ScalaDF(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled  

def ScalaDF_2(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled, scaler

def feature_impact_Lasso(df):
    X = df.drop(columns=["Price"])
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
    scaler=preprocessing.StandardScaler().fit(X_train)
    X_trainScaled=scaler.transform(X_train)
    X_testScaled=scaler.transform(X_test)
    alpha_vector = np.logspace(-3,1,50)
    param_grid = {'alpha': alpha_vector }
    grid = GridSearchCV(Lasso(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv = 10, return_train_score=True,verbose=1)
    grid.fit(X_trainScaled, y_train)
    
    print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
    print("best parameters: {}".format(grid.best_params_))
    scores = -1*np.array(grid.cv_results_['mean_test_score'])
    plt.semilogx(alpha_vector,scores,'-o')
    plt.xlabel('alpha',fontsize=16)
    plt.ylabel('10-Fold MSE')
    plt.show()
    PredictLasso(grid.best_params_['alpha'],X_trainScaled,X_testScaled,y_test,y_train,X.columns)
 
    
def PredictLasso(bestalpha,XtrainScaled,XtestScaled,y_test,y_train,feature_names):    
    lasso = Lasso(alpha = bestalpha).fit(XtrainScaled,y_train)
    ytrainLasso = lasso.predict(XtrainScaled)
    ytestLasso  = lasso.predict(XtestScaled)
    mseTrainModelLasso = mean_squared_error(y_train,ytrainLasso)
    mseTestModelLasso = mean_squared_error(y_test,ytestLasso)

    print('MSE Modelo Lasso (train): %0.3g' % mseTrainModelLasso)
    print('MSE Modelo Lasso (test) : %0.3g' % mseTestModelLasso)

    print('RMSE Modelo Lasso (train): %0.3g' % np.sqrt(mseTrainModelLasso))
    print('RMSE Modelo Lasso (test) : %0.3g' % np.sqrt(mseTestModelLasso))
    print("\nImpacto de las características en el modelo Lasso:")
    w = lasso.coef_
    for f,wi in zip(feature_names,w):
        print(f,wi)

def features_impact_ELASTICNET(df):
    X = df.drop(columns=['Price'])  
    y = df['Price']                 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    elastic_net = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],  
    alphas=np.logspace(-3, 2, 100),  
    cv=10, 
    random_state=0
    )
    elastic_net.fit(X_train_scaled, y_train)
    y_test_predict=elastic_net.predict(X_test_scaled)
    y_train_predict=elastic_net.predict(X_train_scaled)
    mse_train = mean_squared_error(y_train, y_train_predict)
    r2_train = r2_score(y_train, y_train_predict)
    mse_test = mean_squared_error(y_test, y_test_predict)
    r2_test = r2_score(y_test, y_test_predict)
    print(f"MSE ElasticNet (train): {mse_train:.4f}")
    print(f"R² ElasticNet (train): {r2_train:.4f}")
    print(f"MSE ElasticNet (test): {mse_test:.4f}")
    print(f"R² ElasticNet (test): {r2_test:.4f}")
    print("Best alpha:", elastic_net.alpha_)
    print("Best l1_ratio:", elastic_net.l1_ratio_)
    feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': elastic_net.coef_
     }).sort_values(by='Coefficient', ascending=False)
    feature_importance.plot(kind='bar', x='Feature', y='Coefficient', figsize=(10, 6))
    plt.title('Importancia de las características (ElasticNet)')
    plt.ylabel('Peso del coeficiente')
    plt.xlabel('Características')
    plt.grid()
    plt.show()
    
def BagginReg(df):
    X = df.drop(columns=['Price'])  
    y = df['Price']  
    features = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    param_grid = {
    'estimator__max_depth': range(3, 10),
    }
    kf = KFold(n_splits=8, shuffle=True, random_state=0)
    grid = GridSearchCV(BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=0, n_estimators=200),
                        param_grid=param_grid, cv=kf, verbose=1)
    grid.fit(X_train, y_train)
   
    print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
    print("best parameters: {}".format(grid.best_params_))
    scores = np.array(grid.cv_results_['mean_test_score'])
    plt.plot( range(3,10), scores, '-o')
    plt.axvline(grid.best_params_['estimator__max_depth'], color='r', linestyle='--', label='Mejor max_depth')
    plt.xlabel('max_depth')
    plt.ylabel('Cross-Validation R² Score')
    plt.title('Optimización de max_depth con K-Fold')
    plt.legend()
    plt.show()
    maxDepthOptimo = grid.best_params_['estimator__max_depth']
    baggingModel = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=maxDepthOptimo), 
                                    n_estimators=200, random_state=0).fit(X_train, y_train)
   
    print("Train R^2: ", baggingModel.score(X_train, y_train))
    print("Test R^2: ", baggingModel.score(X_test, y_test))
    y_train_pred = baggingModel.predict(X_train)
    y_test_pred = baggingModel.predict(X_test)
    print("MAE Train: ", mean_absolute_error(y_train, y_train_pred))
    print("MAE Test: ", mean_absolute_error(y_test, y_test_pred))
    print("RMSE Train: ", np.sqrt(mean_squared_error(y_train, y_train_pred)))
    print("RMSE Test: ", np.sqrt(mean_squared_error(y_test, y_test_pred)))
    importances = np.mean([tree.feature_importances_ for tree in baggingModel.estimators_], axis=0)
    importances = importances / np.max(importances)
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 10))
    plt.barh(range(X_train.shape[1]), importances[indices], color='skyblue')
    plt.yticks(range(X_train.shape[1]), features[indices])
    plt.gca().invert_yaxis()
    plt.title('Importancia de las características')
    plt.xlabel('Importancia (normalizada)')
    plt.ylabel('Características')
    plt.show()
    
    
def relationship(df,name_target,name_variable):
    sns.regplot(x=f'{name_variable}', y=f'{name_target}', data=df, ci=None)
    plt.title('Relación entre Número de Habitaciones y Precio (con Línea de Tendencia)')
    plt.xlabel(f'{name_variable}')
    plt.ylabel(f'{name_target}')
    plt.show()
    
def features_impact_XGBRegresor(df):    
    X = df.drop(columns=['Price'])
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=0)
    param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        verbose=1,
        n_jobs=-1
    )  
    grid_search.fit(X_train, y_train)
    best_xgb_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    importances = best_xgb_model.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances, color='skyblue')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    plt.title('Importancia de las Características en el Modelo XGBoost')
    plt.gca().invert_yaxis()
    plt.show()

def filtrar_columnas(df, columnas):
    columnas_existentes = [col for col in columnas if col in df.columns]
    return df[columnas_existentes]    

def plot_histograms(df, bins=20):
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        print("No hay columnas numéricas en el DataFrame.")
        return
    num_columns = len(numeric_columns)
    num_rows = (num_columns + 2) // 3 
    plt.figure(figsize=(15, num_rows * 4))
    for i, col in enumerate(numeric_columns, start=1):
        plt.subplot(num_rows, 3, i)
        plt.hist(df[col].dropna(), bins=bins, color='skyblue', edgecolor='black')
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

def save_label_encoder(encodes):
    for code in encodes.keys():
       with open(f'models/label_encoder_{code}.pkl', 'wb') as le_file:
          pickle.dump(encodes[code], le_file)    