# Import libraries
import pandas as pd
import pickle
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Define the log transformer
log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)


from sklearn.model_selection import train_test_split , GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("diamonds.csv")
list(set(data.dtypes.tolist()))


num_cols = data.select_dtypes(include = ['int64','float64'])
num_cols.columns

X = data.drop('price', axis=1)
y = data['price']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


features_with_outliers = []
for feature in num_cols:
    percentile25 = data[feature].quantile(0.25)
    percentile75 = data[feature].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    outliers = data[(data[feature] > upper_limit) | (data[feature] < lower_limit)]
    proportion_of_outliers = len(outliers) / len(data) * 100
    if len(outliers) > 0:
        features_with_outliers.append(feature)
train_data = pd.concat([x_train, y_train], axis=1)

train_data = train_data[(train_data["depth"] < 75) & (train_data["depth"] > 45)]
train_data = train_data[(train_data["table"] < 89) & (train_data["table"] > 40)]
train_data = train_data[(train_data["x"]<15)]
train_data = train_data[(train_data["y"]<15)]
train_data = train_data[(x_train["z"]<15)]

x_train=train_data.drop('price',axis=1)
y_train=train_data['price']

x_train_copy = x_train.copy()

x_train_copy['x']=x_train_copy['x'].replace(0, x_train_copy['x'].median())
x_train_copy['y']=x_train_copy['y'].replace(0, x_train_copy['y'].median())
x_train_copy['z']=x_train_copy['z'].replace(0, x_train_copy['z'].median())

num_cols = x_train_copy.select_dtypes(include = ['int64','float64'])

from sklearn.preprocessing import OrdinalEncoder

# Define the ordinal categories for each column in the correct order
ordinal_categories = [
    ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],  # for 'cut'
    ['J', 'I', 'H', 'G', 'F', 'E', 'D'],  # for 'color'
    ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']  # for 'clarity'
]


ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)

# Apply the encoder to the columns
x_train_copy[['cut', 'color', 'clarity']] = ordinal_encoder.fit_transform(x_train_copy[['cut', 'color', 'clarity']])

x_train_copy['volume']=(x_train_copy.x*x_train_copy.y*x_train_copy.z)
x_train_copy.drop(['x','y','z'],axis=1,inplace=True)

numerical_features = ['carat', 'depth', 'table', 'volume']
categorical_features = ['cut', 'color', 'clarity']
columns_to_drop = ['x','y','z']

class ReplaceZerosWithMedian(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        X = pd.DataFrame(X)
        # Calculate the median of each column, excluding 0s
        self.medians_ = X.apply(lambda col: col[col > 0].median())
        return self

    def transform(self, X):

        X = pd.DataFrame(X)
        # Replace 0s with median values
        for col in X.columns:
            X[col] = np.where(X[col] == 0, self.medians_[col], X[col])
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a new volume feature from x, y, z
        X['volume'] = X['x'] * X['y'] * X['z']

        X = X.drop(columns=['x', 'y', 'z'], errors='ignore')

        return X
    
    log_transformer = FunctionTransformer(np.log1p, validate=True)

numerical_pipeline = Pipeline(steps=[
    ('replace_zeros', ReplaceZerosWithMedian()),
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transform', log_transformer),
    ('scaler', StandardScaler())
])

# Categorical Pipeline: Imputation + One-Hot Encoding
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=ordinal_categories))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('feature_engineering', FeatureEngineer()),
    ('preprocessor', preprocessor)
])

x_train = pipeline.fit_transform(x_train)
x_test = pipeline.transform(x_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "KNN Regressor": KNeighborsRegressor(),
    "Support Vector Regressor": SVR(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "XGBoost Regressor": XGBRegressor()


}


param_grids = {
    "Linear Regression": {},
    "Ridge Regression": {"model__alpha": [0.1, 1, 10]},
    "Lasso Regression": {"model__alpha": [0.1, 1, 10]},
    "KNN Regressor": {"model__n_neighbors": [3, 5, 7]},
    "Support Vector Regressor": {"model__kernel": ["linear", "rbf"], "model__C": [1, 10]},
    "Decision Tree Regressor": {"model__max_depth": [None, 5, 10]},
    "Random Forest Regressor": {"model__n_estimators": [100, 200], "model__max_depth": [None, 5]},
    "AdaBoost Regressor": {"model__n_estimators": [50, 100], "model__learning_rate": [0.1,0.05]},
    "Gradient Boosting Regressor": {"model__n_estimators": [100, 150], "model__learning_rate": [0.1, 0.05]},
    "XGBoost Regressor": {"model__n_estimators": [100, 150], "model__learning_rate": [0.1, 0.05]}

}

def train_model(model_name, model, param_grid, x_train, y_train):
    pipeline = Pipeline([('model', model)])
    random_search = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring='r2')
    random_search.fit(x_train, y_train)
    best_model = random_search.best_estimator_
    return best_model

model_name = "Random Forest Regressor"
model = models[model_name]
param_grid = param_grids[model_name]
best_model = train_model(model_name, model, param_grid, x_train, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(x_train, y_train)


# Saving the model to disk
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

# Assume you have already loaded your trained model (like rf_model or best_model)
# If you want to use the RandomForestRegressor model (rf_model), you can pickle it and load it
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the function to make predictions
def predict_diamond_price(carat, cut, color, clarity, depth, table, volume):
    """
    Function to predict the diamond price based on input features.
    
    Parameters:
    - carat (float): Carat weight of the diamond
    - cut (str): Cut quality (Fair, Good, Very Good, Premium, Ideal)
    - color (str): Diamond color (D, E, F, G, H, I, J)
    - clarity (str): Clarity (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
    - depth (float): Depth percentage
    - table (float): Table percentage
    - volume (float): Calculated volume (x * y * z)

    Returns:
    - price (float): Predicted diamond price
    """
    
    # Categorical encoding using your ordinal encoder
    cut_map = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_map = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity_map = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    
    # Map cut, color, and clarity to their ordinal values
    cut_encoded = cut_map.index(cut)
    color_encoded = color_map.index(color)
    clarity_encoded = clarity_map.index(clarity)

    # Combine features into a single array
    features = np.array([[carat, cut_encoded, color_encoded, clarity_encoded, depth, table, volume]])

    # Make a prediction using the pre-trained model
    predicted_price = model.predict(features)

    # Return the predicted price
    return predicted_price[0]
