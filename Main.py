#%%
import pandas as pd 
import numpy as np  
import matplotlib as mp 
import matplotlib.pyplot as plt
import seaborn as sns
from pygments.styles.dracula import background
from sklearn import linear_model

from sklearn.linear_model import LinearRegression 
#%%
data = pd.read_csv('2017_2020_bmi.csv')
df = data[data['height'] >= 100]
print(df.head())
#%%
x_1 = df ['height'].values
x_1 [0:20]
#%%
x_1.shape
#%%
x_1 = df ['height'].values.reshape(-1,1)
print ('shape', x_1.shape)
print (x_1[0:10])
#%%
y = df ['weight'].values
y [0:20]
#%%
model_1 = LinearRegression()
model_1.fit(x_1, y)
slope = model_1.coef_[0]
intercept=model_1.intercept_
print(f"Intercept: {intercept}")
print(f"Slope: {slope}")
#%%
plt.scatter(x_1, y, s=10, marker='x', color = 'red')
plt.plot(x_1, model_1.predict(x_1), color='purple', label='regression line')

equation_text = f'y = {slope:.3f}*x + {intercept:.3f}'
plt.text(120, 50, equation_text, color='black', fontsize=10)

plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.title('Weight/height growth')
plt.legend(loc='lower right')
plt.show()
#%%
sns.lmplot(x='height', y='weight', data=df)
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.title('Weight/height growth')
plt.show()
#%%
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('weight ~ height', data=df).fit()  # Here, weight is the dependent variable
print (model.summary())
#%%
# Create regression diagnostic plots for height predicting weight
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(model, 'height', fig=fig)
plt.show()

#%%
df = data[['age', 'height', 'weight', 'bmi']].copy()  # Create a copy to avoid SettingWithCopyWarning

# Define a mapping for age categories to midpoints
age_mapping = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
    '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 
    '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
    '16': 16, '17': 17, '18': 18, '19-44': 31.5,
    '45-64': 54.5, '65+': 70
}

# Replace age categories and ensure numeric types
df.loc[:, 'age'] = df['age'].astype(str).replace(age_mapping)  # Use .loc and convert to str first
df.loc[:, 'age'] = pd.to_numeric(df['age'], errors='coerce')  # Convert to numeric

# Drop rows with NaN values in the specified columns
df = df.dropna(subset=['age', 'height', 'weight', 'bmi'])

# Ensure all relevant columns are numeric
df.loc[:, 'height'] = pd.to_numeric(df['height'], errors='coerce')
df.loc[:, 'weight'] = pd.to_numeric(df['weight'], errors='coerce')
df.loc[:, 'bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# Drop any remaining NaN values
df = df.dropna(subset=['age', 'height', 'weight', 'bmi'])

# Fit a multiple regression model
X = df[['height', 'age', 'bmi']]  # Independent variables
y = df['weight']  # Dependent variable

# Check and convert data types for OLS
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop any remaining NaN values after conversion
X = X.dropna()
y = y[X.index]  # Ensure y matches the index of X

X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()  # Fit the model
print(model.summary())  # Print model summary

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

#%%
# Create a 3D scatter plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df['height'], df['age'], df['weight'], c='yellow', marker='o', label='Data points')

# Create a grid for the regression plane
height_range = np.linspace(df['height'].min(), df['height'].max(), 10)
age_range = np.linspace(df['age'].min(), df['age'].max(), 10)
height_grid, age_grid = np.meshgrid(height_range, age_range)

# Predict weight based on the regression model
bmi_mean = df['bmi'].mean()  # Use the mean BMI for the prediction
weight_pred = (
    model.params['const'] + 
    model.params['height'] * height_grid + 
    model.params['age'] * age_grid + 
    model.params['bmi'] * bmi_mean
)

# Plot the regression plane
ax.plot_surface(height_grid, age_grid, weight_pred, color='purple', alpha=0.5)

# Labels and title
ax.set_xlabel('Height (cm)')
ax.set_ylabel('Age (years)')
ax.set_zlabel('Weight (kg)')
ax.set_title('3D Scatter Plot with Regression Plane')
ax.legend()

plt.show()
