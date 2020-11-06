import pandas as pd

df = pd.read_csv('Life Expectancy Data - cleaned.csv')

# Variable to be predicted is'Life expectancy'
y = df['Life expectancy ']

country_list = df['Country'].unique().tolist()

# Creating dictonary for mapping 'Country' feature into numbers
country_mapping = dict()
i = 1

for country in country_list:
    country_mapping[country] = i
    i+=1
    
# Mapping 'Country' feature into numbers
df['Country'] = df['Country'].map(country_mapping)

# Features to be used for prediction
X = df.drop(['Life expectancy '], axis=1)

# Using Lasso Regression
from sklearn.linear_model import Lasso

# Here, taking alpha = 0.000001
lasso_reg = Lasso(alpha=0.000001)

lasso_reg.fit(X,y)

# Importing pickle for deployment
import pickle

pickle_out = open('final_project.pkl','wb')
pickle.dump(lasso_reg,pickle_out)
pickle_out.close()