---
layout: post
title: Predicting Customer Loyalty Using Machine Learning
image: "../img/posts/regression-title-img.png"
tags: [Customer Loyalty, Machine Learning, Regression, Python]
---

Our client, a grocery retailer, hired a market research consulting agency to append market-level customer loyalty information to the database. However, only around half of the client's customer base could be tagged, thus the other half did not have this information present. In this project I use Machine Learning to predict the missing information.

# Table of Contents

- [00. Data Source](#data-source)
- [01. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth and Next Steps](#overview-growth)
    - [Key Definition](#overview-definition)
- [02. Data Overview](#data-overview)
- [03. Modeling Overview](#modeling-overview)
- [04. Linear Regression](#linreg-title)
- [05. Decision Tree](#regtree-title)
- [06. Random Forest](#rf-title)
- [07. Modeling Summary](#modeling-summary)
- [08. Predicting Missing Loyalty Scores](#modeling-predictions)
- [09. Growth and Next Steps](#growth-next-steps)

___

### Data Source <a name="data-source"></a>

Dataset provided as part of a data science training program. The data is designed to reflect a real-world business scenario.

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client, a grocery retailer, hired a market research consulting agency to append market-level customer loyalty information to the database. However, only around half of the client's customer base could be tagged, thus the other half did not have this information present.

The goal of this project is to accurately predict the *loyalty score* for those customers who could not be tagged. This will give our client a clearer understanding of true customer loyalty, regardless of total spend volume and will allow more accurate and relevant customer tracking, targeting, and communications.

To achieve this, we built out a model that finds relationships between customer metrics and *loyalty score* for those customers who were tagged. This model is used to predict the loyalty score metric for customers who do not yet have a loyalty score in the database.

<br>

### Actions <a name="overview-actions"></a>

We first gathered the necessary data from tables in the database, including key customer metrics that may help predict the dependent variable *loyalty score*, and appending the dependent variable. We then separated the data into two chunks --- one for customers who did have a *loyalty score* present and one for customers who did not have a *loyalty score*.

As the *loyalty score* is a numerical output, we tested three regression modeling approaches:

* Linear Regression
* Decision Tree
* Random Forest

<br>

### Results <a name="overview-results"></a>

Our testing found that the Random Forest had the highest predictive accuracy.

**Metric 1: Adjusted $R^2$ (Test Set)**

* Random Forest = 0.955
* Decision Tree = 0.886
* Linear Regression = 0.754

**Metric 2: $R^2$ (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.925
* Decision Tree = 0.871
* Linear Regression = 0.853

As the most important outcome for this project was predictive accuracy, rather than explicitly understanding weighted drivers of prediction, we chose the Random Forest as the model to use for making predictions for the customers who were missing a *loyalty score*.

<br>

### Growth and Next Steps <a name="overview-growth"></a>

While predictive accuracy was relatively high, other modeling approaches could be tested, especially those somewhat similar to Random Forest, to see if even more accuracy could be gained.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available as possible for predicting customer loyalty.

<br>

### Key Definition  <a name="overview-definition"></a>

The *loyalty score* metric measures the % of grocery spend (market-level) that each customer allocates to the client vs. all of the client's competitors. 

Example 1: Customer A has spent $100 total on groceries, and all of this was spent with our client. Customer A has a *loyalty score* of **1.0**.

Example 2: Customer B has spent $200 total, but only 20% was spent with our client. The remaining 80% was spent with competitors. Customer B has a *customer loyalty score* of **0.2**.

___

# Data Overview  <a name="data-overview"></a>

We will be predicting the *loyalty_score* metric. This metric exists (for half of the customer base) in the *loyalty_scores* table of the client database.

The key variables hypothesized to predict the missing loyalty scores will come from the client database, namely the *transactions* table, the *customer_details* table, and the *product_areas* table.

Using the Pandas library in Python, we merge these tables together for all customers, creating a single dataset that can be used for modeling.

```python

# Import required packages
import pandas as pd
import pickle

# Import the data
loyalty_scores = ...
customer_details = ...
transactions = ...

# Merge loyalty score data and customer details data, at the customer level
data_for_regression = pd.merge(customer_details, loyalty_scores, how = 'left', on = 'customer_id')

# Aggregate sales data from the transactions table
sales_summary = transactions.groupby('customer_id').agg({'sales_cost': 'sum',
                                                         'num_items': 'sum',
                                                         'transaction_id': 'count',
                                                         'product_area_id': 'nunique'}).reset_index()

# Rename columns for clarity
sales_summary.columns = ['customer_id', 'total_sales', 'total_items', 'transaction_count', 'product_area_count']

# Add a column with the average cart value for each customer
sales_summary['average_cart_value'] = sales_summary.total_sales / sales_summary.transaction_count

# Merge the sales summary with the overall customer data
data_for_regression = pd.merge(data_for_regression, sales_summary, how = 'inner', on = 'customer_id')

# Split out the data for modeling (customers with loyalty score) and for predicting (loyalty score is missing)
regression_modeling = data_for_regression.loc[data_for_regression['customer_loyalty_score'].notna()]
regression_scoring = data_for_regression.loc[data_for_regression['customer_loyalty_score'].isna()]
regression_scoring.drop(['customer_loyalty_score'], axis = 1, inplace = True)

# Save our datasets for future use
pickle.dump(regression_modeling, open('data/abc_regression_modeling.p', 'wb'))
pickle.dump(regression_scoring, open('data/abc_regression_scoring.p', 'wb'))

```

<br>

After this data preprocessing in Python, we have a dataset for modeling that contains the following fields:

| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| loyalty_score | Dependent | The % of total grocery spending that each customer allocates to the client vs. competitors |
| distance_from_store | Independent | The distance in miles from the customer's home address to the store |
| gender | Independent | The gender provided by the customer |
| credit_score | Independent | The customer's most recent credit score |
| total_sales | Independent | Total spending by the customer with the client within the latest 6 months |
| total_items | Independent | Total products purchased by the customer with the client within the latest 6 months |
| transaction_count | Independent | Total unique transactions made by the customer with the client within the latest 6 months |
| product_area_count | Independent | The number of product areas in the client's store that customers have shopped within the latest 6 months |
| average_cart_value | Independent | The average amount spent per transaction for the customer with the client within the latest 6 months |

___

# Modeling Overview

We will build a model that accurately predicts the *loyalty_score* metric for customers that were able to be tagged, based on the customer metrics listed above.

If an accurate model can be achieved, we will use the model to predict the customer loyalty score for the customers that were unable to be tagged by the agency.

As the *loyalty score* is a numerical output, we tested three regression modeling approaches:

* Linear Regression
* Decision Tree
* Random Forest

___

# Linear Regression <a name="linreg-title"></a>

We use the scikit-learn library in Python to model the data using Linear Regression. The code sections below are broken up into four key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>

### Data Import <a name="linreg-import"></a>

We import the modeling data from the pickle file we saved. We remove the id column, and we also shuffle the data.

```python

# Import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

# Import sample data
data_for_model = pd.read_pickle('data/abc_regression_modeling.p')

# Drop unnecessary columns
data_for_model.drop('customer_id', axis = 1, inplace = True)

# Shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

```

<br>

### Data Preprocessing <a name="linreg-preprocessing"></a>

For Linear Regression we have certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* The effect of outliers
* Encoding categorical variables to numeric form
* Multicollinearity and feature selection

<br>

##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows.

```python

# Remove rows where values are missing
data_for_model.isna().sum()
data_for_model.dropna(how = 'any', inplace = True)

```

<br>

##### Outliers

The fit of a Linear Regression model can be negatively impacted if there are outliers present. There is no right or wrong way to deal with outliers, but it is something worth careful consideration on a case by case basis. The key thing to keep in mind is that a value being high or low does not automatically mean it should be excluded.

In this code section, we use **.describe()** from Pandas to investigate the spread of values for each of the predictor variables. The results of this can be seen in the table below.

| **metric** | **distance_from_store** | **credit_score** | **total_sales** | **total_items** | **transaction_count** | **product_area_count** | **average_cart_value** |
|---|---|---|---|---|---|---|---|
| mean | 2.02 | 0.60 | 1846.50 | 278.30 | 44.93 | 4.31 | 36.78 |
| std | 2.57 | 0.10 | 1767.83 | 214.24 | 21.25 | 0.73 | 19.34 |
| min | 0.00 | 0.26 | 45.95 | 10.00 | 4.00 | 2.00 | 9.34 |
| 25% | 0.71 | 0.53 | 9.07 | 201.00 | 41.00 | 4.00 | 22.41 |
| 50% | 1.65 | 0.59 | 1471.49 | 258.50 | 50.00 | 4.00 | 30.37 |
| 75% | 2.91 | 0.66 | 2104.73 | 318.50 | 53.00 | 5.00 | 47.21 |
| max | 44.37 | 0.88 | 9878.76 | 1187.00 | 109.00 | 5.00 | 102.34 |

<br>

Based on this investigation, we see some *max* column values for *distance_from_store*, *total_sales*, and *total_items* are much higher than the *median* value. For example, the median *distance_from_store* is 1.65 miles, but the maximum is over 44 miles.

We use the "boxplot approach" to remove any rows where the values within those predictor variable columns are outside of the interquartile range multiplied by 2.

```python

# Deal with outliers
outlier_investigation = data_for_model.describe()
outlier_columns = ['distance_from_store', 'total_sales', 'total_items']

# Boxplot approaach
for column in outlier_columns:
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iq_range = upper_quartile - lower_quartile
    iqr_extended = iq_range * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f'{len(outliers)} outliers detected in column {column}')
    
    data_for_model.drop(outliers, inplace = True)

```

<br>

##### Split Out Data For Modeling

In the next code block we split the data into an **X** object which contains only the independent variables and a **y** object that contains only the dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can validate the accuracy of the predictions on data that was not used in training. We have allocated 80% of the data for training, and the remaining 20% for validation.

```python

# Split data into input variables & output variable (X & y) for modeling
X = data_for_model.drop(['customer_loyalty_score'], axis = 1)
y = data_for_model['customer_loyalty_score']

# Split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

<br>

##### Categorical Predictor Variables

In our dataset, we have one categorical variable *gender* which has values of "M" for Male, "F" for Female, and "U" for Unknown.

The Linear Regression algorithm needs this variable to be in numerical form in order to assess the relationship with the dependent variable. We apply One Hot Encoding to the categorical column.

One Hot Encoding is a way to represent categorical variables as binary vectors --- a set of *new* columns for each categorical value (in our case for "M", "F", and "U") with either a 1 or a 0 saying whether that value is true or not for that observation. These new columns would go into our model as input variables, and the original column is discarded.

We also drop one of the new columns using the parameter *drop = "first"*. We do this because our newly created encoded columns perfectly predict each other, which violates the assumption that there is no multicollinearity, an important consideration for linear regression. Multicollinearity occurs when two or more input variables are highly correlated with each other, and when it is present we cannot trust the statistics around how well the model is performing and the effect each input variable is truly having.

After we have applied One Hot Encoding, we turn our training and test objects back into Pandas dataframes, with the column names applied.

```python

# List of categorical variables for encoding
categorical_vars = ['gender']

# Instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse_output = False, drop = 'first')

# Apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Turn objects back to pandas dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>

##### Feature Selection

Feature Selection is the process used to select the input variables that are most important to your Machine Learning task. The potential benefits of Feature Selection are:

* **Improved Model Accuracy** - eliminating noise can help true relationships stand out
* **Lower Computational Cost** - the model becomes faster to train, and faster to make predictions
* **Explainability** - understanding and explaining outputs for stakeholder and customers becomes much easier

There are many ways to apply Feature Selection, ranging from simple methods such as a *Correlation Matrix* showing variable relationships, to *Univariate Testing* which helps us understand statistical relationships between variables, to more powerful approaches like *Recursive Feature Elimination (RFE)* --- an approach that starts with all input variables, and then iteratively removes variables with the weakest relationships with the output variable.

For our task we applied a variation of Recursive Feature Elimination called *Recursive Feature Elimination With Cross Validation (RFECV)*. We split the data into many chunks, and the RFECV algorithm iteratively trains and validates models on each chunk separately. Each time we assess different models with different variables included or eliminated, the algorithm knows how accurate each of those models was. From the suite of model scenarios that are created, the algorithm can determine which provided the best accuracy, and from that we can infer the best set of input variables to use.

```python

# Instantiate RFECV and the model type to be used
regressor = LinearRegression()
feature_selector = RFECV(regressor)     # uses 5-fold cross validation by default (splits into 5 chunks)

# Fit RFECV to the training data
fit = feature_selector.fit(X_train, y_train)

# Extract and print the optimal number of features
optimal_feature_count = feature_selector.n_features_
print(f'Optimal number of features: {optimal_feature_count}')

# Limit our training and test sets to only include the selected varaibles
X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

```

<br>

The code below produces a plot that visualizes the cross-validated accuracy with each potential number of features.

```python

plt.style.use('seaborn-v0_8-poster')
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()

```

<br>

This creates the following plot, which shows us that the highest cross-validated accuracy (0.8625) is achieved when we include all eight of our original input variables. This is marginally higher than six or seven included variables. We will continue on with all eight.

![alt text](/img/posts/lin-reg-feature-selection-plot.png "Linear Regression Feature Selection Plot")

<br>

### Model Training <a name="linreg-model-training"></a>

Instantiating and training the Linear Regression model is done using the below code:

```python

# Model training
regressor = LinearRegression()
regressor.fit(X_train, y_train)

```

<br>

### Model Performance Assessment <a name="linreg-model-assessment"></a>

##### Predict On The Test Set

To assess how well our model is predicting for new data, we use the trained model object to predict the *loyalty_score* variable for the test set.

```python

# Predict on the test set
y_pred = regressor.predict(X_test)

```

<br>

##### Calculate $R^2$

$R^2$ is a metric that gives the percentage of variance in the output variable *y* that is being explained by the input variable(s) *x*. The value can range between 0 and 1, with a higher value showing a higher level of explained variance. For example, an $R^2$ of 0.8 would mean that 80% of the variation in our output variable is being explained by our input variables, and another variable(s) not in the model accounts for the other 20%.

To calculate $R^2$, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test):

```python

# Calculate R-Squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)

```

The resulting $R^2$ score from this is **0.78**.

<br>

##### Calculate Cross-Validated $R^2$

An even more powerful and reliable way to assess model performance is to utilize Cross Validation.

Instead of simply dividing our data into a single training set, and a single test set, with Cross Validation we break our data into a number of chunks and then iteratively train the model on all but one of the chunks, testing the model on the remaining chunk until each chunk has had a chance to be the test set.

The result of this is that we are provided more than one set of validation results from the tests, and the average of these scores gives a more reliable view of how our model will perform on new data.

In the code below, we specify that we want 4 chunks. We pass in our regressor object, training set, and test set. We also specify the metric we want to assess with, in this case $R^2$.

Finally, we take a mean of all four test set results.

```python

# Calculate the mean cross-validated R-Squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = 'r2')
cv_scores.mean()

```

The mean cross-validated $R^2$ score from this is **0.853**.

<br>

##### Calculate Adjusted $R^2$

When applying Linear Regression with *multiple* input variables, the $R^2$ metric is an overestimate of the goodness of fit. This is because each input variable will have an *additive* effect on the overall $R^2$ score, i.e., every input variable added to the model increases the $R^2$ value and never decreases it, even if the relationship is by chance. 

**Adjusted $R^2$** is a metric that compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by chance. It is best practice to use Adjusted $R^2$ when assessing the results of a Linear Regression with multiple input variables, as it gives a fairer perception the fit of the data, so this is done in the following code:

```python

# Calculated Adjusted R-Squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

```

The resulting adjusted $R^2$ score from this is **0.754**. As expected, this is slightly lower than the $R^2$.

<br>

### Model Summary Statistics <a name="linreg-model-summary"></a>

Although the overall goal for this project is predictive accuracy, rather than an explicit understanding of the relationships of each of the input variables to the output variable, it is always interesting to look at the summary statistics for these.

```python

# Extract model coefficients
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names, coefficients], axis = 1)
summary_stats.columns = ['input_variable', 'coefficient']

# Extract model intercept
regressor.intercept_

```

<br>

The information from that code block can be found in the table below:

| **input_variable** | **coefficient** |
|---|---|
| intercept | 0.516 |
| distance_from_store | -0.201 |
| credit_score | -0.028 |
| total_sales | 0.000 |
| total_items | 0.001 |
| transaction_count | -0.005 |
| product_area_count | 0.062 |
| average_cart_value | -0.004 |
| gender_M | -0.013 |

<br>

The coefficient value for each of the input variables plus the intercept would make up the equation for the line of best fit for this particular model (or more accurately, the plane of best fit, as we have multiple input variables). For each input variable, the coefficient value we see above tells us how many units the output variable (loyalty score) would change with a *one unit change* in this particular input variable, if every other input variable remained constant.

For example, in the table above the *distance_from_store* input variable has a coefficient value of -0.201. So the *loyalty_score* decreases by about 20% for every additional mile that a customer lives from the store. This makes intuitive sense, as customers who live far from the store most likely live near another store where they might do some of their shopping as well, whereas customers who live near this store probably do a greater proportion of their shopping at this store and therefore have a higher loyalty score.

___

# Decision Tree <a name="regtree-title"></a>

We will again utilize the scikit-learn library within Python to model our data using a Decision Tree. The code sections below are broken up into four key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>

### Data Import <a name="regtree-import"></a>

We import the modeling data from the pickle file we saved. We remove the id column, and we also shuffle the data.

```python

# Import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

# Import modeling data
data_for_model = pd.read_pickle('data/abc_classification_modeling.p')

# Drop unnecessary columns
data_for_model.drop('customer_id', axis = 1, inplace = True)

# Shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

```

<br>

### Data Preprocessing <a name="regtree-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers and highly correlated input variables, Decision Trees are not, so the required preprocessing here is lighter. We still, however, put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>

##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows.

```python

# Remove rows with missing values
data_for_model.isna().sum()
data_for_model.dropna(how = 'any', inplace = True)

```

<br>

##### Split Out Data For Modeling

In the same way we did for Linear Regression, in the next code block we split the data into an **X** object which contains only the independent variables and a **y** object that contains only the dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can validate the accuracy of the predictions on data that was not used in training. We have allocated 80% of the data for training, and the remaining 20% for validation.

```python

# Split data into X and y objects for modeling
X = data_for_model.drop(['customer_loyalty_score'], axis = 1)
y = data_for_model['customer_loyalty_score']

# Split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

<br>

##### Categorical Predictor Variables

In our dataset, we have one categorical variable *gender* which has values of "M" for Male, "F" for Female, and "U" for Unknown.

Just like the Linear Regression algorithm, the Decision Tree cannot deal with data in this format as it can't assign any numerical meaning to it when assessing the relationship between the variable and the dependent variable. We again apply One Hot Encoding to the categorical column.

```python

# List of categorical variables
categorical_vars = ['gender']

# Instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse_output = False, drop = 'first')

# Apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Turn objects back to pandas dataframes
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>

### Model Training <a name="regtree-model-training"></a>

The below code instantiates and trains our Decision Tree model. We use the *random_state* parameter to ensure we get reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.

```python

# Instantiate the model object
regressor = DecisionTreeRegressor(random_state = 42)

# Fit the model using training sets
regressor.fit(X_train, y_train)

```

<br>

### Model Performance Assessment <a name="regtree-model-assessment"></a>

##### Predict On The Test Set

To assess how well our model is predicting for new data, we use the trained model object to predict the *loyalty_score* variable for the test set.

```python

# Predict on the test set
y_pred = regressor.predict(X_test)

```

<br>

##### Calculate $R^2$

To calculate $R^2$, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test):

```python

# Calculate R-Squared for test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)

```

The resulting $R^2$ score is **0.898**.

<br>

##### Calculate Cross Validated $R^2$

As we did when testing Linear Regression, we will again use Cross Validation.

Instead of simply dividing our data into a single training set, and a single test set, with Cross Validation we break our data into a number of chunks and then iteratively train the model on all but one of the chunks, testing the model on the remaining chunk until each chunk has had a chance to be the test set.

The result of this is that we are provided more than one set of validation results from the tests, and the average of these scores gives a more reliable view of how our model will perform on new data.

In the code below, we again specify 4 chunks. We pass in our regressor object, training set, and test set, and specify $R^2$ for scoring.

Finally, we take a mean of all four test set results.

```python

# Calculate the mean cross-validated r-squared for test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = 'r2')
cv_scores.mean()

```
<br>

The mean cross-validated $R^2$ score from this is **0.876** which is slighter higher than what we saw for Linear Regression.

<br>

##### Calculate Adjusted $R^2$

Just like we did with Linear Regression, we will also calculate the Adjusted $R^2$ to compensate for the addition of input variables.

```python

# Calculated Adjusted R-Squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

```
<br>

The resulting adjusted $R^2$ score from this is **0.887**, which is slightly lower than the score we got for $R^2$, as expected.

<br>

### Decision Tree Regularization <a name="regtree-model-regularization"></a>

Decision Trees can be prone to over-fitting --- without any limits on their splitting, they will learn the training data perfectly. It is better to have a model with a more generalized set of rules, as this will be more robust and reliable when making predictions on new data.

One effective method of avoiding this over-fitting is to apply a *max depth* to the Decision Tree, meaning we only allow it to split the data a certain number of times before it is required to stop.

To find the best number of splits to use for this, the below code loops over a variety of values for max depth. We will assess which gives us the best predictive performance.

```python

# Finding the best max_depth

# Set up range for search, and empty list to append accuracy scores to
max_depth_list = list(range(1, 9))
accuracy_scores = []

# Loop through each possible depth, train and validate the model, and append the accuracy
for depth in max_depth_list:
    regressor = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Get max accuracy and optimal depth
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# Plot accuracy by max depth
plt.plot(max_depth_list, accuracy_scores)
plt.tick_params(axis='both', labelsize=10)
plt.scatter(optimal_depth, max_accuracy, marker = 'x', color = 'red')
plt.title(f'Accuracy by Max Depth \nOptimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy, 4)})', fontsize = 10)
plt.xlabel('Max Depth', fontsize = 10)
plt.ylabel('Accuracy', fontsize = 10)
plt.tight_layout()
plt.show()

```

<br>

The code gives us the below plot to visualize the result:

![alt text](/img/posts/regression-tree-max-depth-plot.png "Decision Tree Max Depth Plot")

<br>

In the plot we can see that the *maximum* classification accuracy on the test set is found when applying a *max_depth* value of 7. However, we lose very little accuracy with a value of 4, and this would result in a simpler model that can generalize even better on new data. We make the decision to re-train our Decision Tree with a maximum depth of 4.

<br>

### Visualize Our Decision Tree <a name="regtree-visualize"></a>

To see the decisions that have been made in the (re-fitted) tree, we can use the **plot_tree** functionality that we imported from scikit-learn. To do this, we use the below code:

```python

# Re-fit using max depth of 4
regressor = DecisionTreeRegressor(max_depth = 4, random_state = 42)
regressor.fit(X_train, y_train)

# Plot the nodes of the decision tree
plt.figure(figsize=(25, 15))
tree = plot_tree(regressor, 
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)

```

<br>

That code gives us the below plot:

![alt text](/img/posts/regression-tree-nodes-plot.png "Decision Tree Max Depth Plot")

<br>

This is a visual that can be shown to stakeholders in the business to ensure they understand exactly what is driving the predictions. For example, one thing that could be noted is that the very first split uses the variable *distance_from_store*, implying this is an important variable when it comes to predicting loyalty.

<br>

___

# Random Forest <a name="rf-title"></a>

We will again utilize the scikit-learn library within Python to model our data using a Random Forest. The code sections below are broken up into four key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>

### Data Import <a name="rf-import"></a>

Again, we import the modeling data from the pickle file we saved. We remove the id column, and we also shuffle the data.

```python

# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

# import modeling data
data_for_model = pickle.load(open("data/customer_loyalty_modeling.p", "rb"))

# drop unnecessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

```

<br>

### Data Preprocessing <a name="rf-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers and highly correlated input variables, Random Forests (like Decision Trees) are not, so the required preprocessing here is lighter. We still, however, put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>

##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows.

```python

# remove rows where values are missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)

```

<br>

##### Split Out Data For Modeling

In the same way we did for Linear Regression, in the next code block we split the data into an **X** object which contains only the independent variables and a **y** object that contains only the dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can validate the accuracy of the predictions on data that was not used in training. We have allocated 80% of the data for training, and the remaining 20% for validation.

<br>

```python

# split data into X and y objects for modeling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

<br>

##### Categorical Predictor Variables

In our dataset, we have one categorical variable *gender* which has values of "M" for Male, "F" for Female, and "U" for Unknown.

Just like the Linear Regression algorithm, Random Forests cannot deal with data in this format as it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As *gender* doesn't have any explicit *order* to it, in other words, Male isn't higher or lower than Female and vice versa - we would again apply One Hot Encoding to the categorical column.

```python

# list of categorical variables that need encoding
categorical_vars = ["gender"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>

### Model Training <a name="rf-model-training"></a>

Instantiating and training our Random Forest model is done using the below code. We use the *random_state* parameter to ensure we get reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.

We leave the other parameters at their default values, meaning that we will just be building 100 Decision Trees in this Random Forest.

```python

# instantiate our model object
regressor = RandomForestRegressor(random_state = 42)

# fit our model using our training & test sets
regressor.fit(X_train, y_train)

```

<br>

### Model Performance Assessment <a name="rf-model-assessment"></a>

##### Predict On The Test Set

To assess how well our model is predicting on new data - we use the trained model object (here called *regressor*) and ask it to predict the *loyalty_score* variable for the test set

```python

# predict on the test set
y_pred = regressor.predict(X_test)

```

<br>

##### Calculate $R^2$

To calculate $R^2$, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test)

```python

# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)

```

The resulting $R^2$ score from this is **0.957** - higher than both Linear Regression and the Decision Tree.

<br>

##### Calculate Cross Validated $R^2$

As we did when testing Linear Regression and our Decision Tree, we will again utilize Cross Validation (for more info on how this works, please refer to the Linear Regression section above)

```python

# calculate the mean cross validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()

```

<br>

The mean cross-validated $R^2$ score from this is **0.923** which agian is higher than we saw for both Linear Regression and our Decision Tree.

<br>
##### Calculate Adjusted $R^2$

Just like we did with Linear Regression and our Decision Tree, we will also calculate the *Adjusted $R^2$* which compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability.

```python

# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

```

The resulting adjusted $R^2$ score from this is **0.955** which as expected, is slightly lower than the score we got for $R^2$ on its own - but again higher than for our other models.

<br>
### Feature Importance <a name="rf-model-feature-importance"></a>

In our Linear Regression model, to understand the relationships between input variables and our output variable, loyalty score, we examined the coefficients. With our Decision Tree we looked at what the earlier splits were. These allowed us some insight into which input variables were having the most impact.

Random Forests are an ensemble model, made up of many, many Decision Trees, each of which is different due to the randomness of the data being provided, and the random selection of input variables available at each potential split point.

Because of this, we end up with a powerful and robust model, but because of the random or different nature of all these Decision trees - the model gives us a unique insight into how important each of our input variables are to the overall model. 

As we’re using random samples of data, and input variables for each Decision Tree - there are many scenarios where certain input variables are being held back and this enables us a way to compare how accurate the models predictions are if that variable is or isn’t present.

So, at a high level, in a Random Forest we can measure *importance* by asking *How much would accuracy decrease if a specific input variable was removed or randomized?*

If this decrease in performance, or accuracy, is large, then we’d deem that input variable to be quite important, and if we see only a small decrease in accuracy, then we’d conclude that the variable is of less importance.

At a high level, there are two common ways to tackle this. The first, often just called **Feature Importance** is where we find all nodes in the Decision Trees of the forest where a particular input variable is used to split the data and assess what the Mean Squared Error (for a Regression problem) was before the split was made, and compare this to the Mean Squared Error after the split was made. We can take the *average* of these improvements across all Decision Trees in the Random Forest to get a score that tells us *how much better* we’re making the model by using that input variable.

If we do this for *each* of our input variables, we can compare these scores and understand which is adding the most value to the predictive power of the model!

The other approach, often called **Permutation Importance** cleverly uses some data that has gone *unused* at when random samples are selected for each Decision Tree (this stage is called "bootstrap sampling" or "bootstrapping")

These observations that were not randomly selected for each Decision Tree are known as *Out of Bag* observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the *Out of Bag* observations are gathered and then passed through. Once all of these observations have been run through the Decision Tree, we obtain an accuracy score for these predictions, which in the case of a regression problem could be Mean Squared Error or $R^2$.

In order to understand the *importance*, we *randomize* the values within one of the input variables - a process that essentially destroys any relationship that might exist between that input variable and the output variable - and run that updated data through the Decision Tree again, obtaining a second accuracy score. The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

*Permutation Importance* is often preferred over *Feature Importance* which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.

Let's put them both in place, and plot the results...

<br>
```python

# calculate feature importance
feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names,feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable","feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

# plot feature importance
plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# calculate permutation importance
result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state = 42)
permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names,permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable","permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

# plot permutation importance
plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

```
<br>
That code gives us the below plots - the first being for *Feature Importance* and the second for *Permutation Importance*!

<br>
![alt text](/img/posts/rf-regression-feature-importance.png "Random Forest Feature Importance Plot")
<br>
<br>
![alt text](/img/posts/rf-regression-permutation-importance.png "Random Forest Permutation Importance Plot")

<br>
The overall story from both approaches is very similar, in that by far, the most important or impactful input variable is *distance_from_store* which is the same insights we derived when assessing our Linear Regression and Decision Tree models.

There are slight differences in the order or "importance" for the remaining variables but overall they have provided similar findings.

___
<br>
# Modeling Summary  <a name="modeling-summary"></a>

The most important outcome for this project was predictive accuracy, rather than explicitly understanding the drivers of prediction. Based upon this, we chose the model that performed the best when predicted on the test set - the Random Forest.

<br>
**Metric 1: Adjusted $R^2$ (Test Set)**

* Random Forest = 0.955
* Decision Tree = 0.886
* Linear Regression = 0.754

<br>
**Metric 2: $R^2$ (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.925
* Decision Tree = 0.871
* Linear Regression = 0.853

<br>
Even though we were not specifically interested in the drivers of prediction, it was interesting to see across all three modeling approaches, that the input variable with the biggest impact on the prediction was *distance_from_store* rather than variables such as *total sales*. This is interesting information for the business, so discovering this as we went was worthwhile.

<br>
# Predicting Missing Loyalty Scores <a name="modeling-predictions"></a>

We have selected the model to use (Random Forest) and now we need to make the *loyalty_score* predictions for those customers that the market research consultancy were unable to tag.

We cannot just pass the data for these customers into the model, as is - we need to ensure the data is in exactly the same format as what was used when training the model.

In the following code, we will

* Import the required packages for preprocessing
* Import the data for those customers who are missing a *loyalty_score* value
* Import our model object and any preprocessing artifacts
* Drop columns that were not used when training the model (customer_id)
* Drop rows with missing values
* Apply One Hot Encoding to the gender column (using transform)
* Make the predictions using .predict()

<br>
```python

# import required packages
import pandas as pd
import pickle

# import customers for scoring
to_be_scored = ...

# import model and model objects
regressor = ...
one_hot_encoder = ...

# drop unused columns
to_be_scored.drop(["customer_id"], axis = 1, inplace = True)

# drop missing values
to_be_scored.dropna(how = "any", inplace = True)

# apply one hot encoding (transform only)
categorical_vars = ["gender"]
encoder_vars_array = one_hot_encoder.transform(to_be_scored[categorical_vars])
encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)
encoder_vars_df = pd.DataFrame(encoder_vars_array, columns = encoder_feature_names)
to_be_scored = pd.concat([to_be_scored.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis = 1)
to_be_scored.drop(categorical_vars, axis = 1, inplace = True)

# make our predictions!
loyalty_predictions = regressor.predict(to_be_scored)

```
<br>
Just like that, we have made our *loyalty_score* predictions for these missing customers. Due to the impressive metrics on the test set, we can be reasonably confident with these scores. This extra customer information will ensure our client can undertake more accurate and relevant customer tracking, targeting, and comms.

___
<br>
# Growth and Next Steps <a name="growth-next-steps"></a>

While predictive accuracy was relatively high - other modeling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

We could even look to tune the hyperparameters of the Random Forest, notably regularization parameters such as tree depth, as well as potentially training on a higher number of Decision Trees in the Random Forest.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty.




