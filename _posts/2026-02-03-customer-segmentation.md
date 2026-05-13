---
layout: post
title: The "You Are What You Eat" Customer Segmentation
image: "../img/posts/clustering-title-img.png"
tags: [Customer Segmentation, Machine Learning, Clustering, Python]
---

In this project I use *k*-means clustering to segment the customer base in order to increase business understanding, which can enhance the relevancy of targeted messaging and customer communications.

# Table of Contents

- [00. Data Source](#data-source)
- [01. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth and Next Steps](#overview-growth)
- [02. Data Overview](#data-overview)
- [03. *K*-means](#kmeans-title)
    - [Concept Overview](#kmeans-overview)
    - [Data Preprocessing](#kmeans-preprocessing)
    - [Finding A Good Value For K](#kmeans-k-value)
    - [Model Fitting](#kmeans-model-fitting)
    - [Appending Clusters To Customers](#kmeans-append-clusters)
    - [Segment Profiling](#kmeans-cluster-profiling)
- [04. Application](#kmeans-application)
- [05. Growth and Next Steps](#growth-next-steps)

___

### Data Source <a name="data-source"></a>

Dataset provided as part of a data science training program. The data is designed to reflect a real-world business scenario.

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Members of the Senior Management team from our client, a grocery retailer, are disagreeing about how customers are shopping and how lifestyle choices may affect which food categories customers are or are not purchasing.

We will use data and Machine Learning to segment or group customers based on customer engagement with each of the major food categories. This will aid business understanding of the customer base, and that understanding can be used by the client to enhance the relevancy of targeted messaging and customer communications.

<br>

### Actions <a name="overview-actions"></a>

We first gathered the necessary data from two tables in the client database --- the *transactions* table and the *product_areas* table. We joined together the relevant information using Pandas, and then aggregated transactional data across product areas at the customer level. The final data for clustering is the percentage of sales allocated to each product area for each individual customer.

As a starting point, we test and apply *k*-means clustering for this task. We apply some data pre-processing, most importantly feature scaling to ensure all variables exist on the same scale.

As *k*-means is an *unsupervised learning* approach, meaning we are trying to understand patterns or trends rather than trying to predict outcomes, we use a process known as *Within Cluster Sum of Squares (WCSS)* to understand what a "good" number of clusters or segments is. We then apply the *k*-means algorithm to the product area data and profile the resulting customer segments to understand what the differentiating factors were.

<br>

### Results <a name="overview-results"></a>

Based upon iterative testing using WCSS we settled on a customer segmentation with 3 clusters. These clusters ranged in size, with Cluster 1 accounting for 73.6% of the customer base, Cluster 0 accounting for 14.6%, and Cluster 2 accounting for 11.8%.

There were some interesting findings from profiling the clusters.

For *Cluster 1* we saw a significant portion of spending being allocated to each of the four product areas --- showing customers without any particular dietary preference. 

For *Cluster 2* we saw high proportions of spending being allocated to Fruit and Vegetables, but very little to Dairy and Meat. It could be hypothesized that these customers are following a vegan diet. 

Finally, customers in *Cluster 0* spent significant portions within Dairy, Fruit and Vegetables, but very little in the Meat product area, so similarly, we could make a hypothesis that these customers are following a vegetarian diet.

To help embed this segmentation into the business, we have proposed to call this the "You Are What You Eat" segmentation.

<br>

### Growth and Next Steps <a name="overview-growth"></a>

It would be interesting to run this clustering/segmentation at a lower level of product areas. Rather than just the four areas of Meat, Dairy, Fruit, and Vegetables, we could cluster spending across the sub-categories *below* those categories. This would mean we could create more specific clusters, and get a more granular understanding of dietary preferences within the customer base. If we had more data on other product categories (e.g., Bakery, Deli, Packaged Foods, etc.), we could also see if any additional spending patterns emerge.

Also, for this analysis we have only focused on variables that are linked directly to sales. It could be interesting to include customer metrics such as distance to store or gender to give an even more well-rounded customer segmentation.

We could also compare these results to other clustering approaches such as hierarchical clustering or DBSCAN.

___

# Data Overview  <a name="data-overview"></a>

We are primarily interested in segments of customers based on their transactions within *food*-based product areas, so we want to only select data related to food purchases.

In the code below, we:

* Import the required python packages and libraries
* Import the tables from the database
* Merge the tables to tag on *product_area_name*, which only exists in the *product_areas* table
* Drop the non-food categories
* Aggregate the sales data for each product area, at the customer level
* Pivot the data to get it into the right format for clustering
* Change the values from raw dollars into a percentage of spending for each customer to ensure each customer is comparable

```python
# Import required Python packages
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# Import tables
transactions = ...
product_areas = ...

# Merge on product area name
transactions = pd.merge(transactions, product_areas, how = 'inner', on = 'product_area_id')

# Drop the non-food category
transactions.drop(transactions[transactions['product_area_name'] == 'Non-Food'].index, inplace = True)

# Aggregate sales at customer level (by product area)
transaction_summary = transactions.groupby(['customer_id', 'product_area_name'])['sales_cost'].sum().reset_index()

# Pivot data to place product areas as columns
transaction_summary_pivot = transactions.pivot_table(index = 'customer_id',
                                                    columns = 'product_area_name',
                                                    values = 'sales_cost',
                                                    aggfunc = 'sum',
                                                    fill_value = 0,
                                                    margins = True,
                                                    margins_name = 'Total').rename_axis(None, axis = 1)
                                          
# Turn sales into % sales
transaction_summary_pivot = transaction_summary_pivot.div(transaction_summary_pivot['Total'], axis = 0)

# Drop the "Total" column
data_for_clustering = transaction_summary_pivot.drop(['Total'], axis = 1)
```

After the data pre-processing using Pandas, the dataset for clustering looks like the below sample:

| **customer_id** | **Dairy** | **Fruit** | **Meat** | **Vegetables** |
|---|---|---|---|---|
| 1 | 0.272 | 0.204 | 0.401 | 0.123  |
| 2 | 0.246 | 0.198 | 0.394 | 0.162  |
| 3 | 0.142 | 0.233 | 0.528 | 0.097  |
| 4 | 0.341 | 0.245 | 0.272 | 0.142  |
| 5 | 0.213 | 0.250 | 0.430 | 0.107  |
| 6 | 0.180 | 0.178 | 0.546 | 0.095  |
| 7 | 0.000 | 0.517 | 0.000 | 0.483  |

The data is at the customer level, and we have a column for each of the highest level food product areas. Within each of those columns for the food product areas, we have the percentage of spending that each customer allocated to that product area over the past six months.

___

# *K*-Means <a name="kmeans-title"></a>

### Concept Overview <a name="kmeans-overview"></a>

*K*-means is an *unsupervised learning* algorithm, meaning it tries to isolate patterns within unlabeled data rather than predicting known labels, values, or outcomes. The algorithm partitions data points into distinct groups (clusters/segments) based on their *similarity* to each other. This similarity is most often based on the Euclidean (straight-line) distance between data points in n-dimensional space. Each variable that is included counts as another dimension in space. The number of clusters is determined by the value that is set for *k*. We use a process known as *Within Cluster Sum of Squares (WCSS)* to understand what a "good" number for *k* (the number of clusters) is. We then apply the *k*-means algorithm to the product area data and profile the resulting customer segments to understand what the differentiating factors were.

The algorithm does this iteratively over four key steps:

1. It selects *k* random points (known as centroids) in space.
2. It assigns each of the data points to the nearest centroid (based on Euclidean distance).
3. It repositions the centroids to the *mean* of the values in its cluster.
4. It reassigns each data point to the nearest centroid.

Steps 3 & 4 are repeated until no data points are reassigned to a closer centroid.

<br>

### Data Preprocessing <a name="kmeans-preprocessing"></a>

There are three vital preprocessing steps for *k*-means:

* Missing values in the data
* The effect of outliers
* Feature Scaling

<br>

##### Missing Values

Missing values can cause issues for *k*-means, as the algorithm will not be able to find the distance along the dimension where the value is not present. If we have observations with missing values, the most common options are to remove the observations or to use an imputer to fill or estimate what the missing values might be.

As we aggregated the data for each customer, we actually do not have missing values and do not need to address any missing values in this project.

<br>

##### Outliers

Outliers can cause problems in distance-based algorithms like *k*-means. The main issue presented by outliers is when scaling is performed on the input variables. If variables are “bunched up” close to each other due to a single outlier value, it will make it hard to compare their values to the other input variables. We should always investigate outliers rigorously, but as we are dealing with percentages between 0 and 1, outliers are not an issue in this case.

<br>

##### Feature Scaling

As *k*-means is a *distance-based* algorithm, i.e., it depends on how similar or different data points are across different dimensions in n-dimensional space, the application of *Feature Scaling* is extremely important.

In Feature Scaling we force the values from different columns to exist on the same scale in order to enhance the learning capabilities of the model. There are two common approaches for this --- *Standardization* and *Normalization*.

Standardization rescales data to have a mean of 0, and a standard deviation of 1, meaning most datapoints will most often fall between values of around -4 and +4.

Normalization rescales datapoints so that they exist in a range between 0 and 1.

Here, we normalize the data as this will ensure all variables will have the same range, fixed between 0 and 1. The variables would also be compatible with any categorical variables that we have encoded as 1’s and 0’s (although there are not any categorical variables in our task here). We have put the data in the form of percentages, so our values are already spread between 0 and 1. However, we still normalize the data because one of the product areas might commonly make up a large proportion of customer sales, and this may end up dominating the clustering space. If we normalize all of our variables, even product areas that make up smaller volumes will be spread proportionately between 0 and 1.

The below code uses MinMaxScaler from scikit-learn to apply Normalization to all of the variables.

```python
# Create the scaler object
scale_norm = MinMaxScaler()

# Normalize the data
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering),
                                          columns = data_for_clustering.columns)
```

<br>

### Finding A Good Value For k <a name="kmeans-k-value"></a>

At this point, the data is ready to be fed into the *k*-means clustering algorithm. But first we want to understand what number of clusters (*k*) to use in the algorithm. There is no right or wrong value for this --- it really depends on the data and what the goal is. For our client and the question at hand, having a high number of clusters might not be appropriate as it would be too hard for the business to understand the nuance of each in a way where they can apply the right strategies.

We use a data-driven approach known as *Within Cluster Sum of Squares (WCSS)* to find a "good" value for *k*. WCSS measures the sum of the squared Euclidean distances of each data point from its closest centroid. Comparing WCSS for different values of *k* can help us identify the point where adding *more clusters* (increasing *k*) provides little additional benefit.

By default, the *k*-means algorithm in scikit-learn will use k = 8, i.e., it will split the data into eight distinct clusters. In the code below multiple values for *k* are tested. We then plot how the WCSS metric changes for each value of *k*. As we increase the value for *k* the WCSS value will always decrease. However, the decreases will get smaller and smaller each time we add another centroid or cluster. We will be looking for a point in the plot where this decrease is quite prominent and significant right *before* we start to see diminishing returns.

We specify *n_init* = 10, meaning the *k*-means algorithm will run 10 different times. The *k* centroids' values will be different from run to run since they are randomized (see Step 1 above). The WCSS that we record for each *k* will be from whichever run resulted in the best (lowest) WCSS.

```python
# Set up range for search, create empty list to track WCSS
k_values = list(range(1, 10))
wcss_list = []

# Loop through each possible value of k, fit to the data, and record the WCSS
for k in k_values:
    kmeans = KMeans(n_clusters = k, n_init = 10, random_state = 42)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)

# Plot WCSS by k
plt.plot(k_values, wcss_list)
plt.title('Within Cluster Sum of Squares -  by k')
plt.xlabel('k')
plt.ylabel('WCSS Score')
plt.tight_layout()
plt.show()
```

The code gives us the below plot:

![K-Means Optimal k Value Plot](/img/posts/kmeans-optimal-k-value-plot.png "K-Means Optimal k Value Plot")

Based on the shape of the above plot, there appears to be a sharp bend at *k* = 3. Prior to that we see a significant drop in the WCSS score, but following *k* = 3 the decreases in WCSS are much smaller, so this is a point that suggests adding *more clusters* will provide little extra benefit in terms of separating our data. A small number of clusters can be beneficial when considering how easy it is for the business to understand each one, so we will fit the *k*-means clustering solution with *k* = 3.

<br>

### Model Fitting <a name="kmeans-model-fitting"></a>

The below code will instantiate our *k*-means object with *k* = 3. We fit this object to the scaled dataset to separate the data into three distinct segments or clusters.

```python
# Instantiate k-means object
kmeans = KMeans(n_clusters = 3, n_init = 10, random_state = 42)

# Fit to the data
kmeans.fit(data_for_clustering_scaled)
```

<br>

### Append Clusters To Customers <a name="kmeans-append-clusters"></a>

With the *k*-means algorithm fitted to our data, we tag each customer with the cluster number that they most closely fit into based on their sales data over each product area. In the code below we add the cluster numbers onto the original dataframe.

```python
# Add cluster labels to the original data
data_for_clustering['cluster'] = kmeans.labels_
```

<br>

### Cluster Profiling <a name="kmeans-cluster-profiling"></a>

Once the data is separated into distinct clusters, our client needs to understand *what* is driving the separation. This means the business can understand the customers within each cluster and the behaviors that make them unique.

##### Cluster Sizes

First we assess how many customers fall into each cluster:

```python
# Check cluster sizes
data_for_clustering["cluster"].value_counts(normalize=True)
```

Running this code shows that the three clusters are different in size, with the following proportions:

* Cluster 1: **73.6%** of customers
* Cluster 0: **14.6%** of customers
* Cluster 2: **11.8%** of customers

Based on these results, it appears Cluster 1 is larger with Clusters 0 and 2 being proportionally smaller. This is showing us segments of the customer base that are exhibiting different behaviors or characteristics, with some behaviors more common than others.

<br>

##### Cluster Attributes

To understand what these different behaviors or characteristics are, we can analyze the attributes of each cluster in terms of the variables we fed into the *k*-means algorithm.

```python
# Profile the clusters
cluster_summary = data_for_clustering.groupby('cluster')[['Dairy', 'Fruit', 'Meat', 'Vegetables']].mean().reset_index()
```

The code results in the following table:

| **Cluster** | **Dairy** | **Fruit** | **Meat** | **Vegetables** |
|---|---|---|---|---|
| 0 | 36.4% | 39.4% | 2.9% | 21.3%  |
| 1 | 22.1% | 26.5% | 37.7% | 13.8%  |
| 2 | 0.2% | 63.8% | 0.4% | 35.6%  |

For *Cluster 1*, we see a reasonably significant portion of spending being allocated to each of the product areas. For *Cluster 2* there are high proportions of spending being allocated to Fruit and Vegetables, but very little to the Dairy and Meat product areas. It could be hypothesized that these customers are following a vegan diet. Finally, customers in *Cluster 0* spend, on average, significant portions within Dairy, Fruit and Vegetables, but very little in the Meat product area. We could make an early hypothesis that these customers may be following a vegetarian diet. Of course, there could be other things that would explain these spending behaviors, or there could be a mix of behaviors that lead to similar patterns and thus land customers in the same cluster. But this is a good starting point for explaining the patterns we see in the different clusters.

___

# Application <a name="kmeans-application"></a>

Although this is a simple solution based on high level product areas, it can help leaders in the business gain a clearer understanding of the customer base. Tracking these clusters over time would allow the client to react to dietary trends more quickly and adjust their messaging and inventory accordingly. The client will be able to target customers more accurately, promoting products and discounts to customers that are truly relevant to them.

___

# Growth and Next Steps <a name="growth-next-steps"></a>

It would be interesting to run this clustering/segmentation at a lower level of product areas. Rather than just the four areas of Meat, Dairy, Fruit, and Vegetables, we could cluster spending across the sub-categories *below* those categories. This would mean we could create more specific clusters, and get a more granular understanding of dietary preferences within the customer base. If we had more data on other product categories (e.g., Bakery, Deli, Packaged Foods, etc.), we could also see if any additional spending patterns emerge.

Also, for this analysis we have only focused on variables that are linked directly to sales. It could be interesting to include customer metrics such as distance to store or gender to give an even more well-rounded customer segmentation.

We could also compare these results to other clustering approaches such as hierarchical clustering or DBSCAN.
