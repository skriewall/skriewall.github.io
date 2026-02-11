---
layout: post
title: Understanding Alcohol Product Relationships Using Association Rule Learning
image: "../img/posts/association-rules-title-img.png"
tags: [Association Rule Learning, Python]
---

In this project I use Association Rule Learning to analyze the transactional relationships between products in the alcohol section of a grocery store.

# Table of Contents

- [00. Data Source](#data-source)
- [01. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth and Next Steps](#overview-growth)
- [02. Data Overview](#data-overview)
- [03. Apriori Overview](#apriori-overview)
- [04. Data Preparation](#apriori-data-prep)
- [05. Applying The Apriori Algorithm](#apriori-fit)
- [06. Interpreting The Results](#apriori-results)
- [07. Growth and Next Steps](#growth-next-steps)

___

### Data Source <a name="data-source"></a>

Dataset provided as part of a data science training program. The data is designed to reflect a real-world business scenario.

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client wants to reorganize the alcohol section within their store, but is unsure what arrangement would be better. Customers often complain they can't find the products they want, and they also ask for recommendations on other products to try. Also, the client's marketing team would like to start running bundled promotions as this has worked well in other areas of the store, but they need guidance with selecting which products to put together.

They have provided a sample of 3,500 alcohol transactions to see if we can find any insights that might help the business address these problems.

<br>

### Actions <a name="overview-actions"></a>

Based on the two open-ended tasks, we apply a specific type of Association Rule Learning called *Apriori* to examine and analyze the strength of various relationships between different products represented in the transactional data.

We installed the *apyori* package in Python, which contains the required functionality for this association rule learning task.

We imported the sample data, and got it into the right format for the Apriori algorithm to deal with.

The we applied the Apriori algorithm to provide us with the following metrics:

* Support
* Confidence
* Expected Confidence
* Lift

These metrics examine product relationships in different ways. We use each one to suggest ideas that address each of the tasks at hand. You can read more about these metrics and the Apriori algorithm in the relevant sections below.

<br>

### Results <a name="overview-results"></a>

The strongest relationship exists between two products labeled as "gifts". The store's category managers may want to ensure that gift products are available in one section of the aisle, rather than being spread out among their respective product types. We also see some strong relationships between French wines and other French wines, indicating that grouping wine in sections by country rather than by type may make it easier for customers to find what they are looking for.

Another interesting association is between products labeled "small". At this point, we do not know exactly what that means since we are not given any more information about the data, but it is something to tell the client as they may be able to make sense of it and turn it into an actionable insight. Perhaps customers are looking for a few different types of wine in smaller bottles than the standard size.

Lastly, it could be useful to create a search tool for category managers which they can use to look up products by keyword in the product association table. For example, we searched for any products that are strongly associated with "New Zealand" products. There appears to be *some* relationship between New Zealand wines and other New Zealand wines. Many New Zealand wines seem to be more closely associated with French, Spanish (Iberian), and South American wines than they are with Australian wines (interesting given the relative proximity of New Zealand to Australia). New Zealand and Australia are often grouped together, but in terms of wine this does not seem to make sense --- possibly different climates matter much more than geographical proximity, as it can affect how similar the wines taste. This is only a hypothesis, but good information to share with the client.

<br>

### Growth and Next Steps <a name="overview-growth"></a>

As this is an exploratory project, we will present the results to the client's Category Managers. We will discuss the results, our views on what actions might be taken based on these insights, and any considerations that need to be taken into account when interpreting the results.

We will also propose building a "Keyword Search Engine" which will help Category Managers extract and make use of the insights held within the data.

Association rule learning could also be applied within all other product categories, as well as across the full-product range.

___

# Data Overview  <a name="data-overview"></a>

The initial dataset contains 3,500 transactions, each of which shows the alcohol products that were present in that transaction. 

In the code below, we import *pandas*, as well as the Apriori algorithm from the *apyori* library, and we import the raw data.

```python

# Import required packages
from apyori import apriori
import pandas as pd

# Import data
alcohol_transactions = pd.read_csv('data/sample_data_apriori.csv')

```

<br>

The first 10 transactions in the data can be seen below. There are 45 product columns in the data; to simplify, only the first 5 product columns are shown. Blank columns indicate a product was not in the transaction.

| **transaction_id** | **product1** | **product2** | **product3** | **product4** | **product5** | **…** |
|---|---|---|---|---|---|---|
| 1 | Premium Lager | Iberia |  |  |  | ... |
| 2 | Sparkling | Premium Lager | Premium Cider | Own Label | Italy White | ... |
| 3 | Small Sizes White | Small Sizes Red | Sherry Spanish | No/Low Alc Cider | Cooking Wine | ... |
| 4 | White Uk | Sherry Spanish | Port | Italian White | Italian Red | ... |
| 5 | Premium Lager | Over-Ice Cider | French White South | French Rose | Cocktails/Liqueurs | ... |
| 6 | Kosher Red |  |  |  |  | ... |
| 7 | Own Label | Italy White | Australian Red |  |  | ... |
| 8 | Brandy/Cognac |  |  |  |  | ... |
| 9 | Small Sizes White | Bottled Ale |  |  |  | ... |
| 10 | White Uk | Spirits Mixers | Sparkling | German | Australian Red | ... |
| ... | ... | ... | ... | ... | ... | ... |

<br>

The *apyori* library needs the data to be passed in as a *list of lists*, so we will modify the format of this data. The code and logic for this can be found in the Data Preparation section below.

___

# Apriori Overview  <a name="apriori-overview"></a>

*Association Rule Learning* is an approach that discovers the strength of relationships between different datapoints. It is commonly used to understand which products are frequently (or infrequently) purchased together.

This can provide useful information that helps to optimize:

* Product Arrangement/Placement (making the customer route through the store more efficient)
* Product Recommendations (customers who purchased product A also purchased product B)
* Bundled Discounts (which products should/should not be put together)

One powerful and common algorithm for Association Rule Learning is **Apriori**.

In Apriori there are four key metrics:

* Support
* Confidence
* Expected Confidence
* Lift

Each of these metrics helps us to understand, in different ways, items and their relationships with other items.

<br>

##### Support

The *Support* metric is simply the percentage of all transactions that contain both Item A and Item B and is a way of measuring how commmon or popular a particular pair of items is. It is calculated by counting all transactions that include both items and dividing that number by the total number of transactions.

<br>

##### Confidence

*Confidence* looks more explicitly at the relationship between the two items. It is calculated by counting all transactions that include both Item A and Item B and then dividing by the number of transactions that contained item A. Confidence answers the question, "Of all transactions that included Item A, what proportion also included Item B?"  

A high score for Confidence can mean a strong product relationship, but not always. When one of the items is very popular in general, the score can be inflated. We can also look at two further metrics to guide us --- Expected Confidence and Lift.

<br>

##### Expected Confidence

*Expected Confidence* is the percentage of all transactions that contained Item B. It provides an indication of what the Confidence *would be* if there was no relationship between the items.

<br>

##### Lift

*Lift* is the factor by which the Confidence exceeds the Expected Confidence. Lift indicates how likely it is that Item B is purchased when Item A is purchased, while controlling for how popular Item B is. Lift is calculated by dividing Confidence by Expected Confidence.

A Lift score *greater than 1* indicates that Items A and B appear together *more often* than expected, and conversely a Lift score *less than 1* indicates that Items A and B appear together *less often* than expected.

<br>

##### Using the Four Metrics

The metrics above are calculated between *all* pairs of products. We can then sort by Lift score (for example) and see exactly what the strongest or weakest relationships were. This information can guide the client's decisions regarding product layout, recommendations for customers, or promotions.

<br>

##### An Important Consideration

One thing to note when assessing the results of Apriori is that product relationships that have a *high Lift score* but a *low Support score* should be interpreted with caution. The relationship with the highest Lift score might appear to be very strong, but it could be that the relationship is only by chance due to the rarity of the item set.

___

# Data Preparation  <a name="apriori-data-prep"></a>

As mentioned in the Data Overview section above, the *apyori* algorithm needs the data passed in as a *list of lists*. The following code modifies it: 

The code below does the following:

* Removes the ID column as it is not relevant to this analysis
* Iterates over the DataFrame, creates a list out of each transaction (row), and appends that list to a master list
* Prints out the first 10 lists from the master list

```python

# Drop ID column
alcohol_transactions.drop('transaction_id', axis = 1, inplace = True)

# Modify data for Apriori algorithm
transactions = []
for index, row in alcohol_transactions.iterrows():
    transaction = list(row.dropna())
    transactions.append(transaction)

# Print out the first 10 transactions
print(transactions[:10])

>>[['Premium Lager', 'Iberia'],
['Sparkling', 'Premium Lager', 'Premium Cider', 'Own Label', 'Italy White', 'Italian White', 'Italian Red', 'French Red', 'Bottled Ale'],
['Small Sizes White', 'Small Sizes Red', 'Sherry Spanish', 'No/Low Alc Cider', 'Cooking Wine', 'Cocktails/Liqueurs', 'Bottled Ale'],
['White Uk', 'Sherry Spanish', 'Port', 'Italian White', 'Italian Red'],
['Premium Lager', 'Over-Ice Cider', 'French White South', 'French Rose', 'Cocktails/Liqueurs', 'Bottled Ale'],
['Kosher Red'],
['Own Label', 'Italy White', 'Australian Red'],
['Brandy/Cognac'],
['Small Sizes White', 'Bottled Ale'],
['White Uk', 'Spirits Mixers', 'Sparkling', 'German', 'Australian Red', 'American Red']]

```

<br>

Each transaction from the initial DataFrame is now contained within a list, all transactions together making up the master list.

___

# Applying The Apriori Algorithm <a name="apriori-fit"></a>

In the code below we apply the Apriori algorithm from the apyori library.

We set the following association rules for the algorithm:

* A minimum *Support* of 0.003 to eliminate very rare product sets
* A minimum *Confidence* of 0.2
* A minimum *Lift* of 3 to ensure we only focus on product sets with strong relationships
* A minimum and maximum length of 2 to limit the algorithm to considering product *pairs* rather than larger sets

```python

# Apply the Apriori algorithm
apriori_rules = apriori(transactions, 
                        min_support = 0.003,
                        min_confidence = 0.2,
                        min_lift = 3.0,
                        min_length = 2,
                        max_length = 2)

# Convert the output to a list
apriori_rules = list(apriori_rules)

# Examine the first element
apriori_rules[0]

RelationRecord(items=frozenset({'America White', 'American Rose'}), support=0.020745724698626296, ordered_statistics=[OrderedStatistic(items_base=frozenset({'American Rose'}), items_add=frozenset({'America White'}), confidence=0.5323741007194245, lift=3.997849299507762)])

```

<br>

The code below converts the output from the algorithm to a list to make it easier to manipulate and analyze. Based on the parameters we set when applying the algorithm, we get 132 product pairs.

```python

# "List comprehension" - extract the information from the rules
product1 = [list(rule[2][0][0])[0] for rule in apriori_rules]
product2 = [list(rule[2][0][1])[0] for rule in apriori_rules]
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2][0][2] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

# Gather into a single dataframe
apriori_rules_df = pd.DataFrame({'product1' : product1,
                                 'product2' : product2,
                                 'support' : support,
                                 'confidence' : confidence,
                                 'lift' : lift
                                 })

```

<br>

A sample of this data (the first 5 product pairs --- not in any order) can be seen below:

| **product1** | **product2** | **support** | **confidence** | **lift** |
|---|---|---|---|---|
| American Rose | America White | 0.021 | 0.532 | 3.998 |
| America White | American White | 0.054 | 0.408 | 3.597 |
| Australian Rose | America White | 0.005 | 0.486 | 3.653 |
| Low Alcohol A.C | America White | 0.003 | 0.462 | 3.466 |
| American Rose | American Red | 0.016 | 0.403 | 3.575 |
| … | … | … | … | … |

<br>

In the DataFrame we have the two products in the pair being considered, and the three key metrics of Support, Confidence, and Lift. 

___

# Interpreting The Results <a name="apriori-results"></a>

### Associated Products

Now that the data and Apriori results are in a usable format, we can look at the product pairs with the strongest relationships by sorting the Lift column in descending order.

```python

# Sort pairs by descending lift
apriori_rules_df.sort_values(by = 'lift', ascending = False, inplace = True)

```

<br>

The table below shows the ten highest product relationships, based on Lift.

| **product1** | **product2** | **support** | **confidence** | **lift** |
|---|---|---|---|---|
| Wine Gifts | Beer/Lager Gifts | 0.004 | 0.314 | 10.173 |
| Beer/Lager Gifts | Spirits & Fortified | 0.013 | 0.427 | 9.897 |
| Wine Gifts | Spirits & Fortified | 0.006 | 0.412 | 9.537 |
| Red Wine Bxes & 25Cl | White Boxes | 0.015 | 0.474 | 9.344 |
| French White Rhone | French Red | 0.003 | 0.480 | 8.691 |
| Small Sizeswhite Oth | Small Sizes White | 0.005 | 0.559 | 8.340 |
| Small Sizes Red | Small Sizes White | 0.025 | 0.486 | 7.258 |
| French White Loire | French White South | 0.004 | 0.349 | 6.763 |
| French White Rhone | French White 2 | 0.005 | 0.760 | 6.661 |
| Small Sizeswhite Oth | Small Sizes Red | 0.003 | 0.324 | 6.306 |

<br>

The strongest relationship exists between two products labeled as "gifts". The store's category managers may want to ensure that gift products are available in one section of the aisle, rather than being spread out among their respective product types. We also see some strong relationships between French wines and other French wines, indicating that grouping wine in sections by country rather than by type may make it easier for customers to find what they are looking for.

Another interesting association is between products labeled "small". At this point, we do not know exactly what that means since we are not given any more information about the data, but it is something to tell the client as they may be able to make sense of it and turn it into an actionable insight. Perhaps customers are looking for a few different types of wine in smaller bottles than the standard size.

<br>

### Search Tool For Category Managers

With the data stored as a DataFrame, we can also propose building a simple search tool for Category Managers to use. An example of how this might work would be to check associations with wines from New Zealand. Perhaps we guess that wines from Australia will be associated with wines from New Zealand, since Australia and New Zealand are often grouped together.

The code below uses a string function to get all rows in the DataFrame where *product1* contains the words "New Zealand".

```python

# Search for associations based on products containing text - New Zealand products
apriori_rules_df[apriori_rules_df['product1'].str.contains('New Zealand')]

```

<br>

The results of this search, in order of descending Lift are as follows:

| **product1** | **product2** | **support** | **confidence** | **lift** |
|---|---|---|---|---|
| New Zealand Red | Malt Whisky | 0.005 | 0.271 | 5.629 |
| New Zealand Red | Iberia White | 0.007 | 0.371 | 4.616 |
| New Zealand Red | New Zealand White | 0.013 | 0.643 | 4.614 |
| New Zealand Red | French White South | 0.004 | 0.229 | 4.431 |
| New Zealand Red | French White 2 | 0.010 | 0.486 | 4.257 |
| New Zealand Red | French Red | 0.004 | 0.214 | 3.880 |
| New Zealand Red | French Red South | 0.006 | 0.329 | 3.868 |
| New Zealand Red | South America | 0.011 | 0.557 | 3.800 |
| New Zealand Red | Other Red | 0.004 | 0.229 | 3.592 |
| New Zealand Red | Iberia | 0.012 | 0.614 | 3.528 |
| New Zealand Red | Champagne | 0.009 | 0.443 | 3.526 |
| New Zealand White | South America White | 0.049 | 0.354 | 3.423 |
| New Zealand Red | French Red 2 | 0.010 | 0.514 | 3.360 |
| New Zealand Red | South America White | 0.007 | 0.343 | 3.314 |
| New Zealand Red | Australia White | 0.007 | 0.371 | 3.216 |

<br>

There appears to be *some* relationship between New Zealand wines and other New Zealand wines. Many New Zealand wines seem to be more closely associated with French, Spanish (Iberian), and South American wines than they are with Australian wines (interesting given the relative proximity of New Zealand to Australia). New Zealand and Australia are often grouped together, but in terms of wine this does not seem to make sense --- possibly different climates matter much more than geographical proximity, as it can affect how similar the wines taste. This is only a hypothesis, but good information to share with the client.

___

# Growth and Next Steps <a name="growth-next-steps"></a>

As this is an exploratory project, we will present the results to the client's Category Managers. We will discuss the results, our views on what actions might be taken based on these insights, and any considerations that need to be taken into account when interpreting the results.

We will also propose building a "Keyword Search Engine" which will help Category Managers extract and make use of the insights held within the data.

Association rule learning could also be applied within all other product categories, as well as across the full-product range.
