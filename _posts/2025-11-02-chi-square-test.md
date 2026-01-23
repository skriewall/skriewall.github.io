---
layout: post
title: Assessing Campaign Performance Using the Chi-Squared Test for Independence
image: "../img/posts/ab-testing-title-img.png"
tags: [AB Testing, Hypothesis Testing, Chi-Squared, Python]
---

In this project I apply the Chi-Squared Test For Independence (a hypothesis test) to assess the performance of two types of mailers that were sent out to promote a new service.

# Table of Contents

- 00. [Data Source](#data-source)
- 01. [Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results and Discussion](#overview-results)
- 02. [Concept Overview](#concept-overview)
- 03. [Data Overview and Preparation](#data-overview)
- 04. [Applying the Chi-Squared Test For Independence](#chi-squared-application)
- 05. [Analyzing the Results](#chi-squared-results)
- 06. [Discussion](#discussion)

___

### Data Source <a name="data-source"></a>
Dataset provided as part of a data science training program. The data is designed to reflect a real-world business scenario.

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Earlier in the year, our client, a grocery retailer, ran an ad campaign to promote their new "Delivery Club" --- an initiative that costs $100/year for membership, but offers free grocery deliveries rather than the normal cost of $10 per delivery.

For the ad campaign, customers were randomly split into three groups. Customers in the first group received a low quality, low cost mailer; customers in the second group received a high quality, high cost mailer; and the third group was a control group, where customers received no mailer.

The client knows that customers who were contacted signed up for the Delivery Club at a far higher rate than the control group, but the client now wants to understand if there is a significant difference in sign-up rate between the cheap and expensive mailers. This will enable them to make more informed decisions in the future, with the overall goal of optimizing their campaign return on investment (ROI).

<br>

### Actions <a name="overview-actions"></a>

For this test, as it is focused on comparing the *rates* of two groups, we applied the Chi-Squared Test For Independence. What follows is a summary of the test and results. Full details of this test can be found in the dedicated section below.

From the *campaign_data* table in the client database, we isolated customers that received "Mailer 1" (low cost) and "Mailer 2" (high cost) for this campaign, and excluded customers who were in the control group.

We set our hypotheses and significance level for the test, as follows:

**Null Hypothesis:** There is no relationship between mailer type and sign-up rate. They are independent.  
**Alternative Hypothesis:** There is a relationship between mailer type and sign-up rate. They are not independent.  
**Significance Level:** 0.05

As a requirement of the Chi-Squared Test For Independence, we aggregated this data down to a 2x2 matrix for *signup_flag* by *mailer_type* and fed it into the algorithm (using the Python *scipy* library) to calculate the Chi-Squared statistic, p-value, degrees of freedom, and expected values.

<br>

### Results and Discussion <a name="overview-results"></a>

Based on the observed values, we get the following sign-up rates:

* Mailer 1 (Low Cost): **32.8%** sign-up rate
* Mailer 2 (High Cost): **37.8%** sign-up rate

The Chi-Squared Test gives us the following statistics:

* Chi-Squared Statistic: **1.94**
* p-value: **0.16**

The Critical Value for our specified Significance Level of 0.05 is **3.84**.

Based on these statistics, we fail to reject the null hypothesis, and conclude that there is no relationship between mailer type and sign-up rate.

In other words, while we saw that the higher cost Mailer 2 had a higher sign-up rate (37.8%) than the lower cost Mailer 1 (32.8%) it appears that this difference is not significant, at least at our Significance Level of 0.05.

Without running this Hypothesis Test, the client may have concluded that they should always go with higher cost mailers, or that the higher quality mailer is worth the investment of the higher cost. From what we've seen in this test, those may not be great decisions. It would result in them spending more, but not *necessarily* gaining any extra revenue as a result.

Our results here also do not say that there *definitely isn't a difference between the two mailers* --- we are only advising that we should not make any rigid conclusions *at this point*. 

Gathering more data and then re-running this test may provide us and the client more insight. Further A/B testing could also be performed on any new types of mailers the client wants to try out in the future.

<br>

___

# Concept Overview  <a name="concept-overview"></a>

#### A/B Testing

An A/B Test is a randomized experiment containing two groups, A and B, that receive different experiences. Within an A/B Test, we seek to understand and measure the response of each group. The information from this helps drive future business decisions.

Applications of A/B testing can range from comparing different online ad strategies, to different email subject lines when contacting customers, to testing the effect of mailing customers a coupon versus a control group. Large companies can run these tests in an almost never-ending cycle, testing new website features or different images on randomized groups of customers, which helps them figure out what works best so that they can stay ahead of their competition.

<br>

#### Hypothesis Testing

A Hypothesis Test is used to assess the likelihood of our assumed viewpoint based on available evidence (sample data).

There are many different scenarios we can run Hypothesis Tests on, and they all have slightly different techniques and formulas. However, they all have some shared, fundamental steps and logic that underpin how they work.

<br>

**The Null Hypothesis**

In any Hypothesis Test, we start with the Null Hypothesis. The Null Hypothesis is where we state our initial viewpoint. In statistics our initial viewpoint or Null Hypothesis is always that the result is purely by chance, i.e., that there is no relationship or association between two outcomes or groups.

<br>

**The Alternative Hypothesis**

The aim of the Hypothesis Test is to look for evidence to either support or reject the Null Hypothesis. If we reject the Null Hypothesis, that would mean we have found enough evidence to support the Alternative Hypothesis. The Alternative Hypothesis is essentially the opposite viewpoint to the Null Hypothesis --- that the result is *not* by chance, or that there *is* a relationship between two outcomes or groups. If we reject the Null Hypothesis, we do not accept the Alternative Hypothesis outright, but conclude the weight of the evidence supports the Alternative Hypothesis.

<br>

**The Significance Level and p-values**

A p-value is the probability of obtaining results as extreme as the observed data, assuming that the null hypothesis is correct.

In a Hypothesis Test, before collecting any data or running any numbers we decide on a Significance Level. This is a p-value threshold at which we’ll decide to reject or fail to reject the null hypothesis. We are essentially setting a limit, asking ourselves, "If I was to run this test many times over and over, what proportion of those times would I want to see different results come out in order to feel confident that my results are not just some unusual, random occurrence?"

Conventionally, we set our Significance Level to 0.05. If we need to be more confident that something did not occur through chance alone, we could lower this value down to something even smaller, meaning that we only come to the conclusion that the outcome was special or rare if the probability of getting the observed results by random chance is extremely small.

To summarize, in a Hypothesis Test we test the Null Hypothesis using a p-value and then make a decision about the Null Hypothesis based on the Significance Level.

<br>

**Types of Hypothesis Tests**

There are many different types of Hypothesis Tests, each of which is appropriate for use in different scenarios depending on a) the type of data that you’re testing, and b) the question that you’re asking of that data.

In the case of our task here, where we are looking to understand the difference in sign-up *rate* between two groups, we use the Chi-Squared Test For Independence.

<br>

#### Chi-Squared Test for Independence

The Chi-Squared Test For Independence is a type of Hypothesis Test that assumes observed frequencies for categorical variables (as opposed to numerical variables) will match the expected frequencies.

The *assumption* is the Null Hypothesis, which is the viewpoint that the two groups will have equal frequencies. With the Chi-Squared Test For Independence we look to calculate a statistic which we will use to either reject or support this initial assumption, based on the specified Significance Level.

The *observed frequencies* are the true values that we’ve seen.

The *expected frequencies* are essentially what we would *expect* to see based on all of the data.

**Note:** Another option when comparing rates is a test known as the *Z-Test For Proportions*. While we could use this test, it is preferable to use the Chi-Squared Test For Independence because:

* The resulting test statistic for both tests will be the same.
* The Chi-Squared Test can be represented using 2x2 tables of data, and can therefore be easier to explain to stakeholders.
* The Chi-Squared Test can extend out to more than 2 groups, unlike the Z-Test for Proportions, so the business can have one consistent approach to measuring significance.

___

# Data Overview and Preparation  <a name="data-overview"></a>

In the client database, we have a *campaign_data* table which shows us which customers received each type of Delivery Club mailer, which customers were in the control group, and which customers joined the club as a result.

For this task, we are looking to find evidence that the Delivery Club sign-up rate for customers that received "Mailer 1" (low cost) was different than the rate for those who received "Mailer 2" (high cost). Therefore, we will extract the entries for customers from those 2 groups from the *campaign_data* table, and we will exclude customers who were in the control group.

In the code below, we:

* Load in the Python libraries we require for importing the data and performing the chi-squared test (using scipy).
* Import the required data from the *campaign_data* table.
* Exclude customers in the control group, giving us a dataset with Mailer 1 & Mailer 2 customers only.

<br>

```python

# IMPORT REQUIRED PACKAGES
import pandas as pd
from scipy.stats import chi2_contingency, chi2

# IMPORT DATA
campaign_data = pd.read_excel('grocery_database.xlsx', sheet_name = 'campaign_data')

# FILTER OUR DATA TO REMOVE CUSTOMERS IN THE CONTROL GROUP
campaign_data = campaign_data.loc[campaign_data['mailer_type'] != 'Control']

```

<br>

A sample of this data (the first 10 rows) can be seen below:
<br>
<br>

| **customer_id** | **campaign_name** | **mailer_type** | **signup_flag** |
|---|---|---|---|
| 74 | delivery_club | Mailer1 | 1 |
| 524 | delivery_club | Mailer1 | 1 |
| 607 | delivery_club | Mailer2 | 1 |
| 343 | delivery_club | Mailer1 | 0 |
| 322 | delivery_club | Mailer2 | 1 |
| 115 | delivery_club | Mailer2 | 0 |
| 1 | delivery_club | Mailer2 | 1 |
| 120 | delivery_club | Mailer1 | 1 |
| 52 | delivery_club | Mailer1 | 1 |
| 405 | delivery_club | Mailer1 | 0 |
| 435 | delivery_club | Mailer2 | 0 |

<br>
In the DataFrame we have:

* customer_id
* campaign name
* mailer_type (either Mailer1 or Mailer2)
* signup_flag (either 1 or 0)

___

# Applying the Chi-Squared Test for Independence <a name="chi-squared-application"></a>

#### State Hypotheses and Significance Level for the Test

In the code below we code these in explicitly and clearly so we can use them later to explain the results. We specify the common Significance Level value of 0.05.

```python

# STATE HYPOTHESES & SET ACCEPTANCE CRITERIA
null_hypothesis = 'There is no relationship between mailer type and sign-up rate. They are independent.'
alternative_hypothesis = 'There is a relationship between mailer type and sign-up rate. They are dependent.'
significance_level = 0.05

```

<br>

#### Calculate Observed Frequencies and Expected Frequencies

As mentioned in the section above, in a Chi-Squared Test For Independence, the *observed frequencies* are the actual rates per group in the data itself. The *expected frequencies* are what we would *expect* to see based on *all* of the data combined.

The below code:

* Summarizes our dataset in a 2x2 matrix for *signup_flag* by *mailer_type*.
* Calculates the:
    * Chi-Squared Statistic
    * p-value
    * Degrees of Freedom
    * Expected Values
* Prints out the Chi-Squared Statistic and p-value from the test.
* Calculates the Critical Value based on the Significance Level and the degrees of freedom.
* Prints out the Critical Value.

```python

# SUMMARISE TO GET OUR OBSERVED FREQUENCIES
observed_values = pd.crosstab(campaign_data['mailer_type'], campaign_data['signup_flag']).values

# CALCULATE EXPECTED FREQUENCIES & CHI-SQUARED STATISTIC
chi2_stat, p_value, dof, expected_values = chi2_contingency(observed_values, correction = False)

# PRINT CHI-SQUARED STATISTIC
print(chi2_statistic)
>> 1.94

# PRINT P-VALUE
print(p_value)
>> 0.16

# FIND THE CRITICAL VALUE FOR OUR TEST
critical_value = chi2.ppf(1 - acceptance_criteria, dof)

# PRINT CRITICAL VALUE
print(critical_value)
>> 3.84

```

<br>

Based on the observed values, we get the following sign-up rates:

* Mailer 1 (Low Cost): **32.8%** sign-up rate
* Mailer 2 (High Cost): **37.8%** sign-up rate

From this, we can see that the higher cost mailer has a higher sign-up rate. The results from our Chi-Squared Test will provide us more information about how confident we can be that this difference is significant, or whether it could have occurred by chance.

We have a Chi-Squared Statistic of **1.94** and a p-value of **0.16**. The Critical Value for our specified Significance Level of 0.05 is **3.84**.

**Note** When applying the Chi-Squared Test above, we use the parameter *correction = False*, which means we are not applying the *Yates' Correction*, which is applied when your degrees of freedom is equal to one. The Yates Correction helps to prevent overestimation of statistical significance, but for simplicity it was not used here.

___

# Analyzing the Results <a name="chi-squared-results"></a>

At this point we have everything we need to understand the results of our Chi-Squared test. Since our resulting p-value of **0.16** is *greater* than our Significance Level of 0.05, we will *fail to reject* the Null Hypothesis and conclude that there is no significant difference between the sign-up rates for customers receiving Mailer 1 versus Mailer 2. (We can make the same conclusion based on the Chi-Squared statistic of **1.94** being *lower* than our Critical Value of **3.84**.)

___

# Discussion <a name="discussion"></a>

While the higher cost Mailer 2 had a higher sign-up rate (37.8%) than the lower cost Mailer 1 (32.8%) it appears that this difference is not significant, at least at our Significance Level of 0.05.

Without running this Hypothesis Test, the client may have concluded that they should always go with higher cost mailers, or that the higher quality mailer is worth the investment of the higher cost. From what we've seen in this test, those may not be great decisions. It would result in them spending more, but not *necessarily* gaining any extra revenue as a result.

Our results here also do not say that there *definitely isn't a difference between the two mailers* --- we are only advising that we should not make any rigid conclusions *at this point*. 

Gathering more data and then re-running this test may provide us and the client more insight. Further A/B testing could also be performed on any new types of mailers the client wants to try out in the future.
