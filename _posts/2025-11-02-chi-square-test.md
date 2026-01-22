---
layout: post
title: Assessing Campaign Performance Using Chi-Squared Test For Independence
image: "/img/posts/ab-testing-title-img.png"
tags: [AB Testing, Hypothesis Testing, Chi-Squared, Python]
---

In this project I apply the Chi-Squared Test For Independence (a hypothesis test) to assess the performance of two types of mailers that were sent out to promote a new service.

# Table of Contents

- [00. Data Source](#data-source)
- [01. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [02. Concept Overview](#concept-overview)
- [03. Data Overview & Preparation](#data-overview)
- [04. Applying Chi-Squared Test For Independence](#chi-squared-application)
- [05. Analysing The Results](#chi-squared-results)
- [06. Discussion](#discussion)

___

### Data Source <a name="data-source"></a>
Dataset provided as part of a data science training program. The data is designed to reflect a real-world business scenario.

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Earlier in the year, our client, a grocery retailer, ran an ad campaign to promote their new "Delivery Club" -- an initiative that costs $100/year for membership, but offers free grocery deliveries rather than the normal cost of $10 per delivery.

For the ad campaign, customers were randomly split into three groups. Customers in the first group received a low quality, low cost mailer; customers in the second group received a high quality, high cost mailer; and the third group was a control group, where customers received no mailer.

The client knows that customers who were contacted signed up for the Delivery Club at a far higher rate than the control group, but the client now wants to understand if there is a significant difference in signup rate between the cheap and expensive mailers. This will enable them to make more informed decisions in the future, with the overall goal of optimizing their campaign return on investment (ROI).

<br>

### Actions <a name="overview-actions"></a>

For this test, as it is focused on comparing the *rates* of two groups, we applied the Chi-Squared Test For Independence. Full details of this test can be found in the dedicated section below.

From the *campaign_data* table in the client database, we isolated customers that received "Mailer 1" (low cost) and "Mailer 2" (high cost) for this campaign, and excluded customers who were in the control group.

We set out our hypotheses and significance level for the test, as follows:

**Null Hypothesis:** There is no relationship between mailer type and signup rate. They are independent.  
**Alternative Hypothesis:** There is a relationship between mailer type and signup rate. They are not independent.  
**Significance Level:** 0.05

As a requirement of the Chi-Squared Test For Independence, we aggregated this data down to a 2x2 matrix for *signup_flag* by *mailer_type* and fed it into the algorithm (using the Python *scipy* library) to calculate the Chi-Squared statistic, p-value, degrees of freedom, and expected values.

<br>

### Results & Discussion <a name="overview-results"></a>

Based upon our observed values, we can give this all some context with the sign-up rate of each group. We get:

* Mailer 1 (Low Cost): **32.8%** signup rate
* Mailer 2 (High Cost): **37.8%** signup rate

However, the Chi-Squared Test gives us the following statistics:

* Chi-Squared Statistic: **1.94**
* p-value: **0.16**

The Critical Value for our specified Significance Level of 0.05 is **3.84**.

Based upon these statistics, we fail to reject the null hypothesis, and conclude that there is no relationship between mailer type and signup rate.

In other words - while we saw that the higher cost Mailer 2 had a higher signup rate (37.8%) than the lower cost Mailer 1 (32.8%) it appears that this difference is not significant, at least at our Significance Level of 0.05.

Without running this Hypothesis Test, the client may have concluded that they should always look to go with higher cost mailers - and from what we've seen in this test, that may not be a great decision. It would result in them spending more, but not *necessarily* gaining any extra revenue as a result.

Our results here also do not say that there *definitely isn't a difference between the two mailers* - we are only advising that we should not make any rigid conclusions *at this point*. 

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

There are many different scenarios we can run Hypothesis Tests on, and they all have slightly different techniques and formulas - however they all have some shared, fundamental steps and logic that underpin how they work.

<br>

**The Null Hypothesis**

In any Hypothesis Test, we start with the Null Hypothesis. The Null Hypothesis is where we state our initial viewpoint. In statistics our initial viewpoint or Null Hypothesis is always that the result is purely by chance, i.e., that there is no relationship or association between two outcomes or groups.

<br>

**The Alternative Hypothesis**

The aim of the Hypothesis Test is to look for evidence to either support or reject the Null Hypothesis. If we reject the Null Hypothesis, that would mean we have found enough evidence to support the Alternative Hypothesis. The Alternative Hypothesis is essentially the opposite viewpoint to the Null Hypothesis - that the result is *not* by chance, or that there *is* a relationship between two outcomes or groups. If we reject the Null Hypothesis, we do not accept the Alternative Hypothesis outright, but conclude the weight of the evidence supports the Alternative Hypothesis.

<br>

**The Significance Level**

In a Hypothesis Test, before we collect any data or run any numbers - we decide on a Significance Level. This is a p-value threshold at which we’ll decide to reject or fail to reject the null hypothesis. It is essentially a line we draw in the sand, saying, "If I was to run this test many many times, what proportion of those times would I want to see different results come out in order to feel confident that my results are not just some unusual occurrence?"

Conventionally, we set our Significance Level to 0.05. If we need to be more confident that something did not occur through chance alone, we could lower this value down to something much smaller, meaning that we only come to the conclusion that the outcome was special or rare if the probability of getting the observed results by random chance is extremely small.

To summarize, in a Hypothesis Test we test the Null Hypothesis using a p-value and then make a decision about the Null Hypothesis based on the Significance Level.

<br>

**Types Of Hypothesis Tests**

There are many different types of Hypothesis Tests, each of which is appropriate for use in different scenarios - depending on a) the type of data that you’re testing, and b) the question that you’re asking of that data.

In the case of our task here, where we are looking to understand the difference in sign-up *rate* between two groups, we use the Chi-Squared Test For Independence.

<br>

#### Chi-Squared Test For Independence

The Chi-Squared Test For Independence is a type of Hypothesis Test that assumes observed frequencies for categorical variables (as opposed to numerical variables) will match the expected frequencies.

The *assumption* is the Null Hypothesis, which is the viewpoint that the two groups will have equal frequencies. With the Chi-Squared Test For Independence we look to calculate a statistic which we will use to either reject or support this initial assumption, based on the specified Significance Level.

The *observed frequencies* are the true values that we’ve seen.

The *expected frequencies* are essentially what we would *expect* to see based on all of the data.

**Note:** Another option when comparing rates is a test known as the *Z-Test For Proportions*. While we could use this test, it is preferable to use the Chi-Squared Test For Independence because:

* The resulting test statistic for both tests will be the same.
* The Chi-Squared Test can be represented using 2x2 tables of data, and can therefore be easier to explain to stakeholders.
* The Chi-Squared Test can extend out to more than 2 groups, unlike the Z-Test for Proportions, so the business can have one consistent approach to measuring significance.

___

# Data Overview & Preparation  <a name="data-overview"></a>

In the client database, we have a *campaign_data* table which shows us which customers received each type of "Delivery Club" mailer, which customers were in the control group, and which customers joined the club as a result.

For this task, we are looking to find evidence that the Delivery Club signup rate for customers that received "Mailer 1" (low cost) was different than the rate for those who received "Mailer 2" (high cost). Therefore, we will extract the entries for customers from those 2 groups from the *campaign_data* table, and we will exclude customers who were in the control group.

In the code below, we:

* Load in the Python libraries we require for importing the data and performing the chi-squared test (using scipy).
* Import the required data from the *campaign_data* table.
* Exclude customers in the control group, giving us a dataset with Mailer 1 & Mailer 2 customers only.

<br>

```python

# install the required python libraries
import pandas as pd
from scipy.stats import chi2_contingency, chi2

# import campaign data
campaign_data = ...

# remove customers who were in the control group
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]

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

# Applying the Chi-Squared Test For Independence <a name="chi-squared-application"></a>

<br>

#### State Hypotheses and Significance Level for the Test

In the code below we code these in explicitly and clearly so we can use them later to explain the results. We specify the common Significance Level value of 0.05.

```python

# specify hypotheses & significance level for test
null_hypothesis = "There is no relationship between mailer type and signup rate. They are independent."
alternative_hypothesis = "There is a relationship between mailer type and signup rate. They are not independent."
significance_level = 0.05

```

<br>

#### Calculate Observed Frequencies & Expected Frequencies

As mentioned in the section above, in a Chi-Squared Test For Independence, the *observed frequencies* are the actual rates per group in the data itself. The *expected frequencies* are what we would *expect* to see based on *all* of the data combined.

The below code:

* Summarizes our dataset in a 2x2 matrix for *signup_flag* by *mailer_type*.
* Based on this, calculates the:
    * Chi-Squared Statistic
    * p-value
    * Degrees of Freedom
    * Expected Values
* Prints out the Chi-Squared Statistic and p-value from the test.
* Calculates the Critical Value based upon our Significance Level & the degrees of freedom
* Prints out the Critical Value.

```python

# aggregate our data to get observed values
observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values

# run the chi-squared test
chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction = False)

# print chi-squared statistic
print(chi2_statistic)
>> 1.94

# print p-value
print(p_value)
>> 0.16

# find the critical value for our test
critical_value = chi2.ppf(1 - acceptance_criteria, dof)

# print critical value
print(critical_value)
>> 3.84

```

<br>

Based upon our observed values, we can give this all some context with the sign-up rate of each group. We get:

* Mailer 1 (Low Cost): **32.8%** signup rate
* Mailer 2 (High Cost): **37.8%** signup rate

From this, we can see that the higher cost mailer does lead to a higher signup rate. The results from our Chi-Squared Test will provide us more information about how confident we can be that this difference is significant, or if it could have occurred by chance.

We have a Chi-Squared Statistic of **1.94** and a p-value of **0.16**. The critical value for our specified Significance Level of 0.05 is **3.84**.

**Note** When applying the Chi-Squared Test above, we use the parameter *correction = False* which means we are not applying what is known as the *Yates' Correction* which is applied when your Degrees of Freedom is equal to one. This correction helps to prevent overestimation of statistical signficance in this case.

___

# Analysing The Results <a name="chi-squared-results"></a>

At this point we have everything we need to understand the results of our Chi-Squared test - and just from the results above we can see that, since our resulting p-value of **0.16** is *greater* than our Significance Level of 0.05 then we will _retain_ the Null Hypothesis and conclude that there is no significant difference between the signup rates of Mailer 1 and Mailer 2.

We can make the same conclusion based upon our resulting Chi-Squared statistic of **1.94** being _lower_ than our Critical Value of **3.84**

To make this script more dynamic, we can create code to automatically interpret the results and explain the outcome to us...

```python

# print the results (based upon p-value)
if p_value <= acceptance_criteria:
    print(f"As our p-value of {p_value} is lower than our acceptance_criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that: {alternative_hypothesis}")
else:
    print(f"As our p-value of {p_value} is higher than our acceptance_criteria of {acceptance_criteria} - we retain the null hypothesis, and conclude that: {null_hypothesis}")

>> As our p-value of 0.16351 is higher than our acceptance_criteria of 0.05 - we retain the null hypothesis, and conclude that: There is no relationship between mailer type and signup rate. They are independent


# print the results (based upon p-value)
if chi2_statistic >= critical_value:
    print(f"As our chi-squared statistic of {chi2_statistic} is higher than our critical value of {critical_value} - we reject the null hypothesis, and conclude that: {alternative_hypothesis}")
else:
    print(f"As our chi-squared statistic of {chi2_statistic} is lower than our critical value of {critical_value} - we retain the null hypothesis, and conclude that: {null_hypothesis}")
    
>> As our chi-squared statistic of 1.9414 is lower than our critical value of 3.841458820694124 - we retain the null hypothesis, and conclude that: There is no relationship between mailer type and signup rate. They are independent

```

<br>

As we can see from the outputs of these print statements, we do indeed retain the null hypothesis. We could not find enough evidence that the signup rates for Mailer 1 and Mailer 2 were different - and thus conclude that there was no significant difference.

___

# Discussion <a name="discussion"></a>

While the higher cost Mailer 2 had a higher signup rate (37.8%) than the lower cost Mailer 1 (32.8%) it appears that this difference is not significant, at least at our Significance Level of 0.05.

Without running this Hypothesis Test, the client may have concluded that they should always look to go with higher cost mailers. From what we've seen in this test, that may not be a great decision. It would result in them spending more, but not *necessarily* gaining any extra revenue as a result

Our results here also do not say that there *definitely isn't a difference between the two mailers* -- we are only advising that we should not make any rigid conclusions *at this point*. 

Running more A/B Tests like this, gathering more data, and then re-running this test may provide us, and the client more insight!
