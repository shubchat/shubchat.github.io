+++
title = "Why ML metrics matter "
description = "Why ML metrics matter"
date = 2021-09-01T02:13:50Z
author = "Shubh Chatterjee"
+++




# Why ML metrics matter 

## Introduction

In the data science world while we spend a huge amount of our focus and energy on developing state of the art prediction algorithm while trying to optimize some statistical metric not much thought goes into identifying which is a proper metric to be used for a specific problem .

Choosing a metric is like deciding on what really matters for this problem in the current scenario .If you go wrong on this choice you will be optimizing towards a wrong solution .

![Metrics](/images/AUC_vs_PR/img/workflow_ML_using_metrics.png 'ML metrics')

<!-- ![How mathematical metrics play a part in optimization](/media/AUC_vs_PR/img/workflow_ML_using_metrics.png) -->

*How mathematical metrics play a part in optimization*

In this blog we will be focusing on classification problems and will be covering:

- What are some different types of classification problems which might warrant different optimisation metrics?
- What are some of the metrics widely used to mathematically solve ML-based classification problems?
- What are some thumb rules around which metrics to use for which kind of problem?


## What are some different types of classification problems which might warrant different optimisation metrics?

Any problem where you would want to classify an entity or event into different buckets/categories using data points about them is a classification problem.few of the examples of questions which fall in a ML classification problem:
- Will an applicant for loan default or not if we offer loan ?
- Should I be targeting this existing user for another product from our product catalogue?
- Are there any medical abnormalities in this FMRI scan?
- Is this payment from a stolen credit card?
- Which of these comments on the blog is toxic?
- What type of humpback whale is in this image?

Now although all of the problems above are classification problems at their core,they can be categorised as per different traits that we might observe .One of the those is what is the ratio of different classes within the problem .for example :
- In a problem where we might be attempting the credit quality of an applicant while applying for a loan ,the defaulter ratio might be lets say<b> 20% </b>
- A medical imaging problem where we might be attempting to identify cancerous cells in an image .The event rate found in the population might just be <b>0.5%</b>

Both are classification problems but while one of the cases has 20% event rate while other has a much lower 0.5% event rate .Now this is not a data capture  specific trait but rather population heuristic rates or the base rates that we are mentioning here. This changes tons of stuff on how we might change our workflow depending upon whether we are dealing with an imbalanced class based problem or not .Some of the changes we might have to make are:

-  Data capture for ML or any specific analysis 
- Methodology used for model development
- <b>Which metrics should we be using to evaluate our classification models</b>

We in this text will focus on the third point mentioned above ,now although we would like to believe that goodness of fit metrics for evaluating our decision models can be universal ,this is far from the truth. This might impact the applied usage of the models in production as we might pick up sub-optimized models due to us optimizing towards a substandard  metric for the use case that we are working with. Below is an example use case which will clarify this for you where we are using accuracy as a metric across two problems with different base event rates.

Accuracy : Out of all data points that we have in the existing evaluation set in what percentage of cases our model classification was correct .For example if out of 100 data points if our model was able to correctly classify 90 data points the accuracy is 90% 


Problem       | Base rate    | Accuracy of a model
------------- | -------------| -------
Identifying loan defaults  | 20% | 91%
Identifying abnormalities in FMRI scans  | 0.5% | 91%


- Case1:(Base rate 20%) In this case a random model that predicts a no-event in each case should have an accuracy of 80% and Hence our model with an accuracy of 91% should enhance our classification quality over coin toss
- Case2:(Base rate 0.5%) With such a low base rate a random model can have an accuracy ~ 99.5% .Hence,although accuracy of our model is 91% which seems quite high ,the actual pragmatic usability of the model is quite bad . 

Hopefully above example helped bring in the context around how a single metric might behave or provide a different context based on if we are working on an imbalanced class problem.


## What are some of the metrics widely used to mathematically solve ML based classification problems?

Alright,lets now take a classification problem and learn about some of the metrics that might be useful and what do each of them inform about the quality of our prediction.

Problem: A classifier predicts out of a sample of 1000 patients ,how many might develop a disease in the next 12 months .Below is how it performed.

Sample  | predicted to have disease|Actually had disease|Accuracy|TP|FP|Precision|Recall|
------- | ------------------------ |------------------- |--------|---|---|---------|------|
1000    |20                        | 5                 |85%     | 5  | 15   | 25%   | 100%



- Accuracy : Out of all data points that we have in the existing evaluation set in what percentage of cases our model classification was correct
- FPR(False positive rate) : Out of all positive predictions how many are actually negative(in this case 15/(20)=75%)
- Precision :How precise are our predictions out of all cases where we predict a positive how many are actually positive (5/20=25%)
- Recall : Out of all positives ,how many are we able to capture?(5/5=100%)

<b>Coming up soon..</b> 

## What are some thumbrules around which metrics to use for which kind of problem?

## Final pointers



