+++
title = "Why ML metrics matter "
description = "Why ML metrics matter"
date = 2021-09-01T02:13:50Z
author = "Shubh Chatterjee"
+++




# Why ML metrics matter 

## Introduction

In the world of data science we spend a huge amount of our focus and energy on developing state of the art prediction algorithm. These highly sophisticated algorithms are tuned to enhance some statistical metric which we might reckon might be relevant for the problem. In essence the mathematical metric is what decides what we are measuring and how we reckon the perfect solution for our problem might look like. 
Choosing a metric is like deciding on what really matters for this problem in the current scenario . Choosing a wrong metric is like wasting your valuable focus and energy in search of a solution which will lead to no real world gains

![Metrics](/images/AUC_vs_PR/img/workflow_ML_using_metrics.png 'ML metrics')

<!-- ![How mathematical metrics play a part in optimization](/media/AUC_vs_PR/img/workflow_ML_using_metrics.png) -->

*How mathematical metrics play a part in optimization*

In this blog we will be focusing on classification problems and will be covering:

- What are some different types of classification problems which might warrant different optimisation metrics?
- What are some of the metrics widely used to mathematically solve ML-based classification problems?
- What are some thumb rules around which metrics to use for which kind of problem?


## What are some different types of classification problems which might warrant different optimisation metrics?

Any problem where you might want to classify an entity or event into different buckets/categories using data points about them is a classification problem. Examples:
- Will an applicant for loan default or not if we offer loan ?
- Should I be targeting this existing user for another product from our product catalogue?
- Are there any medical abnormalities in this FMRI scan?
- Is this payment from a stolen credit card?
- Which of these comments on the blog is toxic?
- What type of humpback whale is in this image?

Now although all of the problems above are classification problems at their core,they can be categorised as per different traits that we might observe .One of the those is what is the ratio of different classes within the problem .for example :
- In a problem where we might be attempting to predict the credit quality of an applicant while applying for a loan ,the defaulter ratio might be lets say<b> 20% </b>
- A medical imaging problem where we might be attempting to identify cancerous cells in an image .The event rate found in the population might just be <b>0.5%</b>

Both are classification problems but while one of the cases has 20% event rate while other has a much lower 0.5% event rate .Now this is not a data capture  specific trait but rather population heuristic rates or the base rates that we are mentioning here. This changes tons of stuff on how we might change our workflow depending upon whether we are dealing with an imbalanced class based problem or not .Some of the changes we might have to make are:

-  Data capture for ML or any specific analysis 
- Methodology used for model development
- <b>Which metrics should we be using to evaluate our classification models</b>

We in this text will focus on the third point mentioned above ,now although we would like to believe that goodness of fit metrics for evaluating our decision models can be universal ,this is far from the truth. Our choice might impact the applied usage of the models in production as we might pick up sub-optimized model by optimizing towards a substandard  metric for the use case that we are working with. Below is an example use case which will clarify this for you where we are using accuracy as a metric across two problems with different base event rates.

Accuracy : Out of all data points that we have in the existing evaluation set in what percentage of cases our model classification was correct .For example if out of 100 data points if our model was able to correctly classify 90 data points the accuracy is 90% 

**Table:1**
Problem       | Base rate    | Accuracy of a model
------------- | -------------| -------
Identifying loan defaults  | 20% | 91%
Identifying abnormalities in FMRI scans  | 0.5% | 91%

- Case1:(Base rate 20%) In this case a random model that predicts a no-event in each case should have an accuracy of 80% and Hence our model with an accuracy of 91% should enhance our classification quality over coin toss
- Case2:(Base rate 0.5%) With such a low base rate a random model can have an accuracy ~ 99.5% .Hence,although accuracy of our model is 91% which seems quite high ,the actual pragmatic usability of the model is quite bad . 

Hopefully above example helped bring in the context around how a single metric might behave or provide a different context based on if we are working on an imbalanced class problem.


## What are some of the metrics widely used to mathematically solve ML based classification problems?

Alright,lets now take a classification problem and learn about some of the metrics that might be useful and what do each of them inform about the quality of our prediction.

Problem: A classifier predicts out of a sample of 1000 patients ,how many might develop cancer in the next 12 months .Below is how it performed.

**Table:2**
Sample  | predicted to have disease|Actually had disease|Accuracy|TP|FP|Precision|Recall|
------- | ------------------------ |------------------- |--------|---|---|---------|------|
1000    |20                        | 5                 |85%     | 5  | 15   | 25%   | 100%


- Accuracy : Out of all data points that we have in the existing evaluation set in what percentage of cases our model classification was correct
- FPR(False positive rate) : Out of all positive predictions how many are actually negative(in this case 15/(20)=75%)
- Precision :How precise are our predictions out of all cases where we predict a positive how many are actually positive (5/20=25%)
- Recall : Out of all positives ,how many are we able to capture?(5/5=100%)
- log_loss : Logarthmic loss(indicates how close the prediction probability is to the true value)
- Mean average precesion at K: How precise is our algorithm if we only pick the top K predictions for all cases
- Top k accuracy : How accurate are we if we pick the first K predictions
- Calibration chart
- Any many more....

## What are some thumbrules around which metrics to use for which kind of problem?

We now do realize that when attempting to solve a classification problem we have a gamut of metrics that we can optimise but that does not mean that each metric is relevant for each and every use case. Depending upon various aspects of a problem we might want one or more metric to judge a specific aspect of problem solving. Some of the different aspects of a problem we might consider:<br> 

- The base event rate of the problem(Are we dealing with an imbalanced class problem? )
- Cost of a false positive vs a false negative(What will be more impactful in the grand scheme of things?)
- Do we care only about the rank-order(if the model is able to identify one class from another ) or do we also care about how calibrated our results are(predicted probablity vs actual result)?
-  A derivative of previous point,are the Top K( where K is # of predictions ranked by probablity) predictions most important of we care of the entire gamut of predictions
- How unstable is your problem set ( Can we expect a distribution change of event rate in the recent future?)


For any classification problem that we are trying to solve one or more of the above aspects might be considered, then based on which we will pick metrics which help us address that aspect of the problem. Let's learn more by picking the same problem we were addressing in the last section<br>
>Problem: A classifier predicts out of a sample of 1000 patients ,how many might develop cancer in the next 12 months <br>

Sample  | predicted to have disease|Actually had disease|Accuracy|TP|FP|Precision|Recall|
------- | ------------------------ |------------------- |--------|---|---|---------|------|
1000    |20                        | 5                 |85%     | 5  | 15   | 25%   | 100%


Before even looking at the numbers lets think deeper about the problem. In this problem : <br>

- The base event rate is (0.5%) which means we are dealing with an unbalanced class problem
- We are predicting if someone might have cancer, this prediction might be used to take preventive measures which might save someone lives. What are the costs of wrong predictions

    - A False negative means we might not be able to save a life from cancer
    - A False positive means we might make a person who is not a prospective cancer risk take preventative treatment.Impact: Side-effects, wasted resources 

    Considering both of the above options I would rather err on the side of a False negative than a False positive(but not by much)<br>
    
- We need our classifier to have good calibration that is for all cases where lets say prob of cancer is 70% and above the real event rate should be ball park be around the same
- Lets say the Top5 patients with the highest probablity of cancer will receive the most strongest treatment and we want to extremely accurate here as strong treatment means higher chances of side-effects and you don't want a non-cancerous patient to suffer that
- I will say our problem is relatively stable and any drift or change in distribution does not happen fast unless you change the demographics (Just guessing)

Now based on the above we will be defining the metrics we will use to test our cancer classifier.<br>

**To test for overall model discrimination** considering we are dealing with an unbalanced class problem we will use a precesion/recall chat instead of AUC chart to learn how our model is, now why is a PR curve a better idea when we have an unbalanced class problem? Lets assume we developed a Logistic regression model for the problem we are trying to solve and below is how its performing:<br> Going back to the problem and the table2<br>

![Metrics](/images/AUC_vs_PR/img/AUC_PR.png 'AUC vs PR curve')



## Final pointers



