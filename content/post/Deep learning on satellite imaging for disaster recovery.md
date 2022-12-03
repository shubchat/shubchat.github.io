+++
title = "Deep learning-Satellite imaging"
description = "Using Deep learning to identify natural disasters from satellite images"
date = 2014-09-28T02:13:50Z
author = "Shubh Chatterjee"
+++





# Using Deep learning to identify natural disasters from satellite images

The idea that lead to this mini-project was how effective will CNN be to identify natural disasters when fed satellite images.An accurate system which is capable of tagging the type of disaster using close to real time satellite images will lead to better response and relief to the impacted areas.

**Data used:**RGB Satellite images of various resolutions across multiple disaster types from NASA earth observatory were used as training and validation sample.

![Image1-RGB X 2967 X 2967 Satellite image of a wild fire(Source-NASA Earth observatory)](https://cdn-images-1.medium.com/max/2000/1*NXkPtf4HBQduwpffucF_fg.png)

**Data extraction-**For extracting the images from NASA earth observatory I used the really cool eonet API ([https://eonet.sci.gsfc.nasa.gov/api/v2](https://eonet.sci.gsfc.nasa.gov/api/v2).1/events).The API allows you to pull the required images url using GET request.The images are tagged as per the natural disaster type(Example-Fire,drought,storms etc).I personally used requests in python to extract the required urls and their corresponding labels.To extract the images I used a mix of url2,xml.html & requests packages.

**Below code should be able to download images to your local from a csv of links**

    import csv
    import urllib
    import lxml.html
    import requests

    connection = urllib.urlopen(url)

    with open('/required_url.csv') as csvfile:
        csvrows = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvrows:
          if 'view.php' in row[0]:
            filename = row[1]
            url = row[0]
            locn=row[2]
            print (locn)

One of the important points I wanted to study was the impact a [Deep learning](https://hackernoon.com/tagged/deep-learning) model can create when there is lack of data.To make sure my data sample is not image rich I used only 80 images.To keep the task simple I decided not to move into multi class classification and limited the labels to two classes only(Storm_wildfires /others).The end aim for my model was to be able to identify if in an image there is either a storm or a wildfire(every other natural disaster it should classify as others).

The [machine learning](https://hackernoon.com/tagged/machine-learning)/data science/deep learning community comprises of some of the most amazingly smart individuals who are developing cutting edge tools and algorithms which will change the world as we know it.They are also really kind enough to share these with rest of us so that we don’t need to reinvent the wheel.I used transfer learning facilitated by fastai library developed by the team @fastai . Most of the methodologies that were used in this experiment were inspired by the Deep learning part-1 course.I used resnet32 pre-trained on 1000 imagenet classes for the below experiment

### Steps followed:

1)Image resizing

2)Identifying the optimum learning rate

3)Training the output layer while keeping rest of the weights of initial layers fixed

4)Retraining the complete model

5)Analyzing the results

1. **Image resizing**

As I mentioned before most of the images were of different resolutions.I wanted to make sure that we had a uniformity across the image resolutions so that GPU computing could be optimum.Below is an example of how the images looked post resizing:

![](https://cdn-images-1.medium.com/max/2000/1*NXkPtf4HBQduwpffucF_fg.png)

**2)Identifying the optimum learning rate**

One of the most amazing tool that was showcased by Jeremy in DL-1 course is the learning rate finder.Finding the perfect learning rate is always a big pain for ML practitioners across the methodologies(RF,Gradient boosting,DL).Using the learning rate finder below is how the graph looked like.As the training data sample is quite low for me I decided to train the model using a learning rate of 0.001 to make sure the model trains slowly without jumping to conclusions.

**3)Training the output layer while keeping rest of the weights of initial layers fixed**

This step pushed me towards an accuracy of ~65%.Which when compared to one of the papers I was referring to looked quite bad.

![Transfer Learning from Deep Features for Remote Sensing and Poverty Mapping(Michael Xie and Neal Jean and Marshall Burke and David Lobell and Stefano Ermon)](https://cdn-images-1.medium.com/max/2000/1*pF5crSTj6I7Y-0RGPlnLOA.png)*Transfer Learning from Deep Features for Remote Sensing and Poverty Mapping(Michael Xie and Neal Jean and Marshall Burke and David Lobell and Stefano Ermon)*

**4)Retraining the complete model**

Now the resnet34 is pre-trained using 1000 class objects of imagenet.These 1000 objects are not exactly suitable to be used for transfer learning in the case of satellite images .Hence,I retrained all the layers of imagenet for our use case.The results were quite impressive as i now moved to an **accuracy of ~75%**.One of things that I observed that the change in accuracy across the epoch is not that stable which could be due to low training sample <100 due to which the function that the model was trying to fit to is not exactly stable which may lead to fall in performance when I will move out of sample.

**5)Analyzing the results**

Let look how the model is predicting:

**A few correct images at random**

![model correctly predicting storm and wildfires](https://cdn-images-1.medium.com/max/2000/1*HcDLLZEloW2zY35EBBXSRA.png)*model correctly predicting storm and wildfires*

**A few incorrect images at random**

![Model not able to differentiate between other natural disasters and storm and wildfires](https://cdn-images-1.medium.com/max/2000/1*fpj2_HcxflkyNcHtTs26qQ.png)*Model not able to differentiate between other natural disasters and storm and wildfires*

Model not able to differentiate between other natural disasters for example snow and what we are predicting for- that is storm and wildfires(if you think it can be tricky even for a human eye as both disasters pertain to searching for white substance (sleet in case of snow and smoke in case of fire).This may confuse any man or machine.The solution to that could be more training data which would give the machine more data points to drive correct conclusions.

**Most correct Storms and wildfires**

![](https://cdn-images-1.medium.com/max/2000/1*ZXeDlE4GpWMdvCAP2yk0mQ.png)

**Most correct Others (that is not a storm or wildfire)**

![](https://cdn-images-1.medium.com/max/2000/1*U0jlPkZ-jTYQTfxsV7YzsA.png)

**Most incorrect storm or wildfires**

![](https://cdn-images-1.medium.com/max/2000/1*pYWglCstIV7iqp9GWCv74A.png)

**Most incorrect others(that is not a storm or wildfire)**

![](https://cdn-images-1.medium.com/max/2000/1*-Rba6PB3l1iQ7OYF2fqYmw.png)

## Conclusion

Although our deep learning model to predict which of the natural disasters is a storm or wildfire comes up with an accuracy ~75% which looks good (when compared to the academic literature I was following), the model seems to be unstable .Due to the confusing nature of different disasters images(where a snow storm may look similar to a wild fire) even a human eye may be confused.May be using more training points(I used only ~80 images in this experiment should be more fruitful.

## **Next steps**

1. Source more satellite images of natural disasters

2. Try training on the larger sample size

3. Further simplify the problem statement (may be try to predict snow storm from wild fire) rather than having a mix bag in the other class


