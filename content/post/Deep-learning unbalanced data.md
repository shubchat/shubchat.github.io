+++
title = "Deep learning unbalanced training data"
description = "Deep learning unbalanced training data"
date = 2019-06-24T02:13:50Z
author = "Shubh Chatterjee"
+++




# Deep learning unbalanced training data?Solve it like this.



One of the biggest problems that we face when we tackle any machine learning problem is the problem of unbalanced training data.The problem of unbalanced data is such that the academia is split with respect to the definition, implication & possible solutions for the same.We will here try to unravel the mystery of unbalanced classes in the training data using an image classification problem.

## **What is the problem of unbalanced classes?**

In a classification problem when out of all the classes which you want to predict if for one or more classes there are extremely low number of samples you may be facing a problem of unbalanced classes in your data.

**Examples**

1. Fraud prediction(Number of frauds will be much lower that genuine transactions)

1. Natural disaster predictions(Bad events will be much lower than good)

1. Identifying malignant tumor in an image classification(images with a tumor will be much less than that of no tumor within a training sample)

**Why is this a problem?**

The unbalanced classes create a problem due to two main reasons:

1. We don’t get optimized results for the class which is unbalanced in real time as the model/algorithm never gets sufficient look at the underlying class

1. It creates a problem of making a validation or test sample as its difficult to have representation across classes in case number of observation for few classes is extremely less

**What are the different approaches followed to solve this?**

There are three main approaches which are suggested each having its pros and cons:

1. **Undersampling**- Randomly delete the class which has sufficient observations so that the comparative ratio of two classes is significant in our data.Although this approach is really simple to follow but there is a high possibility that the data that we are deleting may contain important information about the predictive class.

1. **Oversampling-**For the unbalanced class randomly increase the number of observations which are just copies of existing samples.This ideally gives us sufficient number of samples to play with.The oversampling may lead to overfitting to the training data

1. **Synthetic sampling(SMOTE)-**The technique asks to synthetically manufacture observations of unbalanced classes which are similar to the existing using nearest neighbors classification.**The problem is what to do when the number of observations of is an extremely rare class .For example-we may have only one picture of a rare species which we want to identify using image classification algorithm**

Although each of the approaches have their own benefits there is no particular heuristic of which technique to use when.We will now look into this problem in detail using a **deep learning specific image classification problem**.

## Unbalanced classes in Image classification

In this section we will pick up a problem of image classification which has an issue of unbalanced classes and then we will solve it using a simple and effective technique.

**The problem**-We picked up **“Humpback Whale Identification Challenge” **on kaggle which we expected to have a challenge of solving for unbalanced classes(as ideally the number of whales classified will be less than non-classified also there will be few rare whale species for which we will have less number of images)

From kaggle:**“In this competition, you’re challenged to build an algorithm to identifying whale species in images. You’ll analyze Happy Whale’s database of over 25,000 images, gathered from research institutions and public contributors. By contributing, you’ll help to open rich fields of understanding for marine mammal population dynamics around the globe.”**

## Lets start looking at the data

As this is a multi-label image classification problem I first wanted to check how is the data distributed across the classes.

![](https://cdn-images-1.medium.com/max/2000/1*iYujQD4FOHkrSBdsNFuvpw.png)

The above chart dictates that **out of 4251 training images more than 2000 have only one image per class**.There are also classes with ~2–5 images .Now, this is a serious problem of unbalanced classes.We can’t expect a DL model to train using just one image per class(Although there are algorithms that may just do that for example one shot classification but we are as of now ignoring that).This also creates a problem how to create a split between training and validation sample.You will ideally want each of the classes to be represented in both training and validation sample.

## What should we do now?

There are two options in particular that we considered:

**option1**-Rigorous data augmentation on the training sample(We could have done that but as we need data augmentation only for specific classes this may not solve our purpose completely).Hence I went for option-2 which looked simple enough to try .

**option2-**Similar to the oversampling option that I mentioned above.I just copied the images of unbalanced classes back into the training data 15 times using different image augmented techniques.This is inspired from [Jeremy Howard](undefined) who I guess mentioned this in one of the deep learning lectures of part-1 [*fast.ai course](http://www.fast.ai/).*

Before we start with option-2 lets view few images from the training sample.

![](https://cdn-images-1.medium.com/max/2000/1*HUKhePnuVA0sbPR6X6XqMw.png)

The images are specific to the fluke of the whales .Hence, identification will be probably quite specific to the way images will be oriented.

I also noticed there are lots of images in the data which are specific B&W or only of R/B/G channel.

Based on these observations I decided to write the below code to do small changes in images which are from unbalanced classes in training sample ans save them:

    import os
    from PIL import Image

    from PIL import ImageFilter

    filelist = train['Image'].loc[(train['cnt_freq']<10)].tolist()

    for count in range(0,2):
      
      for imagefile in filelist:
        os.chdir('/home/paperspace/fastai/courses/dl1/data/humpback/train')
        im=Image.open(imagefile)
        im=im.convert("RGB")
        r,g,b=im.split()
        r=r.convert("RGB")
        g=g.convert("RGB")
        b=b.convert("RGB")
        im_blur=im.filter(ImageFilter.GaussianBlur)
        im_unsharp=im.filter(ImageFilter.UnsharpMask)
        
        os.chdir('/home/paperspace/fastai/courses/dl1/data/humpback/copy')
        r.save(str(count)+'r_'+imagefile)
        g.save(str(count)+'g_'+imagefile)
        b.save(str(count)+'b_'+imagefile)
        im_blur.save(str(count)+'bl_'+imagefile)
        im_unsharp.save(str(count)+'un_'+imagefile)

The above code block does the following to each of the images in unbalanced class(which have frequency less than 10):

1. Save augmented copy of each of the images each as R/B& G

1. Save augmented copy of each image which is blury

1. Save augmented copy of each image which in unsharp

We used pillow (a python image library)rigorously for this exercise as can be seen in the above code

Now we have for all of the unbalanced classes at-least 10 samples.We proceeded with the training .

**Image augmentation-**We kept this simple.We only wanted to make sure that our model is able to get a detailed view of the fluke of the whale.For this we incorporated zoom into image augmentation.

![](https://cdn-images-1.medium.com/max/2000/1*EbptStu_tLG_IlgbBy3C5Q.png)

**Learning rate finder-**We decided upon a learning rate of 0.01 as identified as lr find.

![](https://cdn-images-1.medium.com/max/2000/1*q5_4SxgGXIm5cF2lt__Psg.png)

We ran few iteration using Resnet50 (first frozen and unfrozen).Turns out the frozen model is also quite good for this problem statement as there are images of whale flukes in imagenet.

    epoch      trn_loss   val_loss   accuracy                     
        0      1.827677   0.492113   0.895976  
        1      0.93804    0.188566   0.964128                      
        2      0.844708   0.175866   0.967555                      
        3      0.571255   0.126632   0.977614                      
        4      0.458565   0.116253   0.979991                      
        5      0.410907   0.113607   0.980544                      
        6      0.42319    0.109893   0.981097

**How does this look on test data?**

Finally our moment of truth on kaggle leader board.The solution proposed ranks 34 in this competition with a Mean Average Precision @ 5 of 0.41928 :)

![](https://cdn-images-1.medium.com/max/2350/1*OcC_cv9XD1_a00VP5jQN0Q.png)

**Conclusion**

Sometimes the simple approaches which are most logical (If you don’t have more data just copy existing data again with slight variation pretend most of the class observations will be on same line for the model) are the ones most effective and can get the work done more easily and intuitively.
