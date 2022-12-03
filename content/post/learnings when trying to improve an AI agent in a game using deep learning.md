+++
title = "Improve an AI agent in a game using deep learning"
description = "Improve an AI agent in a game using deep learning"
date = 2019-06-24T02:13:50Z
author = "Shubh Chatterjee"
+++


# What I learned when trying to improve an AI agent in a game using deep learning

Late 2018 I participated in kaggle’s “Quick, Draw! Doodle Recognition Challenge”.For those of you who are unaware, below is a short description of this game:

[*“Quick, Draw!”](https://quickdraw.withgoogle.com/) was released as an experimental game to educate the public in a playful way about how AI works. The game prompts users to draw an image depicting a certain category, such as ”banana,” “table,” etc.*

![](https://cdn-images-1.medium.com/max/2000/1*Ha5GdpPPDfZYwXqPx2jl_A.png)

As part of this competition, a subset of more than 1B drawings was released which had 340 labels. The competitors needed to improve the existing AI [algorithm](https://hackernoon.com/tagged/algorithm) which distinguishes whether a user has correctly been able to draw what was asked for. For each test image, the need was to predict the three most probable classes the doodle might belong to.

    key_id,word
    9000003627287624,The_Eiffel_Tower airplane donut
    9000010688666847,The_Eiffel_Tower airplane donut

The finest algorithm was chosen based on its Mean Average Precision @ 3 (MAP@3).

![](https://cdn-images-1.medium.com/max/2000/1*I85rP6mrAKPkcJjD7w50jg.png)

where U is the number of scored drawings in the test data, P(k) is the precision at cutoff kand n is the number of predictions per drawing.

## **Initial deep dive into the data**

The drawings were captured as timestamped vectors, tagged with metadata including what the player was asked to draw and in which country the player was located. Each of the 340 classes had CSV files in the below format defining how each of the doodles was drawn by the corresponding player.

![A sample for doodles with apple as class](https://cdn-images-1.medium.com/max/2000/1*ngUNgWapNHSbvg4JFdCwbA.png)*A sample for doodles with apple as class*

Using the below code we can convert each of the strokes in the drawing column of the above file into a corresponding image.

    BASE_SIZE = 256
    **def** draw_cv2(raw_strokes, size=299, lw=4, time_color=**False**):
        img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
        **for** t, stroke **in** enumerate(raw_strokes):
            **for** i **in** range(len(stroke[0]) - 1):
                color = 255 - min(t, 10) * 13 **if** time_color **else** 255
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                             (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
        img = cv2.copyMakeBorder(img,4,4,4,4,cv2.BORDER_CONSTANT)
        **if** size != BASE_SIZE:
            **return** cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        **else**:
            **return** img

For example here is one from the snowman file:

![snowman](https://cdn-images-1.medium.com/max/2000/1*ZUpVgFoHlWubOBKlDsmX5Q.png)*snowman*

We converted all of the strokes into corresponding images and stored them in corresponding folders(train & test).

![A batch of images in training data](https://cdn-images-1.medium.com/max/2000/1*Qr-8s-vhiRipJ2P5Nm98dQ.png)*A batch of images in training data*

## Using a convolutional neural network to identify the doodle

Ideally, there are multiple ways this problem could have been tackled, for example as there is a sequential component to it with strokes being a sequence of coordinates a recurrent neural network could also be used. I rather preferred to tackle this as a computer vision problem as it is more easier to test and learn by visualizing the results in an image problem than a sequential one like the one we are working on.

The architecture chosen by us was Resnets and its variants([https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)). We started off with Resnet18 and gradually tested the problem for performance even across bigger architectures. **Empirically I observed Resnet34 gave us more bang for each buck than any of other networks.**

## **What did I learn from initial experimentation from the data**

A look at the data and subsequent runs suggested the need for this problem was a simplified network which could zoom through these large number of doodles. The need of the hour was a simplified network with the ability to run multiple epochs within a limited time frame. Hence, I did not even try any complicated architectures which in the end was a great decision.

Using Resnet34 the highest volume of data that I could run my experimentation on was 30% and it did show that ***more data does help with the generalization ability when you have a simple but quite diverse(more number of labels) dataset.***

## Noise in the data

On further observation, it was observed that there was lots of noise in the training data, that is there were lots of doodles which were wrongly labeled. This was actually impacting the [learning](https://hackernoon.com/tagged/learning) capability of the model as you are inherently giving wrong instructions to it. Possible solutions for this, which I could not try are the development of another network to identify wrongly labeled images or hand labeling high loss images(those where there is highest difference between actual and predicted).

## **Where did I land**

The highest MAP@3 I got was 0.91444 on the public leaderboard which generalized quite well with a score of 0.91318 on the private leaderboard. Considering the winner of the competition was on 0.95480 I was on the correct path. The one strategy that could have made a difference was if I had spent more time improving the noise in the data but this is a learning for next time.

### Thanks, everyone for reading my experience of tackling this extremely interesting problem! For anyone looking to try there hand at this below is the link to the competition.
[**Quick, Draw! Doodle Recognition Challenge**
*How accurately can you identify a doodle?*www.kaggle.com](https://www.kaggle.com/c/quickdraw-doodle-recognition)

<iframe src="https://medium.com/media/3c851dac986ab6dbb2d1aaa91205a8eb" frameborder=0></iframe>
