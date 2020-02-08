
# Webvision video track

Welcome to the Webvision Video Track 2020 challenge!

The idea of this challenge is to learn action representations from video subtitles without any human supervision. 

The training data is based on the recently released MiningYouTube dataset (working title is 'Weak YouTube', but some people found that this sounds too negtive). Please find the homepage here: https://github.com/hildekuehne/Weak_YouTube_dataset

The dataset comprises ~ 20,000 YouTube videos that display explain various egg recipes, namely for fried egg, scrambled egg, pancake, omelet, and egg roll.  

For training, we provide the video indexes with the respective subtitles (downloaded in 2017/2018) as well as pre-extracted video clips with tentative labels as described in the paper. For the pre-extracted videos, we further provide pre-computed TSN features (https://github.com/yjxiong/temporal-segment-networks) based on the Kinetics pretrained model.  

The training data is available under: https://github.com/hildekuehne/Weak_YouTube_dataset/tree/master/train 

The training data is based on subtitles only. It additionally comprises a set of ~5000 videos with class labels and a human annotation if this class label is present in the video, which can be used for training or validation.

The testing data (and validation data for the challenge) is available under: https://github.com/hildekuehne/Weak_YouTube_dataset/tree/master/test
 

## Two tracks

Note that we will have two tracks for this challenge, one based on the original videos and one based on precomputed features only!

## Benchmark

The benchmark comprises a vanilla baseline implementation of the training and evaluation with precomputed features in pytorch.

## Data

The data folder comprises explanation and the download script for the challenge test data.

The challenge data did a bit better in our first vanilla test than the original test dataset, but results are comparable. It can be expected that whatever worked on the original test data, will also work on the challenge data.

Please find a comparision of the vanilla system on the dataset test data and the challenge test data here (50 epochs, 2048 dim embedding layer):


![Comparison of test data and challenge data](https://hildekuehne.github.io/img/comp_test_challenge.png)



## Submission

Details for submission will the announced!

