
# Benchmark implementation for the Webvision video track

The benchmark comprises a vanilla baseline implementation of the training and evaluation with precomputed features in pytorch. 

- Please first download the data as described here:

- Unpack the data into a data folder e.g. /my/data/folder/

- Adjust the path in the ./config/train_config_relu_org_files_2048dim.json by replacing /my/data/folder/ with your path.

## Training

The training is run by the function

./src/train_multic_webvisionTest.py

The dataset needs ~150GB memory + some overhead in case you want to balance the training data. As not all systems provide this memory, we provide a full and a sparse data loader

#### Full dataloader

If you have enough memory, please set the "sparse" flag in the config to zero and you can start the training.

#### Sparse dataloader

If you don't have enough memory, please set the sparse flag to 1 and estimate how many frames you want to load per file for each epoch. Depending on the fraction of files you can load, you also need to adapt the number of epochs. Each hdf file has ~150k frames, so if you load 15k frames per file you need 10 epochs until all data has been processed once and you need to multiply the number of epochs by 10 to get the same training.

#### Balancing training data

As the original distribution of the training data is highly imbalanced, it is recommended to balance the training data (downsample most occurring classes, upsample least occurring classes). You can do this by setting the flag "balance_training" to 1. The implementation can be found in the respective dataset classes ./datasets/train_ds_full.py and ./datasets/train_ds_sparse.py and can be modified for different ratios.


## Computing output probabilities

The test function expects a score between [0,..,1] for each class for each frame of the video. You can compute output probabilities with the function:

./src/mp_save_probs_webvisionTest.py

#### Use softmax?

The function uses softmax to convert the output of the last linear layer to [0, .., 1]. If you don't want that, please comment line 49 in mp_save_probs_webvisionTest.py

#### Use conditional probabilities

The function can further make use for the class prior from training to compute the conditional probabilities of the class scores (might be a good idea if the training data is not balanced). You can turn the is function on/off by setting the respective parameter "get_cond_probs" in the config file.

#### Prepare challenge submission

You can also use the output of that function for the challenge submission. A detailed description is given here:


## Testing


Annotation by natural language can be very inconsistent and even contradicting, e.g. we have three different classes "whisk_egg", "beat_egg", and "mix_egg", which obviously all refer to the same action and we have other classes such as "add_pepper" which can refer to the bell pepper as well as to grounded pepper powder.

It is therefore difficult to assess the classification by just comparing the max score label to the annotated one as nobody knows if the annotator was more a "whisk_egg", "beat_egg", or "mix_egg" type of person.

We therefore decided to resort to the task of video alignment. Here, the transcripts of the actions ( i.a. the action labels in the right order) are already given and the task is to find the right boundaries for the given actions in the video. We know from previous work on weak learning for video sequences (see e.g. https://ieeexplore.ieee.org/document/8585084, https://arxiv.org/abs/1610.02237) that this task is usually a good surrogate for the overall classification accuracy. In this case it helps to avoid any language inconsistencies as it aligns the output to the correct action labels only and ignores the rest. It is therefore not so important which score was given to "mix_egg" or "beat_egg", as only the scores of the class "whisk_egg" would be considered (if this was the annotation).

For testing please run the function:

./src/test_multic_webvisionTest.py

We measure accuracy as intersection over union (Jaccard IoU), by first computing the framewise IoU of all classes and take the mean over all present classes as IoU for each video. The final score is computed as mean over all video IoUs. 

The here provided testing routine is run "as is" on the evaluation server.



## Challenge Data

To give you an idea how the the dataset test data and the challenge test data compare on the vanilla benchmark, we run a short evaluation with the same model on both data (trained over 50 epochs, 2048 dim embedding layer):

![Comparison](https://hildekuehne.github.io/img/comp_test_challenge.png)

You can see that the challenge data did a bit better in the vanilla test than the original test dataset, but results are comparable.




