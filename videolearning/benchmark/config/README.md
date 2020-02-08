#Config File

The config file has the following parameters. Please adjust folder names etc according to your environment.

- dataset_root: the root folder of the dataset
- data_folder: the folder of the packed numpy files in hdf format ( e.g. /your/path/here/data_train/packed_numpy_new_flow_rgb )
- label_idx_file: file with index and respective class label  (e.g. ./mapping_without_meta.txt)
- model_folder: model files will be stored here for evey epoch ( e.g. /your/path/here/test_out/models_files) please change storing intervals is you use sparse feature loading 
- n_classes: number of classes ( e.g. 513)
- feature_dim: dimension of input features (2048)
- seed: fix seed for reproducability (e.g. 42)
- balance_training: balance training data 1/0 
- lr: learning rate (e.g. 0.0001)
- lr_adj: adjust learning rate 0/1
- batch_size: 256,
- num_workers: num workers for the pytorch dataloader, 0 for no parallel loading
- embed_dim: size of embedding layer (e.g. 2048)
- epochs: number of epochs to train, note that for sparse loading, one sparse loading run is counted as one epoch, e.g. each hdf file has ~ 150k frames, if only 50k are loaded you need 3x number of epochs for the same training
- act_func: activation function, relu and sigmoid implemented so far
- init_mean: initialization value for mean
- init_var: initialization value for var
- bias: initialization value for bias
- save_model: save model 0/1
- sparse: use sparse data loading, one sparse loading run is counted as one epoch, so number of epcohs need to be increased ! E.g. each hdf file has ~ 150k frames, if only 50k are loaded you need 3x number of epochs for the same
- sparse_num_frames: number of frames to load from each file in one epoch, only used if sparse = 1, ignored otherwise 
- log: log level (e.g. DEBUG)
- log_str: log string
- prefix: model file prefix (e.g. model_ ),
- test_epoch: epoch to process for output computation and video aligment
- get_cond_probs: compute output as conditional probabilitites, divide by class prior, can be 0/1
- out_probs: path for output (e.g. /your/path/here/test_out/out_probs_dim2048_)
- out_segmentation: path for segmentation files (e.g. /your/path/here/test_out/segmentation_dim2048_)
- test_feat: folder of test features (/your/path/here/data/data_test/features)
- transcripts: folder of test transcripts (/your/path/here/data/data_test/transcripts)
- gt: folder of test groundtruht (/your/path/here/data/data_test/groundTruth)
