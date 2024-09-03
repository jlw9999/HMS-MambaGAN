import torch as t
import warnings

class BaseConfig(object):
#     gpu_ids = [0, 1]
    gpu_ids = [0]
    isTrain = True
    continue_train = False

    # training
    lr = 0.0001
    # lr = 0.00005
    lr_policy = 'linear'
    n_epochs = 100
    n_epochs_decay = 100
    epoch_count = 1
    continue_train = False
    batch_size = 1
    # vis
    verbose = False
    # model
    load_iter = 1
    epoch = 'latest'
    
    dataroot = ".save_path"
        
    batchSize = 8
        
    loadSize = 286
    fineSize = 256
  
    input_nc = 1
    output_nc = 1

   # input_nc = 3
   # output_nc = 3

    ngf = 64
    ndf = 64
    which_model_netD = 'basic'
    which_model_netG = 'HMSMambaGAN'
    n_layers_D = 3
    name = 'experiment_name'
    dataset_mode = 'aligned'
    which_direction ='AtoB'
    nThreads=2
    checkpoints_dir = './checkpoints'
    norm ='instance'
    serial_batches ='store_true'
    display_winsize =256
    display_id = 1
    display_server ="http://localhost"
    display_port = 8097
    no_dropout ='store_true'
    max_dataset_size=float("inf")
                                
    resize_or_crop ='resize_and_crop'
    no_flip ='store_true'
    init_type ='normal'
    
    vit_name ='Convmamba-B_16'

    pre_trained_transformer = 0
    pre_trained_resnet = 0

    ntest = float("inf")
    results_dir = './results/'
    aspect_ratio = 1.0
    phase ='test'
    which_epoch ='1'
    how_many = 50

    isMedical_data = True
    

    def _parse(self, kwargs):
        """
        update config
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
