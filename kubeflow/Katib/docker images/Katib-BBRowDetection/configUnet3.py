class config_func_unet3:

    def __init__(self, server = True, net_type = "UNet3"):
        self.server = server       # Flag that indicates whether to use server or local machine
        self.net_type = net_type  # Indicates Architecture that we want to use: UNet3, Unet_orig,UNet3_modified...
        self.load_numpy = True   # Flag that indicates what type of data we use as input
        self.use_mask = False # Flag that indicates are we masking data with boundery and valid masks
        self.use_weights = False  # Flag that indicates whether to use class weights when initializing loss function
        self.do_testing = True   # Flag that indicates whether to do testing after the training is done
        self.count_logs_flag = False # Flag that indicates whether to plot number of classes and pixels in tensorboard, classwise and batch-wise
        self.zscore = False       # Flag that indicates whether we use zscore normalization in preprocessing
        self.binary = True       # Flag that indicates whether we do binary semantic segmentation
        self.freeze_backbone_weights = False # Flag that indicates whether to freeze backbone weights
        self.early_stop = False  # Initial early stopping flag
        self.save_best_model = True # Initial "best model" flag, indicates whether to save model in corresponding epoch
        self.scheduler_lr = 'multiplicative' # Indicates which scheduler to use
        self.dataset = "Borovnica"            # "mini" or "full"

        if self.server:                  # Depending on server flag, we use different device settings:
                self.device = "cuda"     # if server is True, that is, if we are using server machine, device will be set as "cuda"
        elif self.server == False:
            self.device = "cpu"          # else if server is False and we are using local machine or server access node, device will be set as "cpu"
        self.classes_labels = ['Borovnica']  # Classes that we are trying to detect. For BCE, background class is rejected

        self.loss_type = 'bce'       # Indicates loss type we want to use: bce, ce, ce_1
        self.img_data_format = '.npy'  # Indicated the type of data we use as input
        self.set_random_seed = 15    # Setting random seed for torch random generator
        self.batch_size = 4      # Size of batch during the training,validation and testing
        self.shuffle_state = 1   # Random shuffle seeed
        self.GPU_list = [0]      # Indices of GPUs that we want to allocate and use during the training
        self.weight_decay = 0    # L2 penalty
        self.optimizer_patience = 3 #  When using ReduceLR scheduler: Indicates number of epochs without loss minimum decrease,
            # after which the learning rate will be multiplied by a lambda parameter
        self.save_checkpoint_freq = 1    #Initial frequency parameter for model saving


        if self.save_checkpoint_freq < 10:  # Further updating saving frequency parameter
            self.save_checkpoint_freq = 0
        if self.save_checkpoint_freq >= 100:
            self.save_checkpoint_freq = int(self.epochs / 10)  # frequency of saving checkpoints in epochs
        if self.save_checkpoint_freq == 0:
            self.save_checkpoint_freq = 1000

        self.num_channels = 5    # Number of input channels: Red, Green, Blue, NIR, Red-Edge
        self.num_channels_lab = len(self.classes_labels)
        self.img_h = 512 # Input image Height
        self.img_w = 512 # Input image Weight
        self.img_size = [self.img_h, self.img_w] # Input channel and label shape

        self.train_losses = []   # Container in which we store losses for each training batch
        self.validation_losses = [] # Container in which we store losses for each validation batch

        self.epoch_model_last_save = 0 # Counter that counts number of epochs since the most recent model saving
        self.count_train = 0      # Counter that counts number of training batches
        self.count_val = 0        # Counter that counts number of validation batches
        self.count_train_tb = 0

        self.es_min = 1e9    # Initial minimum parameter for early stopping
        self.es_epoch_count = 0  # Epoch counter for early stopping
        self.es_check = 5 # Number of epochs after wich we dont have new minimal validation loss and after wich we apply early stopping

        if self.loss_type == 'bce':
            self.background_flag = False
        else:
            self.background_flag = True
        self.background_names = []
        self.background_area = []
        self.foreground_names = []
        self.foreground_area = []
        self.test_losses = []
        self.iou_per_test_image_fg = []
        self.k_index = 1
    