_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/segformer_b5.py',
    # Synthia->Cityscapes Data Loading
    '../_base_/datasets/synthia_to_cityscapes_768x768.py',
    # Basic UDA Self-Training
    '../_base_/uda/stylization.py',
    # AdamW Optimizer
    '../_base_/schedules/optimizer.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]

# Random Seed
seed = 0
# Modifications to Basic UDA

data = dict(
    train=dict())


uda = dict(
    type='STYLIZATION',
    fdist_scale_min_ratio = 0.75,
    reg_ratio = 1.0,
    disent_ratio = 1.0,
    reg_layers = [0, 1, 2, 3],
    disent_layers = [0, 1],
    original_weight = 0.5,
    style_weight  = 0.5,

    reg_lambda = [0.2, 0.02, 0.002, 0.0002],
    disent_lambda = [0.05, 0.005, 0.0005, 0.00005],

    style_img_folder = 'data/ImageNet/valid',
    total_iter = 20000,
    batch_size = 4,
    crop_size = 768,
    threshold = 0.01,

)

# Optimizer Hyperparameters
optimizer_config = None

optimizer = dict(
    lr=2e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=20000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

# Meta Information for Result Analysis
name = 'iter_{}_lr_{}_orig_{}_style_{}_regw_{}_regr_{}_disentw_{}_disentr_{}_threshold_{}_seed_{}'.format(runner['max_iters'],  optimizer['lr'], uda['original_weight'], uda['style_weight'], uda['reg_lambda'][0], uda['reg_ratio'], uda['disent_lambda'][0], uda['disent_ratio'], uda['threshold'], seed)
exp = 'synthia-mitb5'
name_dataset = 'synthia2cityscapes'
name_architecture = 'segformer_mitb5'
name_encoder = 'mitb5'
name_decoder = 'segformer'
name_uda = 'stylization'
name_opt = 'adamw_2e-05_pmTrue_poly10warm_1x2_20k'