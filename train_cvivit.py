import argparse
from phenaki_pytorch import CViViT, CViViTTrainer
import os

parser = argparse.ArgumentParser(description='CVIVIT Training.')
parser.add_argument('-m', '--mode', default="local", type=str,
                    help='Mode: local, jeanzay')
parser.add_argument('-bs', '--batch-size', default=4, type=int,
                    help='Batch size')
parser.add_argument('-ga', '--grad-accum', default=2, type=int,
                    help='Gradient accumulation')
parser.add_argument('-ema', '--use-ema', default=False, type=str,
                    help='Whether to use EMA')
parser.add_argument('-ts', '--num-train-steps', default=40000, type=int,
                    help='Number of train steps')
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                    help='Learning rate')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                    help='Weight decay for AdamW')
parser.add_argument('-mg', '--max-grad-norm', default=10, type=int,
                    help='Gradient norm clipping value')
parser.add_argument('-lw-st', '--linear-warmup-start-factor', default=0.5, type=float,
                    help='Linear warmup start factor')
parser.add_argument('-lw-ti', '--linear-warmup-total-iters', default=1000, type=int,
                    help='Linear warmup total iters')
parser.add_argument('-ca-max', '--cosine-annealing-t-max', default=40000, type=int,
                    help='Cosine annealing max iters')
parser.add_argument('-ca-min', '--cosine-annealing-min-lr', default=5e-5, type=float,
                    help='Cosine annealing minimum learning rate')
parser.add_argument('-sd', '--save-dir', default='', type=str,
                    help='Save directory')
parser.add_argument('-cl-w', '-commit-loss-weight', default=1., type=float,
                    help='Self explanatory')
parser.add_argument('-gl-w', '-generator-loss-weight', default=1., type=float,
                    help='Self explanatory')
parser.add_argument('-pl-w', '-perceptual-loss-weight', default=1., type=float,
                    help='Self explanatory')
parser.add_argument('-il-w', '-i3d-loss-weight', default=1., type=float,
                    help='Self explanatory')
parser.add_argument('-rl-w', '-reconstruction-loss-weight', default=1., type=float,
                    help='Self explanatory')
parser.add_argument('-ud', '-use-discr', default=1, type=int,
                    help='Self explanatory')
parser.add_argument('-gp-w', '-gp-weight', default=10000, type=int,
                    help='Self explanatory')
parser.add_argument('-ckpt', '--resume-ckpt', default='no', type=str,
                    help='Resume training with ckpt')
args = parser.parse_args()

results_folder = args.save_dir
wandb_mode = "offline" if args.mode == "jeanzay" else "online"
force_cpu = False if args.mode == "jeanzay" else True

train_on_images = False

if args.mode == "jeanzay":
    wandb_mode = "offline"
    dataset_folder = ''
    batch_size = args.batch_size
    dim_head = 64
    local_vgg = True
else:
    if (train_on_images):
        dataset_folder = ''
    else:
        dataset_folder = ''
    batch_size = 2
    dim_head = 2
    local_vgg = False

print("CVIVIT training is starting...\n")

print("Dataset : ", dataset_folder)

cvivit = CViViT(
    dim=512,  # embedding size
    codebook_size=8192,  # codebook size
    image_size=128,  # H,W
    patch_size=8,  # spatial patch size
    local_vgg=local_vgg,
    wandb_mode=wandb_mode,
    force_cpu=force_cpu,
    temporal_patch_size=2,  # temporal patch size
    spatial_depth=4,  # nb of layers in the spatial transfo
    temporal_depth=4,  # nb of layers in the temporal transfo
    dim_head=dim_head,  # hidden size in transfo
    heads=8,  # nb of heads for multi head transfo
    ff_mult=4,  # 32 * 64 = 2048 MLP size in transfo out
    commit_loss_w=args.cl_w,  # commit loss weight
    gen_loss_w=args.gl_w,  # generator loss weight
    perceptual_loss_w=args.pl_w,  # vgg loss weight
    i3d_loss_w=args.il_w,  # i3d loss weight
    recon_loss_w=args.rl_w,  # reconstruction loss weight
    use_discr=args.ud,  # whether to use a stylegan loss or not
    gp_weight=args.gp_w
)


trainer = CViViTTrainer(
    cvivit,
    folder=dataset_folder,
    batch_size=batch_size,
    force_cpu=force_cpu,
    wandb_mode=wandb_mode,
    train_on_images=train_on_images,
    grad_accum_every=args.grad_accum,  # use this as a multiplier of the batch size
    # recommended to be turned on (keeps exponential moving averaged cvivit) unless if you don't have enough resources
    use_ema=args.use_ema,
    num_train_steps=args.num_train_steps,
    lr=args.learning_rate,  # Learning rate
    wd=args.weight_decay,  # Weight decay
    max_grad_norm=args.max_grad_norm,  # gradient clipping
    # start the warmup at this factor of the lr
    linear_warmup_start_factor=args.linear_warmup_start_factor,
    # nb of iterations for the warm up
    linear_warmup_total_iters=args.linear_warmup_total_iters,
    # nb of iterations for the cosine annealing
    cosine_annealing_T_max=args.cosine_annealing_t_max,
    cosine_annealing_eta_min=args.cosine_annealing_min_lr,  # lr at the end of annealing
    results_folder=results_folder
)

if trainer.accelerator.is_main_process:
    from prettytable import PrettyTable

    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        pytorch_total_params = sum(p.numel() for p in cvivit.parameters())
        print('Number of params in total: ', pytorch_total_params)
        return total_params
    
    count_parameters(cvivit)

if args.resume_ckpt != 'no':
    print(args.resume_ckpt)
    trainer.load(args.resume_ckpt)
    print('Model loaded from file:', args.resume_ckpt)
    trainer.scheduler_optim_overhead = int(
        args.resume_ckpt.split('ckpt_accelerate_')[1].split('/')[0])
    print('lr scheduler overhead:', trainer.scheduler_optim_overhead)

cvivit.train()


# reconstructions and checkpoints will be saved periodically to ./results
trainer.train()
