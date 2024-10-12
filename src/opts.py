import argparse
import os


def parse_opts():
    parser = argparse.ArgumentParser()

    # Overall Settings
    default_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--root_path",
        type=str,
        default=default_root_path,
        help="Root path of the project",
    )
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--torch_model_path", type=str, default="None")
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--setting_name", type=str, default="Setup1")

    # mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "sample"],
        default="train",
        help="train or sample mode",
    )

    # Dataset
    parser.add_argument("--min_label", type=int, default=1, metavar="N")
    parser.add_argument("--max_label", type=int, default=60, metavar="N")
    parser.add_argument("--num_channels", type=int, default=1, metavar="N")
    parser.add_argument("--image_size", type=int, default=32, metavar="N")
    parser.add_argument("--max_num_img_per_label", type=int, default=2000, metavar="N")
    parser.add_argument(
        "--max_num_img_per_label_after_replica", type=int, default=200, metavar="N"
    )
    parser.add_argument("--max_power", type=float, default=240000, metavar="N")
    parser.add_argument("--max_wave", type=float, default=1600, metavar="N")

    # Model Config
    parser.add_argument("--model_channels", type=int, default=64, metavar="N")
    parser.add_argument("--num_res_blocks", type=int, default=2, metavar="N")
    parser.add_argument("--num_heads", type=int, default=4, metavar="N")
    parser.add_argument("--num_groups", type=int, default=8, metavar="N")
    parser.add_argument("--attention_resolutions", type=str, default="16_32")
    parser.add_argument("--channel_mult", type=str, default="1_2_4_8")
    parser.add_argument("--cond_drop_prob", type=float, default=0.5)

    # Training
    parser.add_argument("--pred_objective", type=str, default="pred_noise")
    parser.add_argument("--niters", type=int, default=200000, metavar="N")
    parser.add_argument("--resume_niter", type=int, default=0, metavar="N")
    parser.add_argument("--train_timesteps", type=int, default=1000, metavar="N")
    parser.add_argument("--train_batch_size", type=int, default=16, metavar="N")
    parser.add_argument("--train_lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--train_amp", action="store_true", default=False)
    parser.add_argument("--gradient_accumulate_every", type=int, default=1, metavar="N")
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument("--sample_every", type=int, default=1000, metavar="N")
    parser.add_argument("--save_every", type=int, default=10000, metavar="N")

    # Label embedding setting
    parser.add_argument("--net_embed", type=str, default="ResNet34_embed")
    parser.add_argument("--epoch_cnn_embed", type=int, default=200)
    parser.add_argument("--resumeepoch_cnn_embed", type=int, default=0)
    parser.add_argument("--epoch_net_y2h", type=int, default=500)
    parser.add_argument("--dim_embed", type=int, default=128)
    parser.add_argument("--batch_size_embed", type=int, default=256, metavar="N")

    # Vicinal loss
    parser.add_argument("--kernel_sigma", type=float, default=-1.0)
    parser.add_argument(
        "--threshold_type", type=str, default="hard", choices=["soft", "hard"]
    )
    parser.add_argument("--kappa", type=float, default=-1)
    parser.add_argument("--nonzero_soft_weight_threshold", type=float, default=1e-3)

    # Sampling
    parser.add_argument("--sampler", type=str, default="ddpm")
    parser.add_argument("--sample_timesteps", type=int, default=1000, metavar="N")
    parser.add_argument("--sample_cond_scale", type=float, default=6.0)
    parser.add_argument("--ddim_eta", type=float, default=0)
    parser.add_argument("--samp_batch_size", type=int, default=4)
    parser.add_argument(
        "--sample_niter",
        type=int,
        default=0,
        help="Iteration to load for sampling",
    )
    parser.add_argument("--num_samples", type=int, default=4, metavar="N")

    args = parser.parse_args()
    return args
