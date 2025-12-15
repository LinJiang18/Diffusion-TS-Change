import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser("Diffusion-TS-Change")

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    args.device = "cuda"
    args.output_dir = os.path.join("OUTPUT", args.name)

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    config = load_yaml_config(args.config_file)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(args.output_dir)
    logger.save_config(config)


    model = instantiate_from_config(config['model']).cuda()
    dataloader_info =  build_dataloader(config,args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
    else:
        trainer.load(args.milestone)
        samples = trainer.sample(num_samples = config["sample"]["params"]["num_samples"], batch_size = config["sample"]["params"]["batch_size"])
        if config["sample"]["params"]["auto_norm"]:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

if __name__ == '__main__':
    main()
