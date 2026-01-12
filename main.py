import os
import torch
import argparse
import numpy as np

from engine.logger import Logger,DecompTrace,save_trace_npz,load_trace_npz
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader,unnormalize_sample
from Utils.io_utils import load_yaml_config, seed_everything, instantiate_from_config

def parse_args():
    parser = argparse.ArgumentParser("Diffusion-TS-Change")

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--heavy", action="store_true",
                        help="Enable heavy-tailed (Student-t) noise for diffusion.")

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
    config.setdefault('model', {}).setdefault('params', {})
    config['model']['params']['mode'] = bool(args.train)
    config['model']['params']['heavy'] = bool(args.heavy)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(args.output_dir)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()

    dataloader_info =  build_dataloader(config,args)
    dataloader = dataloader_info['dataloader']
    normalizer = dataloader_info['normalizer']
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader, logger=logger)

    if args.train:
        trainer.train()
    else:
        milestone = 10
        trainer.load(milestone)
        if config['sample']['params']['whe_trace']:
            trace = DecompTrace()
            trainer.ema.ema_model.set_trace(trace)
        samples = trainer.sample(num_samples = config["sample"]["params"]["num_samples"], batch_size = config["sample"]["params"]["batch_size"])
        np.save(os.path.join(args.output_dir, f'generate_data_norm_{args.name}.npy'), samples)
        unnorm_samples =  unnormalize_sample(samples,normalizer)
        np.save(os.path.join(args.output_dir, f'generate_data_unnorm_{args.name}.npy'), unnorm_samples)
        save_trace_npz(trace, os.path.join(args.output_dir,f"trace_{args.name}.npz"))



if __name__ == '__main__':
    main()
