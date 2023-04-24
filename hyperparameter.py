import logging
from types import SimpleNamespace
from typing import List
from tap import Tap
import torch
from args import FinetuneArgs
from kd_finetune import main as kd_finetune_main
from finetune_hp import finetune
from kd_finetune import Args as KDFinetuneArgs
from finetune import Args as FinetuneArgs

import ray
from ray import tune
from ray import air
import ray.train.torch
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import setup_wandb
from ray.air.integrations.wandb import WandbLoggerCallback

from ray.tune.logger import DEFAULT_LOGGERS
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from load_glue import load_tokenized_glue_dataset
from transformers import AutoModelForSequenceClassification
from model import BertForSequenceClassificationWithLoss, make_sequence_classifier

from utils import set_random_seed, setup_logging

class Args(Tap):
    hp_lr_low : float = 1e-5 
    hp_lr_high : float = 5e-5 
    hp_wd_low : float = 0.000
    hp_wd_high : float = 0.02
    batch_sizes : List[int] = [16,32]
    trials : int = 12
    num_best_repeats : int = 3
    wandb_name : str

    def configure(self):
        self.add_subparsers(dest="train_type")
        self.add_subparser('finetune', FinetuneArgs)
        self.add_subparser('kd_finetune', KDFinetuneArgs)

def run_training(config,args):
    
    print(tune.is_session_enabled())
    run_args = SimpleNamespace(**args.as_dict())

    run_args.lr = config["lr"]
    run_args.weight_decay = config["weight_decay"]
    run_args.batch_size = config["batch_size"]

    if run_args.train_type == 'finetune':
        finetune_main(run_args)
    if run_args.train_type == 'kd_finetune':
        kd_finetune_main(run_args)
    
def train_func(config,args, model,dataset):
    wandb = setup_wandb(config)

    run_args = SimpleNamespace(**args.as_dict())

    run_args.lr = config["lr"]
    run_args.weight_decay = config["weight_decay"]
    run_args.batch_size = config["batch_size"]

    finetune(ray.air.session.get_local_rank(),BertForSequenceClassificationWithLoss(model), dataset, run_args, in_tune_session=True)

def main(args):
    set_random_seed(args.seed)

    setup_logging(args.outputdir)

    dataset = load_tokenized_glue_dataset(
        args.gluepath, args.dataset, augmented=args.use_augmented_data
    )

    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.modelpath,num_labels=len(dataset.train.possible_labels),ignore_mismatched_sizes=True)
    except OSError:
        model = torch.load(args.modelpath)

    model = make_sequence_classifier(model,len(dataset.train.possible_labels))
    
    config = {
        "lr": tune.loguniform(args.hp_lr_low, args.hp_lr_high),
        "weight_decay": tune.uniform(args.hp_wd_low,args.hp_wd_high),
        "batch_size": tune.choice(args.batch_sizes)
    }
    # config = {
    #     "lr": tune.grid_search([1e-5,2e-5,5e-5]),
    #     "weight_decay": 0.01,
    #     "batch_size": tune.grid_search([16,32])
    # }

    metric="score"
    opt_mode="max"

    scheduler = ASHAScheduler(
        max_t=args.num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    hyperopt_search = HyperOptSearch(metric=metric, mode=opt_mode, random_state_seed=args.seed)

    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker=tune.with_parameters(train_func, args=args,model=model,dataset=dataset),
        scaling_config=air.config.ScalingConfig(num_workers=args.num_gpus, use_gpu=True),
    )

    tuner = tune.Tuner(
        trainable=trainer,
        tune_config=tune.TuneConfig(
            search_alg=hyperopt_search,
            metric=metric,
            mode=opt_mode,
            #scheduler=scheduler,
            #num_samples=args.trials,
        ),
        run_config=air.RunConfig(
            callbacks=[
                WandbLoggerCallback(project=args.wandb_name)
            ]
        ),
        param_space={
            "train_loop_config":config
        },
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric, opt_mode)

    print("Best result after hyperopt:",best_result)

    print(best_result.config)
    best_config = best_result.config["train_loop_config"]

    all_scores = []
    for seed in range(args.num_best_repeats):
        args.seed = seed
        trainer = ray.train.torch.TorchTrainer(
            train_loop_per_worker=tune.with_parameters(train_func, args=args,model=model,dataset=dataset),
            train_loop_config=best_config,
            scaling_config=air.config.ScalingConfig(num_workers=2, use_gpu=True),
        )
        result = trainer.fit()
        print(result.metrics_dataframe)
        all_scores.append(result.metrics_dataframe.score.to_list())
    logging.info(all_scores)

if __name__ == "__main__":
    main(Args().parse_args())