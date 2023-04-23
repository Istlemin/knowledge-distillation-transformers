from types import SimpleNamespace
from tap import Tap
from args import FinetuneArgs
from kd_finetune import main as kd_finetune_main
from finetune import main as finetune_main
from kd_finetune import Args as KDFinetuneArgs
from finetune import Args as FinetuneArgs

from ray import tune
from ray import air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch

class Args(Tap):
    hp_lr_low : float = 5e-6 
    hp_lr_high : float = 2e-4 
    hp_wd_low : float = 0.0
    hp_wd_high : float = 0.03


    def configure(self):
        self.add_subparsers(dest="train_type")
        self.add_subparser('finetune', FinetuneArgs)
        self.add_subparser('kd_finetune', KDFinetuneArgs)

def run_training(config,args):
    print(tune.is_session_enabled())
    run_args = SimpleNamespace(**args.as_dict())

    run_args.lr = config["lr"]
    run_args.weight_decay = config["weight_decay"]

    if run_args.train_type == 'finetune':
        finetune_main(run_args)
    if run_args.train_type == 'kd_finetune':
        kd_finetune_main(run_args)
    

def main(args):
    
    config = {
        "lr": tune.loguniform(args.hp_lr_low, args.hp_lr_high),
        "weight_decay": tune.uniform(args.hp_wd_low,args.hp_wd_high)
    }

    metric="score"
    opt_mode="max"

    scheduler = ASHAScheduler(
        max_t=args.num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    hyperopt_search = HyperOptSearch(metric=metric, mode=opt_mode)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run_training, args=args),
            resources={"cpu": 8, "gpu": args.num_gpus}
        ),
        tune_config=tune.TuneConfig(
            search_alg=hyperopt_search,
            metric=metric,
            mode=opt_mode,
            scheduler=scheduler,
            num_samples=1,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric, opt_mode)

    print(best_result)


if __name__ == "__main__":
    main(Args().parse_args())