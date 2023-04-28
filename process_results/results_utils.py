from ast import parse
from collections import defaultdict
import math
from pathlib import Path, PosixPath
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kd_finetune import Args as KDFinetuneArgs
from finetune import Args as FinetuneArgs

def f(x):
    return f"{x*100:.1f}"

DATASETS = ["QNLI","RTE","SST-2","MRPC","MNLI","QQP","CoLA"]
DEFAULT_METRIC = {
    "QNLI": "accuracy",
    "RTE": "accuracy",
    "SST-2": "accuracy",
    "MNLI": "accuracy",
    "MRPC": "F1_score",
    "QQP": "F1_score",
    "CoLA": "matthews",
}
def parse_command(command):
    if "python kd_finetune.py" in command:
        args = KDFinetuneArgs().parse_args(command.split(" ")[2:])
    if "python finetune.py" in command:
        args = FinetuneArgs().parse_args(command.split(" ")[2:])
    return eval(str(args))

class TrainingRunResult:
    def __init__(self,command):
        self.args = parse_command(command)
        self.intermediate_evals = []
        self.epoch_evals = []


def read_log(file):
    all_runs = {}
    current_run = None

    with open(file,"r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            match = re.match("INFO:root:Step ([0-9]+):\n", line)
            if match is not None:
                step = int(match.group(1))
                if f.readline() == "INFO:root:Eval results:":
                    eval_res = re.match("INFO:root:(.+)\n", f.readline()).group(1)
                    eval_res = eval(eval_res)
                    current_run.intermediate_evals.append(eval_res | {"step":step})
            
            match = re.match("INFO:root:{\n", line)
            if match is not None:
                epoch_eval_lines = ""
                curr_line = "{"
                while curr_line != "}\n":
                    epoch_eval_lines += curr_line
                    curr_line = f.readline()
                epoch_eval_lines += "}"
                epoch_eval = json.loads(epoch_eval_lines)
                current_run.epoch_evals.append(epoch_eval)
                current_run.args["lr"] = epoch_eval["lr"]
                current_run.args["batch_size"] = epoch_eval["batch_size"]

            match = re.match("INFO:root:({'command_line'.+})\n", line)
            if match is not None:
                command = eval(match.group(1))['command_line']
                current_run = TrainingRunResult(command)
                all_runs[str(current_run.args)] = current_run
    return list(run for run in all_runs.values() if len(run.epoch_evals))

def get_metric(epoch_res,metric):
        if "dev_metrics" in epoch_res:
            return epoch_res["dev_metrics"][metric]
        else:
            return epoch_res["dev_accuracy"]

def get_metric(epoch_res,metric):
        if "dev_metrics" in epoch_res:
            return epoch_res["dev_metrics"][metric]
        else:
            return epoch_res["dev_accuracy"]

def to_dataframe(logfile, metric="accuracy"):
    training_runs = read_log(logfile)

    rows = []
    for run in training_runs:
        epoch_metrics = [get_metric(epoch_res, metric) for epoch_res in run.epoch_evals]
        
        if len(epoch_metrics)>0:
            rows.append({
                "lr":run.args["lr"],
                "batch_size":run.args["batch_size"],
                "epoch_metrics": epoch_metrics,
                "avg_metric":np.mean(epoch_metrics),
                "best_metric":max(epoch_metrics),
                "last3_mean":np.mean(epoch_metrics[-3:]),
                "last3_std":np.std(epoch_metrics[-3:]),
            })
    
    return pd.DataFrame(rows)

def make_label(args,label_args):
    label = []
    for argname in label_args:
        label.append(f"{argname}={args[argname]}")
    return ",".join(label)

def make_plot(results : TrainingRunResult, label_args=["lr","batch_size"], metric="accuracy",):
    num_epochs = len(results.epoch_evals)
    metrics = []
    for epoch_res in results.epoch_evals:
        metrics.append(get_metric(epoch_res, metric))
    plt.plot(range(1,num_epochs+1), metrics,label=make_label(results.args,label_args))

def make_plots(logfile, metric="accuracy"):
    training_runs = read_log(logfile)
    done_runs = set()
    for run in training_runs:
        if str(run.args) in done_runs:
            continue
        make_plot(run, metric=metric)
        done_runs.add(str(run.args))
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Dev Loss" if metric=="loss" else "Score")
    plt.show()

import math
import glob

def plot_repeats(dataset, hp_path, rp_path, plot=False):
    print(dataset)
    training_runs_hp = read_log(hp_path)
    rp_files = sorted(glob.glob(rp_path))
    if len(rp_files)==0:
        training_runs_rp = []
    else:
        training_runs_rp = read_log(rp_files[-1])
    
    if len(training_runs_rp)==0:
        training_runs_rp.append(max([(max(eval_res["dev_metrics"][DEFAULT_METRIC[dataset]] for eval_res in run.epoch_evals),run) for run in training_runs_hp],key=lambda x:x[0])[1])
    else:
        for run in training_runs_hp:
            if run.args["lr"]==training_runs_rp[0].args["lr"] and run.args["batch_size"]==training_runs_rp[0].args["batch_size"]:
                training_runs_rp.append(run)

    all_max_scores = []
    for run in sorted(training_runs_rp,key=lambda run:run.args["seed"]):
        scores = []
        for eval_res in run.epoch_evals:
            if "dev_metrics" in eval_res:
                score = eval_res["dev_metrics"][DEFAULT_METRIC[dataset]]
            else:
                assert DEFAULT_METRIC[dataset] == "accuracy"
                score = eval_res["dev_accuracy"]
            if math.isnan(score):
                score = -1
            scores.append(score)
        if plot:
            plt.plot(range(1,len(scores)+1),scores,label=f"seed={run.args['seed']}")
        all_max_scores.append(max(scores))
    if plot:
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        #plt.ylim([0,1])
        plt.show()
    print(all_max_scores)
    print(f"{f(np.max(all_max_scores))} ({f(np.mean(all_max_scores))}{{\\footnotesize±{f(np.std(all_max_scores))})}}")

if __name__=="__main__":
    read_log("../checkpoints/kd_finetune/tinybert/SST-2/tinybert/long_pretrain/prediction/log")