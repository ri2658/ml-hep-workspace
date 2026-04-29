import argparse
from argparse import BooleanOptionalAction

from string import Template

import sys
from os.path import exists
import os

train_command_template = Template(
    """(python -u -W ignore train.py
          --data-train $data_train
          --data-config $data_config_dir/$data_config.yaml
          --network-config networks/$network_config.py
          --model-prefix /hwwtaggervol/models/rk/$model_name/$model_name
          --num-workers 12
          --batch-size 256
          --num-epochs 32
          --start-lr 3e-3
          --optimizer ranger
          --gpu "0,1"
          --fetch-step 0.02
          | tee /hwwtaggervol/models/rk/$model_name/log.txt)"""
)

inference_lines_template = Template(
    """(python -u train.py
          --predict
          --predict-output /hwwtaggervol/inferences/rk/$inference_dir/$sample_name.root
          --data-test '/hwwtaggervol/training/$data_test_dir/$sample/*.root'
          --data-config $data_config_dir/$data_config.yaml
          --network-config networks/$network_config.py
          --model-prefix $model_path
          --gpus "0,1"
          --batch-size 256
          --fetch-step 0.02
          --num-workers 2
          | tee -a /hwwtaggervol/inferences/rk/$inference_dir/log.txt)"""
)

plot_command_template = Template(
    """(python -u plots/plot_classification_fromoutput.py
          --name $plot_model_name
          --ifile /hwwtaggervol/inferences/rk/$inference_dir/qcd.root
          --odir /hwwtaggervol/plots/rk/$plot_dir/
          --signals hww_4q_merged,hww_3q_merged
          --bkgs qcdnolep,qcdnolep
          --mbranch fj_genRes_mass
          --jet AK15
          -v
          --isig $isig
          --isignames $isignames
          | tee -a /hwwtaggervol/plots/rk/$plot_dir/log.txt)"""
)

plot_losses_command_template = Template(
    """(python -u plots/plot_losses.py
          --model-path  /hwwtaggervol/models/rk/$model_name/$model_name
          --output /hwwtaggervol/plots/rk/$plot_dir/losses.png
          | tee -a /hwwtaggervol/plots/rk/$plot_dir/log.txt)"""
)

train_templ_file = "train_template.yml"
inference_templ_file = "inference_sep_samples_template.yml"


def make_inference(args):
    if args.inference_dir == "":
        args.inference_dir = args.model_name

    if args.plot_dir == "":
        args.plot_dir = args.model_name

    file_name = f"inferences/{args.job_name}_inference.yml"

    if exists(file_name):
        print("Job Exists")
        if args.overwrite:
            print("Overwriting")
        else:
            print("Exiting")
            sys.exit()

    with open(inference_templ_file, "r") as f:
        lines = Template(f.read())

    inference_lines = []
    for sample, sample_name in zip(args.inference_samples, args.inference_sample_names):
        inference_lines_args = {
            "sample_name": sample_name,
            "sample": sample,
            "data_config": args.data_config,
            "network_config": args.network_config,
            "model_path": args.model_path
            if args.model_path != ""
            else f"/hwwtaggervol/models/rk/{args.model_name}/{args.model_name}_best_epoch_state.pt",
            "inference_dir": args.inference_dir,
            "data_test_dir": args.data_test_dir,
            "data_config_dir": args.data_config_dir,
        }

        inference_lines.append(inference_lines_template.substitute(inference_lines_args))

    inference_command = " && \n          ".join(inference_lines)

    plot_command_args = {
        "plot_model_name": args.plot_model_name,
        "inference_dir": args.inference_dir,
        "plot_dir": args.plot_dir,
        "isig": ",".join(
            [
                f"/hwwtaggervol/inferences/rk/{args.inference_dir}/{sample}.root"
                for sample in args.plot_samples
            ]
        ),
        "isignames": ",".join(args.plot_sample_names),
    }

    plot_command = plot_command_template.substitute(plot_command_args)

    plot_losses_command_args = {
        "plot_dir": args.plot_dir,
        "model_name": args.model_name,
    }

    plot_losses_command = plot_losses_command_template.substitute(plot_losses_command_args)

    inf_plot_args = {
        "job_name": "-".join(args.job_name.split("_")),  # change underscores to hyphens
        "inference_dir": args.inference_dir,
        "plot_dir": args.plot_dir,
        "inference_command": inference_command,
        "plot_command": plot_command,
        "plot_losses_command": plot_losses_command,
    }

    with open(file_name, "w") as f:
        f.write(lines.substitute(inf_plot_args))

    print(f"Wrote to {file_name}")

    if args.run_inference:
        os.system(f"kubectl create -f {file_name} -n cms-ml-hvv")


def make_train(args):
    file_name = f"trainings/{args.job_name}.yml"

    if exists(file_name):
        print("Job Exists")
        if args.overwrite:
            print("Overwriting")
        else:
            print("Exiting")
            sys.exit()

    with open(train_templ_file, "r") as f:
        lines = Template(f.read())

    train_command_args = {
        "data_config_dir": args.data_config_dir,
        "data_config": args.data_config,
        "network_config": args.network_config,
        "model_name": args.model_name,
        "data_train": " ".join(
            [
                f'"/hwwtaggervol/training/{args.data_train_dir}/{sample}/*.root"'
                for sample in args.train_samples
            ]
        ),
    }

    train_command = train_command_template.substitute(train_command_args)

    train_args = {
        "job_name": "-".join(args.job_name.split("_")),  # change underscores to hyphens
        "model_dir": f"/hwwtaggervol/models/rk/{args.model_name}/",
        "train_command": train_command,
    }

    with open(file_name, "w") as f:
        f.write(lines.substitute(train_args))

    print(f"Wrote to {file_name}")

    if args.run_train:
        os.system(f"kubectl create -f {file_name} -n cms-ml-hvv")


class objectview(object):
    """converts a dict into an object"""

    def __init__(self, d):
        self.__dict__ = d


def from_json(args):
    import json

    with open(args.from_json, "r") as f:
        json_args = json.load(f)

    args_dict = vars(args)

    for arg, val in json_args.items():
        args_dict[arg] = val

    args = objectview(args_dict)

    args.job_name = args.from_json.split(".json")[0]
    args.model_name = args.job_name

    return args


def main(args):
    if args.from_json != "":
        args = from_json(args)
    else:
        if args.job_name == "":
            args.job_name = args.model_name
        else:
            args.job_name = "_".join(args.job_name.split("-"))  # hyphens to underscores

    args.data_config = args.data_config.split(".yaml")[0]
    args.network_config = args.network_config.split(".py")[0]

    if args.make_train:
        make_train(args)
    if args.make_inference:
        make_inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", default="", help="json file to load args from", type=str)
    parser.add_argument("--job-name", default="", help="", type=str)
    parser.add_argument("--inference-dir", default="", help="", type=str)
    parser.add_argument("--data-test-dir", default="ak15_Mar15/test/", help="", type=str)
    parser.add_argument("--data-train-dir", default="ak15_Mar15/train/", help="", type=str)
    parser.add_argument("--plot-dir", default="", help="", type=str)
    parser.add_argument("--model-name", default="", help="", type=str)
    parser.add_argument("--model-path", default="", help="", type=str)
    parser.add_argument("--plot-model-name", default="ParticleNet-PyG-EF", help="", type=str)
    parser.add_argument(
        "--data-config-dir", default="/hwwtaggervol/weaver/data/new_ntuples/", help="", type=str
    )
    parser.add_argument(
        "--data-config", default="ak15_4q3q_flat_eta_genHm_pt300_cw_8_2", help="", type=str
    )
    parser.add_argument(
        "--network-config", default="particle_net_pf_sv_4_layers_pyg_ef", help="", type=str
    )
    parser.add_argument(
        "--train-samples",
        default=[
            "QCD*",
            "BulkGraviton*",
        ],
        help="samples",
        nargs="*",
    )
    parser.add_argument(
        "--inference-samples",
        default=[
            "QCD*",
            "BulkGraviton*",
            "GluGluToBulkGravitonToHHTo4W_JHUGen_M-2500_narrow",
            "GluGluToHHTobbVV_node_cHHH1_pn4q",
            "jhu_HHbbWW",
        ],
        help="samples",
        nargs="*",
    )
    parser.add_argument(
        "--inference-sample-names",
        default=["qcd", "bulkg_hflat", "bulkg_hsm", "HHbbVV", "jhu_HHbbWW"],
        help="samples",
        nargs="*",
    )
    parser.add_argument(
        "--plot-samples",
        default=["bulkg_hflat", "bulkg_hsm", "HHbbVV", "jhu_HHbbWW"],
        help="samples",
        nargs="*",
    )
    parser.add_argument(
        "--plot-sample-names",
        default=["BulkGFlatHiggs", "BulkGSMHiggs", "PythiaHHbbVV", "JHUHHbbWW"],
        help="samples",
        nargs="*",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        help="overwrite old job",
        type=bool,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--make-train",
        default=True,
        help="",
        type=bool,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--run-train",
        default=False,
        help="",
        type=bool,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--make-inference",
        default=True,
        help="",
        type=bool,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--run-inference",
        default=False,
        help="",
        type=bool,
        action=BooleanOptionalAction,
    )
    args = parser.parse_args()

    main(args)
