import os, itertools
import numpy as np
from itertools import cycle

from contextlib import redirect_stdout
from functools import reduce
def get_max_it(release, dataset, batch, experiment):
    if release:
        project_path = os.getcwd() + "/"
        logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)
        #print ("max_iter" + "_" + experiment + ".npy")
        #if os.path.isfile(logs_path + "max_iter" + "_" + experiment + ".npy"):
        return int(np.load(logs_path + 'max_iter' + '_' + experiment + ".npy"))
    else:
           
        max_it_dict = {"quad_1": dict(zip([5, 25, 50, 100], [1000000, 1000000, 500000, 500000])),
                   "quad_2": dict(zip([5, 25, 50, 100], [4207505, 742505, 2970051, 1980051])),
                   "mushrooms": dict(zip([5, 25, 50, 100], [4500000, 2160000, 1910000, 1790000]))
                   }
        return max_it_dict[dataset][batch]

project_path = os.getcwd() + "/"

experiment_name_ar = ["NSYNC", "LS"]
#experiment_name_ar = ["NSYNC"]


#experiment_name_ar = ["NSYNC_", "LS_time_test","NSYNC", "LS"]
experiment_name_ar = ["LS_continue"]
#experiment_name_ar = ["LS"]#
#experiment_name_ar = ["NSYNC_check_last_loss"]
experiment_name_ar = ["LS_averaging"]
#experiment_name_ar = ["LS_time_test", "NSYNC_time_test"]


dataset_ar = ["mushrooms"]
#dataset_ar = ["quad_1", "quad_2", "mushrooms"]
#dataset_ar = ["quad_2"]


sampling_kind_ar = ["uniform", "importance_cd", "importance_acd"] 
sampling_kind_ar = ["uniform"]


loss_func_ar = ["log-reg"]
#loss_func_ar = ["quadratic", "log-reg"]
#loss_func_ar = ["quadratic"]



step_type_ar =["optimal","uniform", "arbitrary"]
#step_type_ar =["arbitrary"]
#step_type_ar =["optimal" ]

batch_ar = [5, 25, 50, 100]
#batch_ar = [25, 50, 100]
batch_ar = [5]
n = 1
launch_number_ar = np.arange (1, n+1)

n_starts = 5

n_starts_str = ""
tol = 1e-12

release = False
generate = False

scaled = "non-scaled"

parallel_setting = False


if not parallel_setting:
    shell_script_name = "m_avg"

    with open(shell_script_name + ".sh", 'w') as f:
        with redirect_stdout(f):
            #print ("%%bash")
            print ("#!/bin/bash")
            datas = [experiment_name_ar, dataset_ar, loss_func_ar, sampling_kind_ar, batch_ar]

            total_n = reduce(lambda x,y: x * y, list(map(len, datas)))

            if generate:
                for i,(dataset, loss_func) in enumerate (itertools.product (dataset_ar, loss_func_ar)):
                    if (loss_func == "log-reg" and dataset in ["quad_1", "quad_2"]) or (loss_func == "quadratic" and dataset == "mushrooms"):
                        continue
                    print ("python3 generate_data.py --dataset {0}  --loss_func {1} ".format(dataset, loss_func))
            for launch_number in launch_number_ar:
                for i, (experiment_name, dataset, loss_func, sampling_kind, batch ) in \
                enumerate (itertools.product (experiment_name_ar, dataset_ar, loss_func_ar, sampling_kind_ar, batch_ar)):

                    if (loss_func == "log-reg" and dataset in ["quad_1", "quad_2"]) or (loss_func == "quadratic" and dataset == "mushrooms"):
                        continue
                    if experiment_name[-9:] == "averaging" or experiment_name[-8:] == "max_iter":
                        n_starts_str = " --n_starts {0}".format(n_starts)

                    if (experiment_name[:2] == "LS"):

                        print ("python3 {6}.py --dataset {0}  --loss_func {1} --sampling_kind {2} --batch {3} --max_it {4} --scaled {5} --launch_number {7} --tol {8}".
                    format(dataset, loss_func, sampling_kind, batch, get_max_it(release, dataset, batch, '{0}_{1}_{2}_{3}'.format(experiment_name[:2]+"_"+scaled, loss_func, sampling_kind, batch)),scaled, experiment_name, launch_number, tol) + n_starts_str)

                    for step_type in step_type_ar:
                        if (experiment_name[:5] == "NSYNC"):
                            if (sampling_kind == "uniform" and (step_type not in ["uniform", "optimal"] )):
                                continue
                            if ((sampling_kind == "importance_cd" or sampling_kind == "importance_acd") and (step_type not in ["arbitrary", "optimal"])):
                                continue

                            print ("python3 {7}.py --dataset {0}  --loss_func {1} --sampling_kind {2} --batch {3} --step_type {6} --max_it {4} --scaled {5} --launch_number {8} --tol {9}".
                        format(dataset, loss_func, sampling_kind, batch, get_max_it(release, dataset, batch, '{0}_{1}_{2}_{3}_{4}'.format(experiment_name[:5]+"_"+scaled, loss_func, sampling_kind,
                                                                      step_type, batch)),scaled, step_type, experiment_name, launch_number, tol) + n_starts_str)

                    #show the script
    f = open(shell_script_name + ".sh", 'r')
    print (f.read())
    f.close()

if parallel_setting:
    shell_script_name = "m_c"
    sbatch_script = shell_script_name + "_sbatch_script"
    with open(sbatch_script + ".sh", 'w') as f1:
        f1.write("#!/bin/bash\n")
        for launch_number in launch_number_ar:
            for i, (experiment_name, dataset, loss_func, sampling_kind, batch ) in \
            enumerate (itertools.product (experiment_name_ar, dataset_ar, loss_func_ar, sampling_kind_ar, batch_ar)):
                #name = shell_script_name + "_{0}.sh".format(i)
                with open(shell_script_name + "_{0}_{1}.sh".format(i, launch_number), 'w') as f:
                    with redirect_stdout(f):
                        #print ("%%bash")
                        print ("#!/bin/bash")
                        if (loss_func == "log-reg" and dataset in ["quad_1", "quad_2"]) or (loss_func == "quadratic" and dataset == "mushrooms"):
                            continue

                        if (experiment_name[:2] == "LS"):
                            print ("python3 {6}.py --dataset {0}  --loss_func {1} --sampling_kind {2} --batch {3} --max_it {4} --scaled {5} --launch_number {7} --tol {8}".
                        format(dataset, loss_func, sampling_kind, batch, get_max_it(release, dataset, batch, '{0}_{1}_{2}_{3}'.format(experiment_name[:2]+"_"+scaled, loss_func, sampling_kind, batch)),scaled, experiment_name, launch_number, tol))

                        for step_type in step_type_ar:
                            if (experiment_name[:5] == "NSYNC"):
                                if (sampling_kind == "uniform" and (step_type not in ["uniform", "optimal"] )):
                                    continue
                                if ((sampling_kind == "importance_cd" or sampling_kind == "importance_acd") and (step_type not in ["arbitrary", "optimal"])):
                                    continue

                                print ("python3 {7}.py --dataset {0}  --loss_func {1} --sampling_kind {2} --batch {3} --step_type {6} --max_it {4} --scaled {5} --launch_number {8} --tol {9}".
                            format(dataset, loss_func, sampling_kind, batch, get_max_it(release, dataset, batch, '{0}_{1}_{2}_{3}_{4}'.format(experiment_name[:5]+"_"+scaled, loss_func, sampling_kind,
                                                                      step_type, batch)),scaled, step_type, experiment_name, launch_number, tol))

                            #show the script
                f = open(shell_script_name + "_{0}_{1}.sh".format(i, launch_number), 'r')
                print (f.read())
                f.close()
                f1.write ("sbatch --cpus-per-task=16 --mem-per-cpu=1 --exclusive /home/common/sokolov.ia/LS4/{0}\n".format(shell_script_name + "_{0}_{1}.sh".format(i, launch_number)))
