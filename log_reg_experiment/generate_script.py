"""
(c) Igor Sokolov
https://github.com/mstkachev/Sirius_project_optimization

"""


LAUNCH_NUMBERS = 2
launch_number_ar = np.arange (1, LAUNCH_NUMBERS)

project_path = os.getcwd() + "/"
experiment_name_ar = ["local_spider_homo"]
dataset_ar = ["mushrooms"]

batch_ar = [20, 50]
batch_ar = [20]

num_workers_ar = [20, 50, 100]
#num_workers_ar = [20]

num_local_steps_ar = [10, 20, 30, 50]
#num_local_steps_ar = [10]

tol = 1e-12
release = True
max_epochs = int(1e5)
max_num_comm = 200
is_continue = 0


shell_script_name = "local_spider_homo_a9a"

with open(shell_script_name + ".sh", 'w') as f:
    with redirect_stdout(f):
        #print ("%%bash")
        print ("#!/bin/bash")

        for launch_number in launch_number_ar:
            for i, (experiment_name, dataset, num_workers, num_local_steps, batch_size ) in \
            enumerate (itertools.product (experiment_name_ar, dataset_ar, num_workers_ar, num_local_steps_ar, batch_ar)):
                print ("python3 {0}.py --dataset {1} --num_workers {2} --num_local_steps {3} --batch_size {4} --continue {5} --max_epochs {6} --max_num_comm {7}\
                       --launch_number {8} --tol {9}".
                format(experiment_name, dataset, num_workers, num_local_steps, batch_size , is_continue, max_epochs, max_num_comm, launch_number, tol))

                #show the script
f = open(shell_script_name + ".sh", 'r')
print (f.read())
f.close()

