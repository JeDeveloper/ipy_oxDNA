from ..simulation.oxdna_simulation import Simulation
import numpy as np
import pickle
import argparse
import os
import multiprocessing as mp
import queue
import nvidia_smi
import warnings
import sys
import subprocess as sp
import timeit
import signal
from time import sleep


#TODO: Think of a way to make my umbrella sampling and generate replica distributed.


# I want to make it so that I can run a single file and it will be able to run
# multisystem replicas for me distributed across multiple nodes. The first question is what do I alreay have
# Currently I have a single run.py file that can run replicas distributaed across a single node.
# Currently I sbatch a single batch file on my slurm cluster and it runs the replicas for me.
# The first question I need to awnser is should I use sbatch to distribute or should I run a script on login node
# with a python script that will distribute the replicas. I think I should use the login node to distribute the replicas


# This means that I need a python script that will have parameters:
# 1. The number of gpus I want to use
# 2. the number of cpus per gpu
# 3. The number of GPUs per sbatch job


# The starting point always has to be the sim_list
# I will run a python file that will create run.py files?
# Trying to think of this in reverse...
# At the end of the day I will need my allocation to be running a exacuatable.
# This exacutable will be a python code. For now let us assume this will be a python script

# The minal python script that takes in a sim_list and runs it would ideally already be build, and it would then just need to be queue
# This would give some generality

# That would me I would have a python script with a function that builds the sim_list and the sims.
# That python function would return the sim_list

# I would then have a premade python function that would take in a sim_list and run it, it can have hyperparameters for the worker_manager
# the batch script will run a single function that will have all the informaiton it needs to run the simulations
# the exacuateble would be like


# The question is where will it get sim_list_slice from?
# On the highest level end, I have a script that has a function to make the sim_lists, build all the sims,
# create a list of sim_list_slices, create the sbatch files, create the rank_n.py files, and then run the sbatch files
# How can I make this as general as possible?

# I have a directory where the main python script
# I can save each sim_list_slice as a pickle file in the rank subdirectory


class SimulationManager:
    manager: mp.Manager
    # todo: replace w/ generator?
    sim_queue: queue.Queue[Simulation]
    process_queue: queue.Queue
    gpu_memory_queue: queue.Queue
    terminate_queue: queue.Queue
    worker_process_list: list

    warnings.filterwarnings(
        "ignore",
        "os.fork\\(\\) was called\\. os\\.fork\\(\\) is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock\\.",
        RuntimeWarning
    )

    """ In conjunction with nvidia-cuda-mps-control, allocate simulations to avalible cpus and gpus."""

    def __init__(self, n_processes=None):
        """
        Initalize the multiprocessing queues used to manage simulation allocation.
        
        The sim_queue utilizes a single process to store all queued simulations and allocates simulations to cpus.
        The process_queue manages the number of processes/cpus avalible to be sent to gpu memory.
        gpu_memory_queue is used to block the process_queue from sending simulations to gpu memory if memoy is near full.
        
        Parameters:
            n_processes (int): number of processes/cpus avalible to run oxDNA simulations in parallel.
        """
        if n_processes is None:
            self.n_processes = self.get_number_of_processes()
        else:
            if type(n_processes) is not int:
                raise ValueError('n_processes must be an integer')
            self.n_processes = n_processes
        self.manager = mp.Manager()
        self.sim_queue = self.manager.Queue()
        self.process_queue = self.manager.Queue(self.n_processes)
        self.gpu_memory_queue = self.manager.Queue(1)
        self.terminate_queue = self.manager.Queue(1)
        self.worker_process_list = self.manager.list()

    def get_number_of_processes(self):
        try:
            # Try using os.sched_getaffinity() available on some Unix systems
            if sys.platform.startswith('linux'):
                return len(os.sched_getaffinity(0))
            else:
                # Fallback to multiprocessing.cpu_count() which works cross-platform
                return mp.cpu_count()
        except Exception as e:
            # Handle possible exceptions (e.g., no access to CPU info)
            print(f"Failed to determine the number of CPUs: {e}")
            return 1  # Safe fallback if number of CPUs can't be determined

    def gpu_resources(self) -> tuple[np.ndarray, int]:
        """ Method to probe the number and current avalible memory of gpus."""
        avalible_memory = []
        try:
            nvidia_smi.nvmlInit()
        except Exception as e:
            print('nvidia-smi not avalible, ensure you have a cuda enabled GPU')
            raise e
        NUMBER_OF_GPU = nvidia_smi.nvmlDeviceGetCount()
        for i in range(NUMBER_OF_GPU):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            avalible_memory.append(self._bytes_to_megabytes(info.total) - self._bytes_to_megabytes(info.used))
        gpu_most_aval_mem_free = max(avalible_memory)
        gpu_most_aval_mem_free_idx = avalible_memory.index(gpu_most_aval_mem_free)
        return np.round(gpu_most_aval_mem_free, 2), gpu_most_aval_mem_free_idx

    def _bytes_to_megabytes(self, byte):
        # TODO: make this not a class method?
        return byte / 1048576

    def get_sim_mem(self, sim: Simulation, gpu_idx):
        """
        Returns the amount of simulation memory requried to run an oxDNA simulation.
        Note: A process running a simulation will need more memory then just required for the simulation.
              Most likely overhead from nvidia-cuda-mps-server
        
        Parameters:
            sim (Simulation): Simulation object to probe the required memory of.
            gpu_idx: depreciated
        """
        steps = sim.input.input_dict['steps']
        last_conf_file = sim.input.input_dict['lastconf_file']
        sim.input_file({'lastconf_file': os.devnull, 'steps': '0'})
        sim.oxpy_run.run(subprocess=False, verbose=False, log=False)
        sim.input_file({'lastconf_file': f'{last_conf_file}', 'steps': f'{steps}'})

        err_split = sim.oxpy_run.sim_output[1].split()
        try:
            mem = err_split.index('memory:')
            sim_mem = err_split[mem + 1]
        except Exception as e:
            print("Unable to determine CUDA memory usage")
            print(traceback.format_exc())
            raise e

        return float(sim_mem)

    def queue_sim(self, sim: Simulation, continue_run=False):
        """ 
        Add simulation object to the queue of all simulations.
        
        Parameters:
            sim (Simulation): Simulation to be queued.
            continue_run (bool): If true, continue previously run oxDNA simulation
        """
        if continue_run is not False:
            sim.input_file({"conf_file": sim.sim_files.last_conf, "refresh_vel": "0",
                            "restart_step_counter": "0", "steps": f"{continue_run}"})
        self.sim_queue.put(sim)

    def worker_manager(self, gpu_mem_block=True, custom_observables=None, run_when_failed=False, cpu_run=False):
        """
        Head process in charge of allocating queued simulations to processes and gpu memory.
        """
        tic = timeit.default_timer()
        if cpu_run is True:
            gpu_mem_block = False
        self.custom_observables = custom_observables
        # as long as there are simulations in the queue
        while not self.sim_queue.empty():
            # get simulation from queue
            if self.terminate_queue.empty():
                pass
            else:
                if run_when_failed is False:
                    for worker_process in self.worker_process_list:
                        os.kill(worker_process, signal.SIGTERM)
                    return print(self.terminate_queue.get())
                else:
                    print(self.terminate_queue.get())
            self.process_queue.put('Simulation worker finished')
            sim = self.sim_queue.get()
            gpu_idx = None
            if cpu_run is False:
                free_gpu_memory, gpu_idx = self.gpu_resources()
                sim.input_file({'CUDA_device': str(gpu_idx)})
            p = mp.Process(target=self.worker_job, args=(sim, gpu_idx,), kwargs={'gpu_mem_block': gpu_mem_block})
            p.start()
            self.worker_process_list.append(p.pid)
            if gpu_mem_block is True:
                sim_mem = self.gpu_memory_queue.get()
                if free_gpu_memory < (3 * sim_mem):
                    wait_for_gpu_memory = True
                    while wait_for_gpu_memory == True:
                        if free_gpu_memory < (3 * sim_mem):
                            free_gpu_memory, gpu_idx = self.gpu_resources()
                            sleep(5)
                        else:
                            print('gpu memory freed')
                            wait_for_gpu_memory = False
            else:
                if cpu_run is False:
                    sleep(0.5)
                elif cpu_run is True:
                    sleep(0.1)

        while not self.process_queue.empty():
            sleep(10)
        toc = timeit.default_timer()
        print(f'All queued simulations finished in: {toc - tic}')

    def worker_job(self, sim: Simulation, gpu_idx: int, gpu_mem_block: bool = True):
        """ Run an allocated oxDNA simulation"""
        if gpu_mem_block is True:
            sim_mem = self.get_sim_mem(sim, gpu_idx)
            self.gpu_memory_queue.put(sim_mem)

        sim.oxpy_run.run(subprocess=False, custom_observables=self.custom_observables)
        if sim.oxpy_run.error_message is not None:
            self.terminate_queue.put(
                f'Simulation exception encountered in {sim.sim_dir}:\n{sim.oxpy_run.error_message}')
        self.process_queue.get()

    def run(self, join=False, gpu_mem_block=True, custom_observables=None, run_when_failed=False,
            cpu_run=False):
        """
        Run the worker manager in a subprocess
        todo: ...logging?
        """
        print('spawning')
        if cpu_run is True:
            gpu_mem_block = False

        p = mp.Process(target=self.worker_manager, args=(),
                       kwargs={'gpu_mem_block': gpu_mem_block, 'custom_observables': custom_observables,
                               'run_when_failed': run_when_failed, 'cpu_run': cpu_run})
        self.manager_process = p
        p.start()
        if join == True:
            p.join()

    def terminate_all(self, ):
        try:
            self.manager_process.terminate()
        except:
            pass
        for process in self.worker_process_list:
            try:
                os.kill(process, signal.SIGTERM)
            except:
                pass
        self.worker_process_list[:] = []

    def start_nvidia_cuda_mps_control(self, pipe='$SLURM_TASK_PID'):
        """
        Begin nvidia-cuda-mps-server.
        
        Parameters:
            pipe (str): directory to pipe control server information to. Defaults to PID of a slurm allocation
        """
        with open('launch_mps.tmp', 'w') as f:
            f.write(f"""#!/bin/bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_{pipe};
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_{pipe};
mkdir -p $CUDA_MPS_PIPE_DIRECTORY;
mkdir -p $CUDA_MPS_LOG_DIRECTORY;
nvidia-cuda-mps-control -d"""
                    )
        os.system('chmod u+rx launch_mps.tmp')
        sp.call('./launch_mps.tmp')
        self.test_cuda_script()
        os.system('./test_script')
        os.system('echo $CUDA_MPS_PIPE_DIRECTORY')

    #         os.system(f"""export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_{pipe};
    # export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_{pipe};
    # mkdir -p $CUDA_MPS_PIPE_DIRECTORY;
    # mkdir -p $CUDA_MPS_LOG_DIRECTORY;
    # nvidia-cuda-mps-control -d;""")

    def restart_nvidia_cuda_mps_control(self):
        os.system("""echo quit | nvidia-cuda-mps-control""")
        sleep(0.5)
        self.start_nvidia_cuda_mps_control()

    def test_cuda_script(self):
        script = """#include <stdio.h>

#define N 2

__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = 2*a[i];
    }
}

int main() {

    int ha[N], hb[N];

    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }


    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

    add<<<N, 1>>>(da, db);

    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);
    
        for (int i = 0; i<N; ++i) {
        printf("%d", hb[i]);
    }

    cudaFree(da);
    cudaFree(db);

    return 0;
}
"""
        with open('test_script.cu', 'w') as f:
            f.write(script)

        os.system('nvcc -o test_script test_script.cu')
        os.system('./test_script')


def run_sim_list_slice(sim_list_slice, continue_run):
    sim_manager = SimulationManager(n_processes=len(os.sched_getaffinity(0)))
    
    for sim in sim_list_slice:
        sim_manager.queue_sim(sim, continue_run=continue_run)
    sim_manager.worker_manager(gpu_mem_block=False)


def cli_parser(prog="distributed.py"):
    # A standard way to create and parse command line arguments.
    parser = argparse.ArgumentParser(prog = prog, description="Distributes a sim_list across an HPC.")
    parser.add_argument('-c', '--n_cpus_per_gpu', metavar='n_cpus_per_gpu', nargs=1, type=int, dest='n_cpus_per_gpu', help="The number of cpus per gpu")
    parser.add_argument('-g', '--n_gpus_per_sbatch', metavar='n_gpus_per_sbatch', nargs=1, type=int, dest='n_gpus_per_sbatch', help='The number of gpus per sbatch job')
    parser.add_argument('-d', '--distributed_directory', metavar='distributed_directory', nargs=1, type=str, dest='distributed_directory', help='The directory where the distributed files will be saved')
    parser.add_argument('-r', '--continue_run', metavar='continue_run', nargs=1, type=float, dest='continue_run', help='Whether to continue the run')

    return parser


def distribute_sim_list_across_nodes(sim_list):
    parser = cli_parser()
    args = parser.parse_args()
    
    continue_run = args.continue_run[0] if args.continue_run else False
    n_cpus_per_gpu = args.n_cpus_per_gpu[0] if args.n_cpus_per_gpu else 1
    n_gpus_per_sbatch = args.n_gpus_per_sbatch[0] if args.n_gpus_per_sbatch else 1
    distributed_directory = args.distributed_directory[0] if args.distributed_directory else f'{os.getcwd()}/distributed'
    distributed_directory = os.path.abspath(distributed_directory)
    sim_list_slices = create_sim_slices(sim_list, n_cpus_per_gpu, n_gpus_per_sbatch)
    
    build_distributed_files(distributed_directory, sim_list_slices, n_cpus_per_gpu, n_gpus_per_sbatch, continue_run)
    
    run_distributed_files(distributed_directory, sim_list_slices)
    
    return None
    
    
def create_sim_slices(sim_list, n_cpus_per_gpu, n_gpus_per_sbatch):
    n_sims = len(sim_list)
        
    # The total number of cpus will be:
    cpus_per_sbatch =  n_cpus_per_gpu * n_gpus_per_sbatch

    # The number of sbatch jobs will be equal to:
    n_sbatch_jobs =  np.ceil(n_sims / cpus_per_sbatch)
    
    # The total number of gpus will be:
    n_total_gpus = n_sbatch_jobs * n_gpus_per_sbatch
    
    sim_list_slices = np.array_split(sim_list, n_sbatch_jobs)

    return sim_list_slices

    
def build_distributed_files(distributed_directory, sim_list_slices, n_cpus_per_gpu, n_gpus_per_sbatch, continue_run):
    
    create_distributed_dirs(distributed_directory, sim_list_slices)
    
    pickle_sim_slices(distributed_directory, sim_list_slices)
    
    create_run_files(distributed_directory, sim_list_slices, continue_run)
    
    create_sbatch_files(distributed_directory, sim_list_slices, n_cpus_per_gpu, n_gpus_per_sbatch)
    
    return None


def run_distributed_files(distributed_directory, sim_list_slices):
    n_dirs = len(sim_list_slices)
    
    for rank in range(n_dirs):
        os.chdir(f'{distributed_directory}/{rank}_job/')
        os.system(f'sbatch {rank}_sbatch.sh')
    return None


def pickle_sim_slices(distributed_directory, sim_list_slices):
    for job_id, sim_list_slice in enumerate(sim_list_slices):
        save_dir = f'{distributed_directory}/{job_id}_job'
        write_pickle_sim_list(save_dir, sim_list_slice, job_id)
    return None


def create_distributed_dirs(distributed_directory, sim_list_slices):
    n_dirs = len(sim_list_slices)
    
    os.makedirs(distributed_directory, exist_ok=True)
    for rank in range(n_dirs):
        os.makedirs(f'{distributed_directory}/{rank}_job', exist_ok=True)
    
    return None
    
    
def create_run_files(distributed_directory, sim_list_slices, continue_run):
    n_dirs = len(sim_list_slices)
    
    file_contents = [f"""from ipy_oxdna.distributed import run_sim_list_slice, read_pickle_sim_list

job_id = {job_id}
sim_list_slice = read_pickle_sim_list(job_id)
run_sim_list_slice(sim_list_slice, {continue_run})
    """ for job_id in range(n_dirs)]
    
    for job_id, file_content in enumerate(file_contents):
        with open(f'{distributed_directory}/{job_id}_job/{job_id}_run.py', 'w') as f:
            f.write(file_content)


def create_sbatch_files(distributed_directory, sim_list_slices, n_cpus_per_gpu, n_gpus_per_sbatch):
    n_dirs = len(sim_list_slices)
    n_cpus_per_sbatch = int(n_cpus_per_gpu * n_gpus_per_sbatch)
    
    file_contents = [f"""#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -n {n_cpus_per_sbatch}           # number of cores 
#SBATCH -t 7-00:00:00   # time in d-hh:mm:ss
#SBATCH -G a100:{n_gpus_per_sbatch} 
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH --job-name="{job_id}_job"
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

module load mamba/latest
module load cuda-11.7.0-gcc-11.2.0
source activate oxdnapy12

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_$SLURM_TASK_PID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_$SLURM_TASK_PID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d

python3 {job_id}_run.py

    """ for job_id in range(n_dirs)]
    
    for job_id, file_content in enumerate(file_contents):
        with open(f'{distributed_directory}/{job_id}_job/{job_id}_sbatch.sh', 'w') as f:
            f.write(file_content)
    return None
 
    
def write_pickle_sim_list(save_dir, sim_list_slice, job_id):
    with open(f'{save_dir}/{job_id}_sim_slice.pkl', 'wb') as f:
        pickle.dump(sim_list_slice, f)
    return None


def read_pickle_sim_list(job_id):
    with open(f'{job_id}_sim_slice.pkl', 'rb') as f:
        return pickle.load(f)
