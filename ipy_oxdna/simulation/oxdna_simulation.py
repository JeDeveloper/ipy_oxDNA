from __future__ import annotations

import abc
import errno
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union
import os
import numpy as np
import shutil
from json import dumps, loads, dump, load
import oxpy
import multiprocessing as mp
import py
from oxDNA_analysis_tools.UTILS.data_structures import TrajInfo, TopInfo
from oxDNA_analysis_tools.UTILS.get_confs import Configuration
from oxDNA_analysis_tools.UTILS.oxview import oxdna_conf
from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs

import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import timeit
import traceback
import json

from .defaults import DefaultInput, SEQ_DEP_PARAMS, NA_PARAMETERS, RNA_PARAMETERS, get_default_input
from .force import Force
from .observable import Observable

# import cupy

"""
interface for file_dir and sim_dir methods
Inheriting classes can either define protected variables to store
values for file_dir and sim_dir or they can refer to other class member vars
"""
class SimDirInfo(abc.ABC):
    @abstractmethod
    def get_file_dir(self) -> Path:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def set_file_dir(self, p: Union[Path, str]):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_sim_dir(self) -> Path:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def set_sim_dir(self, p: Union[Path, str]):
        raise NotImplementedError("Subclasses must implement this method")

    # Properties for file_dir and sim_dir
    # all inheriting classes must include the following:
    '''
    file_dir = property(get_file_dir, set_file_dir)
    sim_dir = property(get_sim_dir, set_sim_dir)
    '''

class Simulation(SimDirInfo):

    _file_dir: Path
    _sim_dir: Path
    sim_files: SimFiles
    build_sim: BuildSimulation
    input: Input
    analysis: Analysis
    protein: Protein
    oxpy_run: OxpyRun
    oat: OxdnaAnalysisTools

    """
    Used to interactivly interface and run an oxDNA simulation.
    
    Parameters:
        file_dir (str): Path to directory containing inital oxDNA dat and top files.
        
        sim_dir (str): Path to directory where a simulation will be run using inital files.
    """

    def __init__(self,
                 file_dir: Union[str, Path],
                 sim_dir: Union[str, Path, None] = None,
                 input_file_params: dict = {}):
        """
        Instance lower level class objects used to compose the Simulation class features.
        """

        # handle alternate param types for file_dir
        if isinstance(file_dir, Path):
            self.file_dir = file_dir
        elif isinstance(file_dir, str):
            self.file_dir = Path(file_dir)
        else:
            raise ValueError(f"Invalid type {type(file_dir)} for parameter file_dir")

        # handle alternate param types for sim_dir
        if sim_dir is None:  # if no sim dir is provided, use file dir
            self.sim_dir = self.file_dir
        elif isinstance(sim_dir, str):
            self.sim_dir = Path(sim_dir)
        elif isinstance(sim_dir, Path):
            self.sim_dir = sim_dir
        else:
            raise ValueError(f"Invalid type {type(sim_dir)} for parameter sim_dir")
        # tolerate sim_dir not existing, we can create it later

        self.sim_files = SimFiles(self)
        self.build_sim = BuildSimulation(self)
        self.input = Input(self)
        self.analysis = Analysis(self)
        self.protein = Protein(self)
        self.oxpy_run = OxpyRun(self)
        self.oat = OxdnaAnalysisTools(self)
        self.sequence_dependant = SequenceDependant(self)


    def build(self, clean_build=False):
        """
        Build dat, top, and input files in simulation directory.
        
        Parameters:
            clean_build (bool): If sim_dir already exsists, remove it and then rebuild sim_dir
        """
        if self.sim_dir.exists():
            # print(f'Exisisting simulation files in {self.sim_dir.split("/")[-1]}')
            if clean_build == True:
                # TODO: support for non-cli contexts
                answer = input('Are you sure you want to delete all simulation files? '
                               'Type y/yes to continue or anything else to return (use clean_build=str(force) to skip this message)')
                if (answer == 'y') or (answer == 'yes'):
                    shutil.rmtree(f'{self.sim_dir}/')
                    self.build_sim.force_cache = None
                else:
                    print('Remove optional argument clean_build and rerun to continue')
                    return None
            elif clean_build == 'force':
                shutil.rmtree(self.sim_dir)
                self.build_sim.force_cache = None
            elif clean_build == False:
                print(
                    'The simulation directory already exists, if you wish to write over the directory set clean_build=force')
                return None
        self.build_sim.build_sim_dir()
        self.build_sim.build_dat_top()
        self.build_sim.build_input()

        self.sim_files.parse_current_files()


    def input_file(self, parameters):
        """
        Modify the parameters of the oxDNA input file, all parameters are avalible at https://lorenzo-rovigatti.github.io/oxDNA/input.html
        
        Parameters:
            parameters (dict): dictonary of oxDNA input file parameters
        """
        self.input.modify_input(parameters)


    def add_protein_par(self):
        """
        Add a parfile from file_dir to sim_dir and add file name to input file
        """
        self.build_sim.build_par()
        self.protein.par_input()


    def add_force_file(self):
        """
        Add a external force file from file_dir to sim_dir and add file name to input
        """
        self.build_sim.get_force_file()
        self.build_sim.build_force_from_file()
        self.input_file({'external_forces': '1'})


    def add_force(self, force_js):
        """
        Add an external force to the simulation.
        
        Parameters:
            force_js (ipy_oxdna.force.Force): A force object, essentially a dictonary, specifying the external force parameters.
        """
        if not os.path.exists(os.path.join(self.sim_dir, "forces.json")):
            self.input_file({'external_forces': '1'})
        self.build_sim.build_force(force_js)


    def add_observable(self, observable_js: Union[Observable, dict[str, Any]]):
        """
        Add an observable that will be saved as a text file to the simulation.
        
        Parameters:
            observable_js (ipy_oxdna.observable.Observable): A observable object, essentially a dictonary, specifying the observable parameters.
        """
        if isinstance(observable_js, dict):
            if not os.path.exists(os.path.join(self.sim_dir, "observables.json")):
                self.input_file({'observables_file': 'observables.json'})
            self.build_sim.build_observable(observable_js)
        else:
            self.add_observable(observable_js.export())


    def slurm_run(self, run_file, job_name='oxDNA'):
        """
        Write a provided sbatch run file to the simulation directory.
        
        Parameters:
            run_file (str): Path to the provided sbatch run file.
            job_name (str): Name of the sbatch job.
        """
        self.sim_files.run_file = os.path.abspath(os.path.join(self.sim_dir, run_file))
        self.slurm_run = SlurmRun(self.sim_dir, run_file, job_name)


    def make_sequence_dependant(self):
        """ Add a sequence dependant file to simulation directory and modify input file to use it."""
        self.sequence_dependant.make_sim_sequence_dependant()


    def get_file_dir(self) -> Path:
        return self._file_dir


    def set_file_dir(self, p: Union[Path, str]):
        if isinstance(p, Path):
            # assert p.exists(), f"Cannot make a simulation from non-existing file dir {str(p)}"
            self._file_dir = p
        else:
            self.set_file_dir(Path(p))


    def get_sim_dir(self) -> Path:
        return self._sim_dir


    def set_sim_dir(self, p: Union[Path, str]):
        if isinstance(p, Path):
            self._sim_dir = p
        else:
            self.set_sim_dir(Path(p))
            

    def pickle_sim(self):
        """ Pickle the simulation object to a file."""
        with open(f'{self.sim_dir}/sim.pkl', 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filename):
        """ Read a pickled simulation object from a file."""
        with open(filename, 'rb') as f:
            sim = pickle.load(f)
        return sim

    file_dir = property(get_file_dir, set_file_dir)
    sim_dir = property(get_sim_dir, set_sim_dir)

class SimulationComponent(SimDirInfo, ABC):
    """
    abstract class for a component of a simulation object
    """
    sim: Simulation

    def __init__(self, sim: Simulation):
        self.sim = sim

    # override methods from SimDirInfo to invoke Simulation object
    # to mimimize potential issues w/ same thing stored in different place
    def get_file_dir(self) -> Path:
        return self.sim.file_dir

    def set_file_dir(self, p: Union[Path, str]):
        if isinstance(p, Path):
            self.sim.file_dir = p
        else:
            self.set_file_dir(Path(p))

    def get_sim_dir(self) -> Path:
        return self.sim.sim_dir

    def set_sim_dir(self, p: Union[Path, str]):
        if isinstance(p, Path):
            self.sim.sim_dir = p
        else:
            self.set_sim_dir(Path(p))

    file_dir = property(get_file_dir, set_file_dir)
    sim_dir = property(get_sim_dir, set_sim_dir)

class Protein(SimulationComponent):
    """
    Methods used to enable anm simulations with proteins
    """

    def par_input(self):
        self.sim.input_file({
            'interaction_type': 'DNANM',
            'parfile': self.sim.build_sim.par
        })


class BuildSimulation(SimulationComponent):
    force: Force
    force_cache: Any
    par: Any
    force_file: Path
    is_file: bool

    # names of top and conf file in file directory
    top_file_name: str
    conf_file_name: str

    # dict which maps names of file in file directory to sim directory
    name_mapper: dict[str, str]

    """ Methods used to create/build oxDNA simulations."""

    def __init__(self, sim: Simulation):
        """ Initalize access to simulation information"""
        SimulationComponent.__init__(self, sim)
        self.force = Force()
        self.force_cache = None

        self.name_mapper = {
            "conf_file": "init.dat"  # default-case: rename last_conf to init
        }
        self.top_file_name = None
        self.conf_file_name = None

    def build_sim_dir(self):
        """Make the simulation directory"""
        if not self.sim.sim_dir.exists():
            os.makedirs(self.sim.sim_dir)

    def build_dat_top(self):
        """
        Write intial conf and toplogy to simulation directory
        """
        # find file-directory top and dat
        if self.top_file_name is None:
            self.top_file_name = find_top_file(self.file_dir, self.sim).name
        if self.conf_file_name is None:
            self.conf_file_name = find_conf_file(self.file_dir, self.sim).name

        if "conf_file" in self.name_mapper:
            self.sim.input.set_conf_file(self.name_mapper["conf_file"])
        else:
            self.sim.input.set_conf_file(self.conf_file_name)
        if "topology" in self.name_mapper:
            self.sim.input.set_top_file(self.name_mapper["topology"])
        else:
            self.sim.input.set_top_file(self.top_file_name)

        # copy dat file to sim directory
        assert (self.file_dir / self.conf_file_name).exists()
        shutil.copy(self.file_dir / self.conf_file_name,
                    self.sim.sim_dir)
        shutil.move(self.sim.sim_dir / self.conf_file_name,
                    self.sim.sim_dir / self.sim.input.get_conf_file())

        # copy top file to sim directory
        assert (self.file_dir / self.top_file_name).exists()
        shutil.copy(self.file_dir / self.top_file_name,
                    self.sim.sim_dir)
        shutil.move(self.sim.sim_dir / self.top_file_name,
                    self.sim.sim_dir / self.sim.input.get_top_file())

    def list_file_dir(self) -> list[str]:
        """
        Returns: a list of files in the data source directory
        """
        return os.listdir(self.sim.file_dir)

    def build_input(self, production=False):
        """Calls a methods from the Input class which writes a oxDNA input file in plain text and json"""
        self.sim.input.initalize_input()
        self.sim.input.write_input(production=production)

    def get_par(self):
        """
        what does "par" mean
        """
        files = self.list_file_dir()
        self.par = [file for file in files if file.endswith('.par')][0]

    def build_par(self):
        """
        what does "par" mean
        """
        self.get_par()
        shutil.copy(os.path.join(self.sim.file_dir, self.par), self.sim.sim_dir)

    def get_force_file(self):
        files = self.list_file_dir()
        force_file = [file for file in files if file.endswith('.txt')][0]
        if len(force_file) > 1:
            force_file = [file for file in files if file.endswith('force.txt')][0]
        self.force_file = self.file_dir / force_file

    def build_force_from_file(self):
        forces = []
        shutil.copy(self.force_file, self.sim.sim_dir)
        with open(self.force_file, 'r') as f:
            lines = f.readlines()

        buffer = []
        for line in lines:
            if line.strip() == '{':
                buffer = []
            elif line.strip() == '}':
                force_dict = {}
                for entry in buffer:
                    key, value = [x.strip() for x in entry.split('=')]
                    force_dict[key] = value
                forces.append({'force': force_dict})
            else:
                if line.strip():  # Check if the line is not empty
                    buffer.append(line.strip())
        for force in forces:
            self.build_force(force)

    def build_force(self, force_js: dict):
        force_file_path = self.sim_dir / "forces.json"

        # Initialize the cache and create the file if it doesn't exist
        if self.force_cache is None:
            if not force_file_path.is_file():
                self.force_cache = {}
                with force_file_path.open("w") as f:
                    json.dump(self.force_cache, f, indent=4)
                self.is_empty = True  # Set the flag to True for a new file
            else:
                with force_file_path.open("r") as f:
                    self.force_cache = json.load(f)
                self.is_empty = not bool(self.force_cache)  # Set the flag based on the cache

        # Check for duplicates in the cache
        for force in list(self.force_cache.values()):
            # TODO: CLEAN UP THIS LINE
            if list(force.values())[1] == list(list(force_js.values())[0].values())[1]:
                return

        # Add the new force to the cache
        new_key = f'force_{len(self.force_cache)}'
        self.force_cache[new_key] = force_js['force']

        # Append the new force to the existing JSON file
        self.append_to_json_file(str(force_file_path),
                                 new_key,
                                 force_js['force'],
                                 self.is_empty)

        self.is_empty = False  # Update the flag

    def append_to_json_file(self,
                            file_path: str,
                            new_entry_key: str,
                            new_entry_value: Any,
                            is_empty: bool):
        with open(file_path, 'rb+') as f:
            f.seek(-1, os.SEEK_END)  # Go to the last character of the file
            f.truncate()  # Remove the last character (should be the closing brace)

            if not is_empty:
                f.write(b',\n')  # Only add a comma if the JSON object is not empty

            new_entry_str = f'    "{new_entry_key}": {json.dumps(new_entry_value, indent=4)}\n}}'
            f.write(new_entry_str.encode('utf-8'))

    def build_observable(self, observable_js: dict, one_out_file=False):
        """
        Write observable file is one does not exist. If a observable file exists add additional observables to the file.
        
        Parameters:
            observable_js (dict): observable dictornary obtained from the Observable class methods
        """
        if not os.path.exists(os.path.join(self.sim.sim_dir, "observables.json")):
            with open(os.path.join(self.sim.sim_dir, "observables.json"), 'w') as f:
                f.write(dumps(observable_js, indent=4))
        else:
            with open(os.path.join(self.sim.sim_dir, "observables.json"), 'r') as f:
                read_observable_js = loads(f.read())
                multi_col = False
                for observable in list(read_observable_js.values()):
                    if list(observable.values())[1] == list(list(observable_js.values())[0].values())[1]:
                        read_observable_js['output']['cols'].append(observable_js['output']['cols'][0])
                        multi_col = True
                if not multi_col:
                    read_observable_js[f'output_{len(list(read_observable_js.keys()))}'] = read_observable_js['output']
                    del read_observable_js['output']
                    read_observable_js.update(observable_js.items())
                with open(os.path.join(self.sim.sim_dir, "observables.json"), 'w') as f:
                    f.write(dumps(read_observable_js, indent=4))

    def build_hb_list_file(self, p1, p2):
        self.sim.sim_files.parse_current_files()
        column_names = ['strand', 'nucleotide', '3_prime', '5_prime']

        try:
            top = pd.read_csv(self.sim.sim_files.top, sep=' ', names=column_names).iloc[1:, :].reset_index(drop=True)
            top['index'] = top.index
            p1 = p1.split(',')
            p2 = p2.split(',')
            i = 1
            with open(os.path.join(self.sim.sim_dir, "hb_list.txt"), 'w') as f:
                f.write("{\norder_parameter = bond\nname = all_native_bonds\n")
            complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
            for nuc1 in p1:
                nuc1_data = top.iloc[int(nuc1)]
                nuc1_complement = complement[nuc1_data['nucleotide']]
                for nuc2 in p2:
                    nuc2_data = top.iloc[int(nuc2)]
                    if nuc2_data['nucleotide'] == nuc1_complement:
                        with open(os.path.join(self.sim.sim_dir, "hb_list.txt"), 'a') as f:
                            f.write(f'pair{i} = {nuc1}, {nuc2}\n')
                        i += 1
            with open(os.path.join(self.sim.sim_dir, "hb_list.txt"), 'a') as f:
                f.write("}\n")
            return None

        except:
            with open(self.sim.sim_files.force, 'r') as f:
                lines = f.readlines()
                lines = [int(line.strip().split()[1].replace('"', '')[:-1]) for line in lines if 'particle' in line]
                line_sets = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]
                line_sets = {tuple(sorted(t)) for t in line_sets}
            with open(os.path.join(self.sim.sim_dir, "hb_list.txt"), 'w') as f:
                f.write("{\norder_parameter = bond\nname = all_native_bonds\n")
                for idx, line_set in enumerate(line_sets):
                    f.write(f'pair{idx} = {line_set[0]}, {line_set[1]}\n')
                f.write("}\n")

            return None


class OxpyRun(SimulationComponent):
    # setup params
    sim_dir: str
    # run params
    subprocess: bool
    verbose: bool
    continue_run: bool
    log: Union[False, str]  # name of log file, or False if log is off
    join: bool
    custom_observables: bool
    sim_output: str
    sim_err: str
    process: mp.Process

    error_message: Union[None, str]

    """Automatically runs a built oxDNA simulation using oxpy within a subprocess"""

    def __init__(self, sim: Simulation):
        """ Initalize access to simulation inforamtion."""
        SimulationComponent.__init__(self, sim)
        self.my_obs = {}

    def run(self,
            subprocess=True,
            continue_run=False,
            verbose=True,
            log: Union[str, bool] = True,
            join=False,
            custom_observables=None):
        """ Run oxDNA simulation using oxpy in a subprocess.
        
        Parameters:
            subprocess (bool): If false run simulation in parent process (blocks process), if true spawn sim in child process.
            continue_run (number): If False overide previous simulation results. If True continue previous simulation run.
            verbose (bool): If true print directory of simulation when run.
            log (bool): If not False, print a log file to simulation directory. If True, the file will be auto-named to "log.log". otherwise it will be given the provided name
            join (bool): If true block main parent process until child process has terminated (simulation finished)
        """
        self.subprocess = subprocess
        self.verbose = verbose
        self.continue_run = continue_run
        if log is True:
            self.log = "log.log"
        else:
            self.log = log
        self.join = join
        self.custom_observables = custom_observables

        if self.verbose:
            print(f'Running: {self.sim.sim_dir}')

        if self.subprocess:
            self.spawn(self.run_complete)
        else:
            self.run_complete()

    def spawn(self, f, args=()):
        """Spawn subprocess"""
        p = mp.Process(target=f, args=args)
        p.start()
        if self.join:
            p.join()
        self.process = p
        self.sim.sim_files.parse_current_files()

    def run_complete(self):
        """Run an oxDNA simulation"""
        self.error_message = None
        tic = timeit.default_timer()
        # capture outputs
        capture = py.io.StdCaptureFD()
        if self.continue_run is not False:
            self.sim.input_file({"conf_file": self.sim.sim_files.last_conf.as_posix(), "refresh_vel": "0",
                                 "restart_step_counter": "0", "steps": f'{self.continue_run}'})
        start_dir = os.getcwd()
        os.chdir(self.sim.sim_dir)
        with open('input.json', 'r') as f:
            my_input = loads(f.read())
        with oxpy.Context():
            ox_input = oxpy.InputFile()
            for k, v in my_input.items():
                # todo: error-handling for vals that don't stringify nicely
                ox_input[k] = str(v)
            try:
                manager = oxpy.OxpyManager(ox_input)
                if hasattr(self.sim.sim_files, 'run_time_custom_observable'):
                    with open(self.sim.sim_files.run_time_custom_observable, 'r') as f:
                        self.my_obs = load(f)
                    for key, value in self.my_obs.items():
                        my_obs = [eval(observable_string, {"self": self}) for observable_string in value['observables']]
                        manager.add_output(key, print_every=value['print_every'], observables=my_obs)
                manager.run_complete()
                del manager
            except oxpy.OxDNAError as e:
                self.error_message = traceback.format_exc()

        # grab captured err and outputs
        self.sim_output, self.sim_err = capture.reset()
        toc = timeit.default_timer()
        if self.verbose:
            print(f'Run time: {toc - tic}')
            if self.error_message is not None:
                print(
                    f'Exception encountered in {self.sim.sim_dir}:\n{type(self.error_message).__name__}: {self.error_message}')
            else:
                print(f'Finished: {self.sim.sim_dir.parent}')

                # if log is set
                print(f'y: {self.sim.sim_dir.parent}')
        if self.log:
            with open('log.log', 'w') as f:
                f.write(self.sim_output)  # write output log
                f.write(self.sim_err)  # write error log
                f.write(f'Run time: {toc - tic}')  # write runtime
                if self.error_message is not None:
                    f.write(f'Exception: {self.error_message}')
        self.sim.sim_files.parse_current_files()
        os.chdir(start_dir)

    def cms_obs(self, *args, name=None, print_every=None):
        self.my_obs[name] = {'print_every': print_every, 'observables': []}
        for particle_indexes in args:
            self.my_obs[name]['observables'].append(f'self.cms_observables({particle_indexes})()')

        self.write_custom_observable()

    def write_custom_observable(self):
        with open(os.path.join(self.sim.sim_dir, "run_time_custom_observable.json"), 'w') as f:
            dump(self.my_obs, f, indent=4)

    def cms_observables(self, particle_indexes):
        class ComPositionObservable(oxpy.observables.BaseObservable):
            def get_output_string(self, curr_step):
                output_string = ''
                np_idx = [list(map(int, particle_idx.split(','))) for particle_idx in particle_indexes]
                particles = np.array(self.config_info.particles())
                indexed_particles = [particles[idx] for idx in np_idx]
                cupy_array = np.array(
                    [np.array([particle.pos for particle in particle_list]) for particle_list in indexed_particles],
                    dtype=np.float64)

                box_length = np.float64(self.config_info.box_sides[0])

                pos = np.zeros((cupy_array.shape[1], cupy_array.shape[2]), dtype=np.float64)
                np.subtract(cupy_array[0], cupy_array[1], out=pos, dtype=np.float64)

                pos = pos - box_length * np.round(pos / box_length)

                new_pos = np.linalg.norm(pos, axis=1)
                r0 = np.full(new_pos.shape, 1.2)
                gamma = 58.7
                shape = 1.2

                final = np.sum(1 / (1 + np.exp((new_pos - r0 * shape) * gamma))) / np.float64(new_pos.shape[0])

                output_string += f'{final} '
                return output_string

        return ComPositionObservable

    # def cms_observables(self, particle_indexes):
    #         class ComPositionObservable(oxpy.observables.BaseObservable):
    #             def get_output_string(self, curr_step):
    #                 output_string = ''
    #                 np_idx = [list(map(int, particle_idx.split(','))) for particle_idx in particle_indexes]
    #                 particles = np.array(self.config_info.particles())
    #                 indexed_particles = [particles[idx] for idx in np_idx]
    #                 cupy_array = np.array([np.array([particle.pos for particle in particle_list]) for particle_list in indexed_particles], dtype=object)
    #                 for array in cupy_array:
    #                     pos = np.mean(array, axis=0)
    #                     output_string += f'{pos[0]},{pos[1]},{pos[2]} '
    #                 return output_string
    #         return ComPositionObservable


class SlurmRun:
    """Using a user provided slurm run file, setup a slurm job to be run"""

    def __init__(self, sim_dir, run_file, job_name):
        self.sim_dir = sim_dir
        self.run_file = run_file
        self.job_name = job_name
        self.write_run_file()

    def write_run_file(self):
        """ Write a run file to simulation directory."""
        with open(self.run_file, 'r') as f:
            lines = f.readlines()
            with open(os.path.join(self.sim_dir, 'run.sh'), 'w') as r:
                for line in lines:
                    if 'job-name' in line:
                        r.write(f'#SBATCH --job-name="{self.job_name}"\n')
                    else:
                        r.write(line)

    def sbatch(self):
        """ Submit sbatch run file."""
        # TODO: better pls
        os.chdir(self.sim_dir)
        os.system("sbatch run.sh")


class Input(SimulationComponent):
    input_dict: dict[str, str]
    default_input: DefaultInput
    """ Lower level input file methods"""

    def __init__(self, sim: Simulation):
        """ 
        Read input file in simulation dir if it exsists, other wise define default input parameters.
        
        Parameters:
            sim_dir (str): Simulation directory
            parameters: depreciated
        """
        SimulationComponent.__init__(self, sim)
        self.default_input = get_default_input("cuda_MD")

        if self.sim.sim_dir.exists():
            self.initalize_input()
        self.input_dict = {}

    def clear(self):
        """
        deletes existing input file data
        """
        self.input_dict = {}
        self.write_input()

    def initalize_input(self, read_existing_input: Union[bool, None] = None):
        """
        Initializes the input file
        If read_existing_
        """
        if read_existing_input or read_existing_input is None:
            existing_input = (self.sim.sim_dir / 'input.json').exists() or (self.sim.sim_dir / 'input').exists()
        else:
            existing_input = False

        if existing_input:
            self.read_input()
        elif read_existing_input:
            raise SimBuildMissingFileException(self.sim, "input.json")

    def swap_default_input(self, default_type: str):
        """
        Swap the default input parameters to a different type of simulation.
        Current Options Include:
        cuda_prod, cpu_prod, cpu_relax
        """
        self.default_input = get_default_input(default_type)
        self.input_dict = self.default_input.get_dict()
        self.write_input()

    def get_last_conf_top(self) -> tuple[str, str]:
        """
        Set attributes containing the name of the inital conf (dat file) and topology
        """
        top, conf = find_top_dat(self.sim.sim_dir, self.sim)
        self.initial_conf = conf.name
        self.top = top.name
        return self.initial_conf, self.top

    def write_input_standard(self):
        """ Write a oxDNA input file to sim_dir"""
        if not self.has_top_conf():
            raise MissingTopConfException(self.sim)
        with oxpy.Context():
            ox_input = oxpy.InputFile()
            for k, v in self.input_dict.items():
                ox_input[k] = v
            with open(os.path.join(self.sim.sim_dir, f'input'), 'w') as f:
                print(ox_input, file=f)

    def write_input(self, production=False):
        """ Write an oxDNA input file as a json file to sim_dir"""
        if production is False:
            if not self.has_top_conf():
                top, conf = find_top_dat(self.sim.sim_dir, self.sim)
                self.set_top_file(top.name)
                self.set_conf_file(conf.name)

        # Write input file
        self.default_input.evaluate(**self.input_dict)
        # local input dict w/ defaults and manually-specified values
        inputdict = {
            **self.default_input.get_dict(),
            **self.input_dict
        }

        with open(os.path.join(self.sim.sim_dir, f'input.json'), 'w') as f:
            input_json = dumps(inputdict, indent=4)
            f.write(input_json)
        with open(os.path.join(self.sim.sim_dir, f'input'), 'w') as f:
            with oxpy.Context(print_coda=False):
                ox_input = oxpy.InputFile()
                for k, v in inputdict.items():
                    ox_input[k] = str(v)
                print(ox_input, file=f)

    def modify_input(self, parameters: dict):
        """ Modify the parameters of the oxDNA input file."""
        if os.path.exists(os.path.join(self.sim.sim_dir, 'input.json')):
            self.read_input()
        for k, v in parameters.items():
            self.input_dict[k] = v
        self.write_input()

    def read_input(self):
        """ Read parameters of exsisting input file in sim_dir"""
        if (self.sim.sim_dir / "input.json").exists() and os.stat(self.sim.sim_dir / "input.json").st_size > 0:
            with (self.sim.sim_dir / "input.json").open("r") as f:
                content = f.read()
                my_input = loads(content)
                self.input_dict = my_input
            # I don't know why you did this
            # it SHOULD throw an error if it finds a mangled JSON file!
            # except json.JSONDecodeError:
            #     self.initalize_input(read_exsisting_input=False)

        else:
            with open(os.path.join(self.sim.sim_dir, 'input'), 'r') as f:
                lines = f.readlines()
                lines = [line for line in lines if '=' in line]
                lines = [line.strip().split('=') for line in lines]
                my_input = {line[0].strip(): line[1].strip() for line in lines}
                # TODO: objects?

            self.input_dict = my_input

    def get_conf_file(self) -> Union[None, str]:
        """
        Returns: the conf file that the simulation will initialize from
        """
        if "conf_file" not in self.input_dict:
            return None
        else:
            return self.input_dict["conf_file"]

    def set_conf_file(self, conf_file_name: str):
        """
        Sets the conf file
        """
        self.input_dict["conf_file"] = conf_file_name

    def get_top_file(self) -> Union[None, str]:
        """
        Returns: the topology file that the simulation will use
        """
        if "topology" not in self.input_dict:
            return None
        else:
            return self.input_dict["topology"]

    def set_top_file(self, top_file_name: str):
        """
        Sets the topology file
        """
        self.input_dict["topology"] = top_file_name

    def has_top_conf(self) -> bool:
        return self.get_conf_file() is not None and self.get_top_file() is not None

    def get_last_conf(self) -> Union[None, str]:
        if "lastconf_file" not in self.input_dict:
            return None
        else:
            return self.input_dict["lastconf_file"]

    def set_last_conf(self, conf_file_name: str):
        self.input_dict["lastconf_file"] = conf_file_name

    initial_conf = property(get_conf_file, set_conf_file)
    top = property(get_top_file, set_conf_file)

    def __getitem__(self, item: str):
        return self.input_dict[item]

    def __setitem__(self, key: str, value: Union[str, float, bool]):
        self.input_dict[key] = value
        self.write_input()


class SequenceDependant(SimulationComponent):
    """ Make the targeted sim_dir run a sequence dependant oxDNA simulation"""
    parameters: str
    na_parameters: str
    rna_parameters: str

    def __init__(self, sim: Simulation):
        SimulationComponent.__init__(self, sim)
        # TODO: hardcode sequence-dependant parameters externally
        self.parameters = "\n".join([f"{name} = {value}" for name, value in SEQ_DEP_PARAMS.items()])

        self.na_parameters = "\n".join([f"{name} = {value}" for name, value in NA_PARAMETERS.items()])

        self.rna_parameters = "\n".join([f"{name} = {value}" for name, value in RNA_PARAMETERS.items()])

    def make_sim_sequence_dependant(self):
        self.sequence_dependant_input()
        self.write_sequence_dependant_file()

    def write_sequence_dependant_file(self):
        # TODO: externalize interaction-type stuff?
        int_type = self.sim.input.input_dict['interaction_type']
        if (int_type == 'DNA') or (int_type == 'DNA2') or (int_type == 'NA'):
            with open(os.path.join(self.sim.sim_dir, 'oxDNA2_sequence_dependent_parameters.txt'), 'w') as f:
                f.write(self.parameters)

        if (int_type == 'RNA') or (int_type == 'RNA2') or (int_type == 'NA'):
            with open(os.path.join(self.sim.sim_dir, 'rna_sequence_dependent_parameters.txt'), 'w') as f:
                f.write(self.rna_parameters)

        if int_type == 'NA':
            with open(os.path.join(self.sim.sim_dir, 'NA_sequence_dependent_parameters.txt'), 'w') as f:
                f.write(self.na_parameters)

    def sequence_dependant_input(self):
        int_type = self.sim.input.input_dict['interaction_type']

        if (int_type == 'DNA') or (int_type == 'DNA2'):
            self.sim.input_file({'use_average_seq': 'no', 'seq_dep_file': 'oxDNA2_sequence_dependent_parameters.txt'})

        if (int_type == 'RNA') or (int_type == 'RNA2'):
            self.sim.input_file({'use_average_seq': 'no', 'seq_dep_file': 'rna_sequence_dependent_parameters.txt'})

        if int_type == 'NA':
            self.sim.input_file({'use_average_seq': 'no',
                                 'seq_dep_file_DNA': 'oxDNA2_sequence_dependent_parameters.txt',
                                 'seq_dep_file_RNA': 'rna_sequence_dependent_parameters.txt',
                                 'seq_dep_file_NA': 'NA_sequence_dependent_parameters.txt'
                                 })


class OxdnaAnalysisTools(SimulationComponent):
    """Interface to OAT"""

    def describe(self):
        """
        what even is this code
        """
        try:
            try:
                self.top_info, self.traj_info = describe(self.sim.sim_files.top_filename, self.sim.sim_files.traj)
            except:
                self.top_info, self.traj_info = describe(self.sim.sim_files.top_filename,
                                                         self.sim.sim_files.last_conf_filename)
        except:
            self.top_info, self.traj_info = describe(self.sim.sim_files.top_filename, self.sim.sim_files.last_conf)

    def align(self, outfile: str = 'aligned.dat', args: str = '', join: bool = False):
        """
        Align trajectory to mean strucutre
        """
        if args == '-h':
            os.system('oat align -h')
            return None

        def run_align(self, outfile, args=''):  # why does this have a `self` param
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat align {self.sim.sim_files.traj} {outfile} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_align, args=(self, outfile,), kwargs={'args': args})
        p.start()
        if join:
            p.join()

    # def anm_parameterize(self, args='', join=False):
    #     if args == '-h':
    #         os.system('oat anm_parameterize -h')
    #         return None
    #     def run_anm_parameterize(self, args=''):
    #         start_dir = os.getcwd()
    #         os.chdir(self.sim.sim_dir)
    #         os.system(f'oat anm_parameterize {self.sim.sim_files.traj} {args}')
    #         os.chdir(start_dir)
    #     p = mp.Process(target=run_anm_parameterize, args=(self,), kwargs={'args':args})
    #     p.start()
    #     if join == True:
    #         p.join()

    # def backbone_flexibility(self, args='', join=False):
    #     if args == '-h':
    #         os.system('oat backbone_flexibility -h')
    #         return None
    #     def run_backbone_flexibility(self, args=''):
    #         start_dir = os.getcwd()
    #         os.chdir(self.sim.sim_dir)
    #         os.system(f'oat backbone_flexibility {self.sim.sim_files.traj} {args}')
    #         os.chdir(start_dir)
    #     p = mp.Process(target=run_backbone_flexibility, args=(self,), kwargs={'args':args})
    #     p.start()
    #     if join == True:
    #         p.join()

    # def bond_analysis(self, args='', join=False):
    #     if args == '-h':
    #         os.system('oat bond_analysis -h')
    #         return None
    #     def run_bond_analysis(self, args=''):
    #         start_dir = os.getcwd()
    #         os.chdir(self.sim.sim_dir)
    #         os.system(f'oat bond_analysis {self.sim.sim_files.traj} {args}')
    #         os.chdir(start_dir)
    #     p = mp.Process(target=run_bond_analysis, args=(self,), kwargs={'args':args})
    #     p.start()
    #     if join == True:
    #         p.join()

    def centroid(self, reference_structure='mean.dat', args='', join=False):
        """
        Extract conformation most similar to reference strucutre (mean.dat by default). centroid is actually a misnomer for this function.
        """
        if args == '-h':
            os.system('oat centroid -h')
            return None

        def run_centroid(self, reference_structure, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat centroid {reference_structure} {self.sim.sim_files.traj} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_centroid, args=(self, reference_structure,), kwargs={'args': args})
        p.start()
        if join:
            p.join()

    #     def clustering(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat clustering -h')
    #             return None
    #         def run_clustering(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat clustering {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_clustering, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    #     def config(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat config -h')
    #             return None
    #         def run_config(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat config {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_config, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    #     def contact_map(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat contact_map -h')
    #             return None
    #         def run_contact_map(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat contact_map {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_contact_map, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    #     def db_to_force(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat db_to_force -h')
    #             return None
    #         def run_db_to_force(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat db_to_force {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_db_to_force, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    def decimate(self, outfile='strided_trajectory.dat', args='', join=False):
        """
        Modify trajectory file, mostly to decrease file size. Use args='-h' for more details
        """
        if args == '-h':
            os.system('oat decimate -h')
            return None

        def run_decimate(self, outfile, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat decimate {self.sim.sim_files.traj} {outfile} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_decimate, args=(self, outfile,), kwargs={'args': args})
        p.start()
        if join:
            p.join()

    def deviations(self, mean_structure='mean.dat', args='', join=False):
        """
        Calculate rmsf and rmsd with respect to the mean strucutre Use args='-h' for more details.
        """
        if args == '-h':
            os.system('oat deviations -h')
            return None

        def run_deviations(self, mean_structure, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat deviations {mean_structure} {self.sim.sim_files.traj} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_deviations, args=(self, mean_structure), kwargs={'args': args})
        p.start()
        if join:
            p.join()

    #     def distance(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat distance -h')
    #             return None
    #         def run_distance(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat distance {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_distance, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    #     def duplex_angle_plotter(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat duplex_angle_plotter -h')
    #             return None
    #         def run_duplex_angle_plotter(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat duplex_angle_plotter {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_duplex_angle_plotter, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    #     def duplex_finder(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat duplex_finder -h')
    #             return None
    #         def run_duplex_finder(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat duplex_finder {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_duplex_finder, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    #     def file_info(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat file_info -h')
    #             return None
    #         def run_file_info(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat file_info {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_file_info, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    #     def forces2pairs(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat forces2pairs -h')
    #             return None
    #         def run_forces2pairs(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat forces2pairs {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_forces2pairs, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    #     def generate_force(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat generate_force -h')
    #             return None
    #         def run_generate_force(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat generate_force {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_generate_force, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    def mean(self, traj='trajectory.dat', args='', join=False):
        """
        Compute the mean strucutre. Use args='-h' for more details
        """
        if args == '-h':
            os.system('oat mean -h')
            return None

        def run_mean(self, traj, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat mean {traj} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_mean, args=(self, traj,), kwargs={'args': args})
        p.start()
        if join:
            p.join()

    def minify(self, traj='trajectory.dat', outfile='mini_trajectory.dat', args='', join=False):
        """
        Reduce trajectory file size. Use args='-h' for more details.
        """
        if args == '-h':
            os.system('oat minify -h')
            return None

        def run_minify(self, traj, outfile, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat minify {traj} {outfile} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_minify, args=(self, traj, outfile,), kwargs={'args': args})
        p.start()
        if join:
            p.join()

    #     def multidimensional_scaling_mean(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat multidimensional_scaling_mean -h')
    #             return None
    #         def run_multidimensional_scaling_mean(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat multidimensional_scaling_mean {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_multidimensional_scaling_mean, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    def output_bonds(self, args='', join=False):
        if args == '-h':
            os.system('oat output_bonds -h')
            return None

        def run_output_bonds(self, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(
                f'oat output_bonds {self.sim.sim_files.input_dict} {self.sim.sim_files.traj} {args} -v bonds.json')
            os.chdir(start_dir)

        p = mp.Process(target=run_output_bonds, args=(self,), kwargs={'args': args})
        p.start()
        if join:
            p.join()

    def oxDNA_PDB(self, configuration='mean.dat', direction='35', pdbfiles='', args='', join=False):
        """
        Turn a oxDNA file into a PDB file. Use args='-h' for more details
        """
        if args == '-h':
            os.system('oat oxDNA_PDB -h')
            return None

        def run_oxDNA_PDB(self, topology, configuration, direction, pdbfiles, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat oxDNA_PDB {topology} {configuration} {direction} {pdbfiles} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_oxDNA_PDB,
                       args=(self, self.sim.sim_files.top_filename, configuration, direction, pdbfiles),
                       kwargs={'args': args})
        p.start()
        if join:
            p.join()

    def pca(self, meanfile='mean.dat', outfile='pca.json', args='', join=False):
        """
        Preform principle componet analysis. Use args='-h' for more details
        """
        if args == '-h':
            os.system('oat pca -h')
            return None

        def run_pca(self, meanfile, outfile, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat pca {self.sim.sim_files.traj} {meanfile} {outfile} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_pca, args=(self, meanfile, outfile,), kwargs={'args': args})
        p.start()
        if join:
            p.join()

    def conformational_entropy(self, traj='trajectory.dat', temperature='293.15', meanfile='mean.dat',
                               outfile='conformational_entropy.json',
                               args='', join=False):
        """
        Calculate a strucutres conformational entropy (not currently supported in general). Use args='-h' for more details.
        """
        if args == '-h':
            os.system('oat conformational_entropy -h')
            return None

        def run_conformational_entropy(self, traj, temperature, meanfile, outfile, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat conformational_entropy {traj} {temperature} {meanfile} {outfile} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_conformational_entropy, args=(self, traj, temperature, meanfile, outfile,),
                       kwargs={'args': args})
        p.start()
        if join == True:
            p.join()

    def radius_of_gyration(self, traj='trajectory.dat', args='', join=False):
        """
        Calculate a strucutres radius_of_gyration (not currently supported in general). Use args='-h' for more details.
        """
        if args == '-h':
            os.system('oat radius_of_gyration -h')
            return None

        def run_radius_of_gyration(self, traj, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat radius_of_gyration {traj} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_radius_of_gyration, args=(self, traj), kwargs={'args': args})
        p.start()
        if join:
            p.join()

    #     def persistence_length(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat persistence_length -h')
    #             return None
    #         def run_persistence_length(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat persistence_length {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_persistence_length, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    #     def plot_energy(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat plot_energy -h')
    #             return None
    #         def run_plot_energy(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat plot_energy {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_plot_energy, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()

    def subset_trajectory(self, args='', join=False):
        """
        Extract specificed indexes from a trajectory, creating a new trajectory. Use args='-h' for more details
        """
        if args == '-h':
            os.system('oat subset_trajectory -h')
            return None

        def run_subset_trajectory(self, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat subset_trajectory {self.sim.sim_files.traj} {self.sim.sim_files.top_filename} {args}')
            os.chdir(start_dir)

        p = mp.Process(target=run_subset_trajectory, args=(self,), kwargs={'args': args})
        p.start()
        if join:
            p.join()

    #     def superimpose(self, args='', join=False):
    #         if args == '-h':
    #             os.system('oat superimpose -h')
    #             return None
    #         def run_superimpose(self, args=''):
    #             start_dir = os.getcwd()
    #             os.chdir(self.sim.sim_dir)
    #             os.system(f'oat superimpose {self.sim.sim_files.traj} {args}')
    #             os.chdir(start_dir)
    #         p = mp.Process(target=run_superimpose, args=(self,), kwargs={'args':args})
    #         p.start()
    #         if join == True:
    #             p.join()
    def com_distance(self, base_list_file_1=None, base_list_file_2=None, base_list_1=None, base_list_2=None, args='',
                     join=False):
        """
        Find the distance between the center of mass of two groups of particles (currently not supported generally). Use args='-h' for more details
        """
        if args == '-h':
            os.system('oat com_distance -h')
            return None

        def build_space_sep_base_list(comma_sep_indexes, filename=None):
            space_seperated = comma_sep_indexes.replace(',', ' ')

            base_filename = 'base_list_'
            counter = 0
            while os.path.exists(os.path.join(self.sim.sim_dir, f"{base_filename}{counter}.txt")):
                counter += 1
            print(f"{base_filename}{counter}.txt")
            filename = os.path.join(self.sim.sim_dir, f"{base_filename}{counter}.txt")
            with open(filename, 'w') as f:
                f.write(space_seperated)
            # print(filename)
            return filename

        def run_com_distance(self, base_list_file_1, base_list_file_2, args=''):
            start_dir = os.getcwd()
            os.chdir(self.sim.sim_dir)
            os.system(f'oat com_distance -i {self.sim.sim_files.traj} {base_list_file_1} {base_list_file_2} {args}')
            os.chdir(start_dir)

        if (base_list_file_1 is None) and (base_list_file_2 is None):
            base_list_file_1 = build_space_sep_base_list(base_list_1)
            base_list_file_2 = build_space_sep_base_list(base_list_2)

        p = mp.Process(target=run_com_distance, args=(self, base_list_file_1, base_list_file_2), kwargs={'args': args})
        p.start()
        if join:
            p.join()


class Analysis(SimulationComponent):
    """ Methods used to interface with oxDNA simulation in jupyter notebook (currently in work)"""

    def __init__(self, simulation):
        """ Set attributes to know all files in sim_dir and the input_parameters"""
        SimulationComponent.__init__(self, simulation)
        self.sim_files = simulation.sim_files

    def get_init_conf(self) -> tuple[tuple[TopInfo, TrajInfo], Configuration]:
        """ Returns inital topology and dat file paths, as well as x,y,z info of the conf."""
        self.sim_files.parse_current_files()
        ti, di = describe(self.sim_files.top_filename,
                          self.sim_files.last_conf_filename)
        return (ti, di), get_confs(ti, di, 0, 1)[0]

    def get_last_conf(self) -> tuple[tuple[TopInfo, TrajInfo], Configuration]:
        """ Returns last topology and dat file paths, as well as x,y,z info of the conf."""
        self.sim_files.parse_current_files()
        ti, di = describe(self.sim_files.top_filename,
                          self.sim_files.last_conf)
        return (ti, di), get_confs(ti, di, 0, 1)[0]

    def view_init(self):
        """ Interactivly view inital oxDNA conf in jupyter notebook."""
        (ti, di), conf = self.get_init_conf()
        oxdna_conf(ti, conf)
        sleep(2.5)

    def view_last(self):
        """ Interactivly view last oxDNA conf in jupyter notebook."""
        self.sim_files.parse_current_files()
        try:
            (ti, di), conf = self.get_last_conf()
            oxdna_conf(ti, conf)
        except Exception as e:
            # TODO: custom exception for missing conf, consider adapting one from pypatchy.patchy.stage
            raise Exception('No last conf file avalible')
        sleep(2.5)

    def get_conf_count(self) -> int:
        """ Returns the number of confs in trajectory file."""
        self.sim_files.parse_current_files()
        ti, di = describe(self.sim_files.top_filename,
                          self.sim_files.traj)
        return len(di.idxs)

    def get_conf(self, conf_id: int):
        """ Returns x,y,z (and other) info of specified conf."""
        self.sim_files.parse_current_files()
        ti, di = describe(self.sim_files.top_filename,
                          self.sim_files.traj)
        l = len(di.idxs)
        if conf_id < l:
            return (ti, di), get_confs(ti, di, conf_id, 1)[0]
        else:
            # TODO: custom exception
            raise Exception("You requested a conf out of bounds.")

    def current_step(self) -> float:
        """ Returns the time-step of the most recently save oxDNA conf."""
        n_confs = float(self.get_conf_count())
        steps_per_conf = float(self.sim.input.input_dict["print_conf_interval"])
        return n_confs * steps_per_conf

    def view_conf(self, conf_id: int):
        """ Interactivly view oxDNA conf in jupyter notebook."""
        (ti, di), conf = self.get_conf(conf_id)
        oxdna_conf(ti, conf)
        sleep(2.5)

    def get_energy_df(self):
        """ Plot energy of oxDNA simulation."""
        try:
            self.sim_files.parse_current_files()
            sim_type = self.sim.input.input_dict['sim_type']
            if (sim_type == 'MC') or (sim_type == 'VMMC'):
                df = pd.read_csv(self.sim_files.energy, delim_whitespace=True, names=['time', 'U', 'P', 'K', 'empty'])
            else:
                df = pd.read_csv(self.sim_files.energy, delim_whitespace=True, names=['time', 'U', 'P', 'K'])

            df = df[df.U <= 10]
            self.energy_df = df

        except Exception as e:
            raise Exception(e)

    def plot_energy(self, fig=None, ax=None, label=None):
        """ Plot energy of oxDNA simulation."""
        try:
            self.sim_files.parse_current_files()
            sim_type = self.sim.input.input_dict['sim_type']
            if (sim_type == 'MC') or (sim_type == 'VMMC'):
                df = pd.read_csv(self.sim_files.energy, delim_whitespace=True, names=['time', 'U', 'P', 'K', 'empty'])
            else:
                df = pd.read_csv(self.sim_files.energy, delim_whitespace=True, names=['time', 'U', 'P', 'K'])
            dt = float(self.sim.input.input_dict["dt"])
            steps = float(self.sim.input.input_dict["steps"])
            # df = df[df.U <= 10]
            # df = df[df.U >= -10]
            # make sure our figure is bigger
            if fig is None:
                plt.figure(figsize=(15, 3))
            # plot the energy
            if ax is None:
                if (sim_type == 'MC') or (sim_type == 'VMMC'):
                    plt.plot(df.time, df.U, label=label)
                else:
                    plt.plot(df.time / dt, df.U, label=label)
                plt.ylabel("Energy")
                plt.xlabel("Steps")
            else:
                if (sim_type == 'MC') or (sim_type == 'VMMC'):
                    ax.plot(df.time, df.U, label=label)
                else:
                    ax.plot(df.time / dt, df.U, label=label)
                ax.set_ylabel("Energy")
                ax.set_xlabel("Steps")

            if np.any(df.U > 10):
                print(self.sim.sim_dir)
                print('Energy is greater than 10, check for errors in the simulation')
            if np.any(df.U < -10):
                print(self.sim.sim_dir)
                print('Energy is less than -10, check for errors in the simulation')

        except Exception as e:
            # TODO: custom exception handling and exception raising
            print(f'{self.sim.sim_dir}: No energy file avalible')

    def plot_observable(self, observable: dict,
                        sliding_window: Union[False, Any] = False, fig=True):
        file_name = observable['output']['name']
        conf_interval = float(observable['output']['print_every'])
        df = pd.read_csv(f"{self.sim.sim_dir}/{file_name}", header=None, engine='pyarrow')
        if sliding_window is not False:
            df = df.rolling(window=sliding_window).sum().dropna().div(sliding_window)
        df = np.concatenate(np.array(df))
        sim_conf_times = np.linspace(0, conf_interval * len(df), num=len(df))
        if fig is True:
            plt.figure(figsize=(15, 3))
        plt.xlabel('steps')
        plt.ylabel(f'{os.path.splitext(file_name)[0]} (sim units)')
        plt.plot(sim_conf_times, df, label=self.sim.sim_dir.split("/")[-1], rasterized=True)

    def hist_observable(self, observable: dict, bins=10, fig=True):
        file_name = observable['output']['name']
        conf_interval = float(observable['output']['print_every'])
        df = pd.read_csv(f"{self.sim.sim_dir}/{file_name}", header=None)
        df = np.concatenate(np.array(df))
        sim_conf_times = np.linspace(0, conf_interval * len(df), num=len(df))
        if fig is True:
            plt.figure(figsize=(15, 3))
        plt.xlabel(f'{os.path.splitext(file_name)[0]} (sim units)')
        plt.ylabel(f'Probablity')
        H, bins = np.histogram(df, density=True, bins=bins)
        H = H * (bins[1] - bins[0])
        plt.plot(bins[:-1], H, label=self.sim.sim_dir.split("/")[-1])

    # Unstable
    def view_traj(self, init=0, op=None):
        print('This feature is highly unstable and will crash your kernel if you scroll through confs too fast')
        # get the initial conf and the reference to the trajectory 
        (ti, di), cur_conf = self.get_conf(init)

        slider = widgets.IntSlider(
            min=0,
            max=len(di.idxs),
            step=1,
            description="Select:",
            value=init
        )

        output = widgets.Output()
        if op:
            min_v, max_v = np.min(op), np.max(op)

        def handle(obj=None):
            conf = get_confs(ti, di, slider.value, 1)[0]
            with output:
                output.clear_output()
                if op:
                    # make sure our figure is bigger
                    plt.figure(figsize=(15, 3))
                    plt.plot(op)
                    print(init)
                    plt.plot([slider.value, slider.value], [min_v, max_v], color="r")
                    plt.show()
                oxdna_conf(ti, conf)

        slider.observe(handle)
        display(slider, output)
        handle(None)

    def get_up_down(self, x_max: float, com_dist_file: str, pos_file: str):
        key_names = ['a', 'b', 'c', 'p', 'va', 'vb', 'vc', 'vp']

        def process_pos_file(pos_file: str, key_names: list) -> dict:
            cms_dict = {}
            with open(pos_file, 'r') as f:
                pos = f.readlines()
                pos = [line.strip().split(' ') for line in pos]
                for idx, string in enumerate(key_names):
                    cms = np.transpose(pos)[idx]
                    cms = [np.array(line.split(','), dtype=np.float64) for line in cms]
                    cms_dict[string] = np.array(cms)
            return cms_dict

        def point_in_triangle(a, b, c, p):
            u = b - a
            v = c - a
            n = np.cross(u, v)
            w = p - a
            gamma = (np.dot(np.cross(u, w), n)) / np.dot(n, n)
            beta = (np.dot(np.cross(w, v), n)) / np.dot(n, n)
            alpha = 1 - gamma - beta
            return ((-1 <= alpha) and (alpha <= 1) and (-1 <= beta) and (beta <= 1) and (-1 <= gamma) and (gamma <= 1))

        def point_over_plane(a, b, c, p):
            u = c - a
            v = b - a
            cp = np.cross(u, v)
            va, vb, vc = cp
            d = np.dot(cp, c)
            plane = np.array([va, vb, vc, d])
            point = np.array([p[0], p[1], p[2], 1])
            result = np.dot(plane, point)
            return True if result > 0 else False

        def up_down(x_max: float, com_dist_file: str, pos_file: str) -> list:
            with open(com_dist_file, 'r') as f:
                com_dist = f.readlines()
            com_dist = [line.strip() for line in com_dist]
            com_dist = list(map(float, com_dist))
            cms_list = process_pos_file(pos_file, key_names)
            up_or_down = [point_in_triangle(a, b, c, p) for (a, b, c, p) in
                          zip(cms_list['va'], cms_list['vb'], cms_list['vc'], cms_list['vp'])]
            over_or_under = [point_over_plane(a, b, c, p) for (a, b, c, p) in
                             zip(cms_list['va'], cms_list['vb'], cms_list['vc'], cms_list['vp'])]

            # true_up_down = []
            # # print(up_or_down)
            # # print(over_or_under)
            # new_coms = []
            # for com, u_d, o_u in zip(com_dist, up_or_down, over_or_under):
            #     if u_d != o_u:
            #         if abs(com) > (x_max * 0.75):
            #             if u_d == 0:
            #                 new_coms.append(-com)
            #             else:
            #                 new_coms.append(com)      
            #         else:
            #             if o_u == 0:
            #                 new_coms.append(-com)
            #             else:
            #                 new_coms.append(com)   
            #     else:
            #         if o_u == 0:
            #             new_coms.append(-com)
            #         else:
            #             new_coms.append(com) 
            # com_dist = new_coms       

            com_dist = [-state if direction == 0 else state for state, direction in zip(com_dist, over_or_under)]

            # if np.mean(com_dist) > :
            #     com_dist = [dist for dist in com_dist if (np.sign(dist) == np.sign(np.mean(com_dist)))]
            # if (abs(max(com_dist) + min(com_dist)) < 2) :
            #     print(np.mean(com_dist))
            #     com_dist = [dist for dist in com_dist if (np.sign(dist) == np.sign(np.mean(com_dist)))]

            com_dist = [x_max - state if state > 0 else -x_max - state for state in com_dist]
            # if max(abs(max(com_dist)),  abs(min(com_dist))) > 15:
            #     com_dist = [dist if (np.sign(dist) == np.sign(np.mean(com_dist))) else -dist for dist in com_dist ]

            #             if max(abs(max(com_dist)),  abs(min(com_dist))) > 15:
            #                 com_dist = [abs(dist) if (np.sign(dist) == -1) else dist for dist in com_dist]

            com_dist = [np.round(val, 4) for val in com_dist]
            return com_dist

        return (up_down(x_max, com_dist_file, pos_file))

    def view_cms_obs(self, xmax, print_every, sliding_window=False, fig=True):
        self.sim_files.parse_current_files()
        new_com_vals = self.get_up_down(xmax, self.sim_files.com_distance, self.sim_files.cms_positions)
        conf_interval = float(print_every)
        df = pd.DataFrame(new_com_vals)
        if sliding_window is not False:
            df = df.rolling(window=sliding_window).sum().dropna().div(sliding_window)
        df = np.concatenate(np.array(df))
        sim_conf_times = np.linspace(0, conf_interval * len(df), num=len(df))
        if fig is True:
            plt.figure(figsize=(15, 3))
        plt.xlabel('steps')
        plt.ylabel(f'End-to-End Distance (sim units)')
        plt.plot(sim_conf_times, df, label=self.sim.sim_dir.split("/")[-1])

    def hist_cms_obs(self, xmax, print_every, bins=10, fig=True):
        new_com_vals = self.get_up_down(xmax, self.sim_files.com_distance, self.sim_files.cms_positions)
        conf_interval = float(print_every)
        df = pd.DataFrame(new_com_vals)
        df = np.concatenate(np.array(df))
        sim_conf_times = np.linspace(0, conf_interval * len(df), num=len(df))
        if fig is True:
            plt.figure(figsize=(15, 3))
        plt.xlabel(f'End-to-End Distance (sim units)')
        plt.ylabel(f'Probablity')
        H, bins = np.histogram(df, density=True, bins=bins)
        H = H * (bins[1] - bins[0])
        plt.plot(bins[:-1], H, label=self.sim.sim_dir.split("/")[-1])


class Observable:
    """ Currently implemented observables for this oxDNA wrapper."""

    @staticmethod
    def distance(particle_1=None, particle_2=None, PBC=None, print_every=None, name=None):
        """
        Calculate the distance between two (groups) of particles
        """
        return ({
            "output": {
                "print_every": print_every,
                "name": name,
                "cols": [
                    {
                        "type": "distance",
                        "particle_1": particle_1,
                        "particle_2": particle_2,
                        "PBC": PBC
                    }
                ]
            }
        })

    @staticmethod
    def hb_list(print_every=None, name=None, only_count=None):
        """
        Compute the number of hydrogen bonds between the specified particles
        """
        return ({
            "output": {
                "print_every": print_every,
                "name": name,
                "cols": [
                    {
                        "type": "hb_list",
                        "order_parameters_file": "hb_list.txt",
                        "only_count": only_count
                    }
                ]
            }
        })

    @staticmethod
    def particle_position(particle_id=None, orientation=None, absolute=None, print_every=None, name=None):
        """
        Return the x,y,z postions of specified particles
        """
        return ({
            "output": {
                "print_every": print_every,
                "name": name,
                "cols": [
                    {
                        "type": "particle_position",
                        "particle_id": particle_id,
                        "orientation": orientation,
                        "absolute": absolute
                    }
                ]
            }
        })

    @staticmethod
    def potential_energy(print_every=None, split=None, name=None, precision=6, general_format=True):
        """
        Return the potential energy
        """
        return ({
            "output": {
                "print_every": f'{print_every}',
                "name": name,
                "cols": [
                    {
                        "type": "potential_energy",
                        "split": f"{split}",
                        "precision" : f'{precision}',
                        "general_format": f'{general_format}'
                    }
                ]
            }
        })

    @staticmethod
    def force_energy(print_every=None, name=None, print_group=None, precision=6, general_format='true'):
        """
        Return the energy exerted by external forces
        """
        if print_group is not None:
            return ({
                "output": {
                    "print_every": f'{print_every}',
                    "name": name,
                    "cols": [
                        {
                            "type": "force_energy",
                            "print_group": f"{print_group}",
                            "precision": f'{precision}',
                            "general_format": f'{general_format}'
                        }
                    ]
                }
            })
        else:
            return ({
                "output": {
                    "print_every": f'{print_every}',
                    "name": name,
                    "cols": [
                        {
                            "type": "force_energy",
                            "precision": f'{precision}',
                        }
                    ]
                }
            })

    @staticmethod
    def kinetic_energy(print_every=None, name=None):
        """
        Return the kinetic energy  
        """
        return ({
            "output": {
                "print_every": f'{print_every}',
                "name": name,
                "cols": [
                    {
                        "type": "kinetic_energy"
                    }
                ]
            }
        })


class Force:
    """ Currently implemented external forces for this oxDNA wrapper."""

    @staticmethod
    def morse(particle=None, ref_particle=None, a=None, D=None, r0=None, PBC=None):
        "Morse potential"
        return ({"force": {
            "type": 'morse',
            "particle": f'{particle}',
            "ref_particle": f'{ref_particle}',
            "a": f'{a}',
            "D": f'{D}',
            "r0": f'{r0}',
            "PBC": f'{PBC}',
        }
        })

    @staticmethod
    def skew_force(particle=None, ref_particle=None, stdev=None, r0=None, shape=None, PBC=None):
        "Skewed Gaussian potential"
        return ({"force": {
            "type": 'skew_trap',
            "particle": f'{particle}',
            "ref_particle": f'{ref_particle}',
            "stdev": f'{stdev}',
            "r0": f'{r0}',
            "shape": f'{shape}',
            "PBC": f'{PBC}'
        }
        })

    @staticmethod
    def com_force(com_list=None, ref_list=None, stiff=None, r0=None, PBC=None, rate=None):
        "Harmonic trap between two groups"
        return ({"force": {
            "type": 'com',
            "com_list": f'{com_list}',
            "ref_list": f'{ref_list}',
            "stiff": f'{stiff}',
            "r0": f'{r0}',
            "PBC": f'{PBC}',
            "rate": f'{rate}'
        }
        })

    @staticmethod
    def mutual_trap(particle=None, ref_particle=None, stiff=None, r0=None, PBC=None):
        """
        A spring force that pulls a particle towards the position of another particle
    
        Parameters:
            particle (int): the particle that the force acts upon
            ref_particle (int): the particle that the particle will be pulled towards
            stiff (float): the force constant of the spring (in simulation units)
            r0 (float): the equlibrium distance of the spring
            PBC (bool): does the force calculation take PBC into account (almost always 1)
        """
        return ({"force": {
            "type": "mutual_trap",
            "particle": particle,
            "ref_particle": ref_particle,
            "stiff": stiff,
            "r0": r0,
            "PBC": PBC
        }
        })

    @staticmethod
    def string(particle, f0, rate, direction):
        """
        A linear force along a vector
    
        Parameters:
            particle (int): the particle that the force acts upon
            f0 (float): the initial strength of the force at t=0 (in simulation units)
            rate (float or SN string): growing rate of the force (simulation units/timestep)
            dir ([float, float, float]): the direction of the force
        """
        return ({"force": {
            "type": "string",
            "particle": particle,
            "f0": f0,
            "rate": rate,
            "dir": direction
        }})

    @staticmethod
    def harmonic_trap(particle, pos0, stiff, rate, direction):
        """
        A linear potential well that traps a particle
    
        Parameters:
            particle (int): the particle that the force acts upon
            pos0 ([float, float, float]): the position of the trap at t=0
            stiff (float): the stiffness of the trap (force = stiff * dx)
            rate (float): the velocity of the trap (simulation units/time step)
            direction ([float, float, float]): the direction of movement of the trap
        """
        return ({"force": {
            "type": "trap",
            "particle": particle,
            "pos0": pos0,
            "rate": rate,
            "dir": direction
        }})

    @staticmethod
    def rotating_harmonic_trap(particle, stiff, rate, base, pos0, center, axis, mask):
        """
        A harmonic trap that rotates in space with constant angular velocity
    
        Parameters:
            particle (int): the particle that the force acts upon
            pos0 ([float, float, float]): the position of the trap at t=0
            stiff (float): the stiffness of the trap (force = stiff * dx)
            rate (float): the angular velocity of the trap (simulation units/time step)
            base (float): initial phase of the trap
            axis ([float, float, float]): the rotation axis of the trap
            mask([float, float, float]): the masking vector of the trap (force vector is element-wise multiplied by mask)
        """
        return ({"force": {
            "type": "twist",
            "particle": particle,
            "stiff": stiff,
            "rate": rate,
            "base": base,
            "pos0": pos0,
            "center": center,
            "axis": axis,
            "mask": mask
        }})

    @staticmethod
    def repulsion_plane(particle, stiff, direction, position):
        """
        A plane that forces the affected particle to stay on one side.
    
        Parameters:
            particle (int): the particle that the force acts upon.  -1 will act on whole system.
            stiff (float): the stiffness of the trap (force = stiff * distance below plane)
            dir ([float, float, float]): the normal vecor to the plane
            position(float): position of the plane (plane is d0*x + d1*y + d2*z + position = 0)
        """
        return ({"force": {
            "type": "repulsion_plane",
            "particle": particle,
            "stiff": stiff,
            "dir": direction,
            "position": position
        }})

    @staticmethod
    def repulsion_sphere(particle, center, stiff, r0, rate=1):
        """
        A sphere that encloses the particle
        
        Parameters:
            particle (int): the particle that the force acts upon
            center ([float, float, float]): the center of the sphere
            stiff (float): stiffness of trap
            r0 (float): radius of sphere at t=0
            rate (float): the sphere's radius changes to r = r0 + rate*t
        """
        return ({"force": {
            "type": "sphere",
            "center": center,
            "stiff": stiff,
            "r0": r0,
            "rate": rate
        }})


class SimFiles:
    """ Parse the current files present in simulation directory"""

    def __init__(self, sim):
        self.sim = sim
        if os.path.exists(self.sim.sim_dir):
            self.file_list = os.listdir(self.sim.sim_dir)
            self.parse_current_files()

    def parse_current_files(self):
        """

        """
        if self.sim.sim_dir.exists():
            self.file_list: list[str] = os.listdir(self.sim.sim_dir)
        else:
            print('Simulation directory does not exsist')
            return None
        for file in self.file_list:
            if not file.endswith('pyidx'):
                if file == 'trajectory.dat':
                    self.traj = self.sim.sim_dir / file
                elif file == 'last_conf.dat':
                    self.last_conf = self.sim.sim_dir / file
                elif file.endswith(".dat") and not any([
                    file.endswith("energy.dat"),
                    file.endswith("trajectory.dat"),
                    file.endswith("error_conf.dat"),
                    file.endswith("last_hist.dat"),
                    file.endswith("traj_hist.dat"),
                    file.endswith("last_conf.dat")
                ]):
                    self.dat = self.sim.sim_dir / file
                elif file.endswith('.top'):
                    self.top = self.sim.sim_dir / file
                elif file == 'forces.json':
                    self.force = self.sim.sim_dir / file
                elif file == 'input':
                    self.input = self.sim.sim_dir / file
                elif file == 'input.json':
                    self.input_js = self.sim.sim_dir / file
                elif file == 'observables.json':
                    self.observables = self.sim.sim_dir / file
                elif file == 'run.sh':
                    self.run_file = self.sim.sim_dir / file
                elif file.startswith('slurm'):
                    self.run_file = self.sim.sim_dir / file
                elif 'energy.dat' in file:
                    self.energy = self.sim.sim_dir / file
                elif 'com_distance' in file:
                    self.com_distance = self.sim.sim_dir / file
                elif 'cms_positions' in file:
                    self.cms_positions = self.sim.sim_dir / file
                elif 'par' in file:
                    self.par = self.sim.sim_dir / file
                elif 'last_hist.dat' in file:
                    self.last_hist = self.sim.sim_dir / file
                elif 'hb_observable.txt' in file:
                    self.hb_observable = self.sim.sim_dir / file
                elif 'potential_energy.txt' in file:
                    self.potential_energy = self.sim.sim_dir / file
                elif 'all_observables.txt' in file:
                    self.all_observables = self.sim.sim_dir / file
                elif 'hb_contacts.txt' in file:
                    self.hb_contacts = self.sim.sim_dir / file
                elif 'run_time_custom_observable.json' in file:
                    self.run_time_custom_observable = self.sim.sim_dir / file


class SimBuildException(Exception, SimulationComponent):
    pass


class MissingTopConfException(SimBuildException):
    def __str__(self) -> str:
        return f"No specified topology and initial configuration files specified in the input file for simulation at {str(self.sim.sim_dir)}"


class SimBuildMissingFileException(SimBuildException):
    missing_file_descriptor: str

    def __init__(self, sim: Simulation, missing_file: str):
        SimulationComponent.__init__(self, sim)
        self.missing_file_descriptor = missing_file

    def __str__(self) -> str:
        return f"No {self.missing_file_descriptor} in directory {str(self.sim.file_dir)}"


def find_top_dat(directory: Path, sim: Union[Simulation, None] = None) -> tuple[Path, Path]:
    """
    Tries to find a top and dat file in the provided directory. simulation object is provided
    for err-messaging purposes only
    """
    # list files in simulation directory

    # skip inputs where we've already set top

    # skip inputs where we've already set top and
    return find_top_file(directory, sim), find_conf_file(directory, sim)


def find_top_file(directory: Path, sim: Union[Simulation, None] = None) -> Path:
    """
    Tries to find a top file in the provided directory. simulation object is provided
    for err-messaging purposes only
    """
    if not directory.exists():
        raise FileNotFoundError(f"{str(directory)} does not exist")
    try:
        return [file for file in directory.iterdir() if file.name.endswith('.top')][0]
    except IndexError:
        if sim is not None:
            raise SimBuildException(sim, "topology file")
        else:
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    f"No valid .top file found in directory {str(directory)}")


def find_conf_file(directory: Path, sim: Union[Simulation, None] = None) -> Path:
    """
    Tries to find a dat file in the provided directory. simulation object is provided
    for err-messaging purposes only
    """
    try:
        last_conf = [file for file in directory.iterdir()
                     if file.name.startswith('last_conf')
                     and not file.name.endswith('pyidx')][0]
    except IndexError:
        try:
            last_conf = [file for file in directory.iterdir() if file.name.endswith(".dat") and not any([
                file.name.endswith("energy.dat"),
                file.name.endswith("trajectory.dat"),
                file.name.endswith("error_conf.dat")])
                         ][0]
        except IndexError:
            if sim is not None:
                raise SimBuildException(sim, "initial conf file")
            else:
                raise FileNotFoundError(errno.ENOENT,
                                        os.strerror(errno.ENOENT),
                                        f"No valid .dat file found in directory {str(directory)}")
    return last_conf