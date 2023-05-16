# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

from matplotlib import pyplot as plt

import math
import numpy as np
from copy import deepcopy
from sys import getsizeof
import pandas as pd
import multiprocessing as mp
import pickle
import os
import random

import gc

from scipy import signal

from numba import njit,prange

import random
import time
import importlib
import os
from collections import deque
from enum import IntEnum
from tqdm.auto import tqdm

import mbes_sim.functions.helperfunctions as hlp
import mbes_sim.functions.create_bubblesfunctions as bubbles
import mbes_sim.functions.navfunctions as nav
import mbes_sim.functions.transformfunctions as tf
import mbes_sim.functions.pulsefunctions as pf
import mbes_sim.functions.gridfunctions as gf
import mbes_sim.functions.beampatternfunctions as bpf
import mbes_sim.mbes as mb
import mbes_sim.scattergrid as sg
import mbes_sim.simulation as SIM

"""Functions to run simulations
"""

# ----- simulation setup classes -----
# these enumerator describe the possible values for the simulation setup

class t_Window(IntEnum):
    """Window types for the beampattern function
    """

    Box = 0,    # Unshaded beam pattern
    Exponential = 1,   # exponentially shaded beam pattern
    Hann = 2,  # hann shaded beam pattern
    idealized = 3 # idealized beampattern (rectangular)

    def __str__(self):
        if self.value == t_Window.Box: return 'Box'
        if self.value == t_Window.Exponential: return 'Exponential'
        if self.value == t_Window.Hann: return 'Hann'

    @staticmethod
    def print_all():
        for i in range(99):
            try:
                print(t_Window(i))
            except:
                pass

class t_Survey(IntEnum):
    """Survey types for the simulation
    
    Note: to create a survey with exagerated motion, use the exagHPR parameter in the SimulationSetup class
    """

    IdealMotion = 0, # No yaw pitch roll motion
    RealMotion = 1 # real vessel motion from navigation data

    def __str__(self):
        if self.value == IdealMotion: return 'Flat'
        if self.value == RealMotion: return 'FromNavigation'

    @staticmethod
    def print_all():
        for i in range(99):
            try:
                print(t_Survey(i))
            except:
                pass

class t_Bubbles(IntEnum):
    """_Bubble target types for the simulation
    """
    SingleBubble = 0,   # Single bubble target
    BubbleStream = 1,   # Bubble stream target

    def __str__(self):
        if self.value == t_Bubbles.SingleBubble: return 'SingleBubble'
        if self.value == t_Bubbles.BubbleStream: return 'BubbleStream'

    @staticmethod
    def print_all():
        for i in range(99):
            try:
                print(t_Bubbles(i))
            except:
                pass


class TargetGenerator(object):
    """Class to generate targets for the simulation in an like an iterator

    Usage: targetGenerator = TargetGenerator(function, **kwargs)
            for target in targetGenerator.iterate(num):
                # do something with target

            TargetGeneratorIt = TargetGenerator(function, **kwargs).iterarte(num of simulations to run)
            target = next(TargetGeneratorIt)
    """

    def __init__(self, function, **kwargs):
        """Initialize a target generator object
        
        Usage: targetGenerator = TargetGenerator(function, **kwargs)
            for target in targetGenerator.iterate(num):
                # do something with target

            TargetGeneratorIt = TargetGenerator(function, **kwargs).iterarte(num of simulations to run)
            target = next(TargetGeneratorIt)

        Parameters
        ----------
        function : target generating function
            E.g. bubbles.create_bubble_stream
        kwargs: keyword arguments
            Arguments for the target generating function
        """
        self.function = function
        self.kwargs = kwargs
        self.num = None

    # implement iterator protocol
    def iterarte(self, num):
        if not num > 0:
            num = 0
        self.num = num
        return self

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.num and self.n >= self.num:
            raise StopIteration
        else:
            self.n += 1
            return self.function(**self.kwargs)

    def __len__(self):
        return self.num

def init_angular_resolution(numAngles: int, verbose = False):
    """Initialize the angular resolution for the beampattern functions

    Parameters
    ----------
    numAngles : int
        Number of angles to use for the beampattern functions
    verbose : bool, optional
        If true: print Angle resolution, by default False
    """
    bpf.init(numAngles)

    if verbose:
        print('Angular resolution for beampattern functions:',bpf.ANGLE_RESOLUTION,'angles')
        print('Angular resolution for beampattern functions:',round(1/bpf.DELTA_ANGLE_DEG,2),'angle steps per °')

def get_AngleResolution():
    """Get the angular resolution for the beampattern functions
    """
    return bpf.ANGLE_RESOLUTION


class SimulationSetup(object):
    """Class to setup a simulation
    """

    def __init__(self,
                 addPlots : bool = False,
                 prefix : str = '',
                 blockAvg : bool = True,
                 downfactor : float = 1,
                 resfactor : float = 1,
                 windowType : t_Window = t_Window.Box,
                 idealizedBeampattern : bool = False,
                 equiDist   : bool = False,
                 surveyType : t_Survey = t_Survey.IdealMotion,
                 motionDataPath = None,
                 voxelsize  : float = 1.5,
                 voxelsizeZ  : float = None,
                 surveyspeedKnots : float = 3,
                 swathdistance : float = 0.8,
                 bubbleType : t_Bubbles = t_Bubbles.SingleBubble,
                 layerDepths : list = [80, 105],
                 layerSizes : list = [20,20],
                 exagHPR : float = 1,
                 maxDepth : float = 125,
                 maxRange : float = 125,
                 NumBeams : int = 256,
                 NumElements : int = 128,
                 SampleDistance: float = 0.324, # 432 µs * 1500 m/s / 2
                 EffectivePulseWidth: float = 0.5625, # 750 µs * 1500 m/s / 2
                 BaseDirectory='simulation_results',
                 load_previous_simresults : bool = True,
                 verbose : bool = False
                 ):
        """Initialize a simulation setup object

        Parameters
        ----------
        addPlots : bool, optional
            If true: add plots to the resulting pandas dataframs, by default False
        prefix : str, optional
            Prefix name for this simulation (will create a new folder for a prefix), by default ''
        blockAvg : bool, optional
            If true: also simulate the block average gridding, by default True
        downfactor : float, optional
            Reduce the number of beams/samples and correspondingly the beam opening angle and pulse length
            This is used to speed up the simulation for quick tests, by default 1
        resfactor : float, optional
            Reducing this factor will increase the beam opening angle and the pulse lengths
            Use for testing onlu, by default 1
        windowType : t_Window, optional
            Window type for the beam pattern generation, by default t_Window.Box
        idealizedBeampattern : bool, optional
            Use idealized (rectangular beam pattern), by default False
        equiDist : bool, optional
            Use equidistant beam spacing (warning: not properly tested), by default False
        surveyType : t_Survey, optional
            Survey type used for the simulation, by default t_Survey.IdealMotion
        motionDataPath : str, optional
            Path to a motion data file used for the RealVessel motion scenario
            See functions.navfunctions.MotionData for details, by default None
        voxelsize : float, optional
            Voxel size of the generated scatter grids, by default 1.5
        voxelsizeZ : float, optional
            Voxel size (Z direction) of the generated scatter grids, by default voxelsize
        surveyspeedKnots : float, optional
            Survey speed in knots, by default 3
        swathdistance : float, optional
            distance between swaths, by default 0.8
        bubbleType : t_Bubbles, optional
            Target type (t_Bubbles.SingleBubble or t_Bubbles.BubbleStream), by default t_Bubbles.SingleBubble
        layerDepths : list, optional
            Depth values where integration layers will be created, by default [80, 105]
        layerSizes : list, optional
            Size of the Layers defined by layerDepths, by default [20,20]
        exagHPR : float, optional
            Exagerate Heave, pitch and roll by this value, by default 1
        maxDepth : float, optional
            Maximum depth for generated targets, by default 125
        maxRange : float, optional
            Maximum range for generated targets, by default 125
        NumBeams : int, optional
            Number of beams in the MBES WCI, by default 256
        NumElements : int, optional
            Number of beamforming elements of the transceiver array, by default 128
        SampleDistance : float, optional
            Physical distance of samples in the MBES WCI [m]
            This must be computed from the sample rate, by default 0.324 m (432 µs * 15000 m/s / 2)
        EffectivePulseWidth: float = 0.5625, (750 µs * 15000 m/s / 2)
            Effective pulse width in meters
            This must be computed from the Effective pulse duration, by default 0.324 m (432 µs * 15000 m/s / 2)
        BaseDirectory : str, optional
            Base directory for the simulation output, by default 'simulation_results'
        load_previous_simresults : bool, optional
            If true, try to load previous sim results (usefull for continuing a simulation), by default True
        verbose : bool, optional
            Print extra output, by default False

        """

        setup = dict()

        self.motionDataPath = motionDataPath

        self.AddPlots     = addPlots
        setup['prefix']        = prefix
        setup['blockAvg']            = blockAvg  # float
        setup['downfactor']            = downfactor  # float
        setup['resfactor']            = resfactor  # float
        setup['windowType']       = windowType  # box, exponential, hann
        setup['idealizedBeampattern'] = idealizedBeampattern  # bool
        setup['equiDist']      = equiDist  # box, exponential, hann
        setup['surveyType']       = surveyType  # flat, nav
        setup['voxelsize']     = voxelsize
        setup['voxelsizeZ']    = voxelsizeZ
        setup['surveyspeedKnots'] = surveyspeedKnots
        setup['swathdistance']     = swathdistance
        setup['bubbleType']       = bubbleType
        setup['layerDepths']   = layerDepths
        setup['layerSizes']    = layerSizes
        setup['exagHPR']       = exagHPR

        setup['MaxDepth']       = maxDepth  #max depth of the bubbles
        setup['MaxRange']       = maxRange  #max recording range of the MBES

        setup['NumBeams']       = NumBeams 
        setup['NumElements']     = NumElements
        setup['SampleDistance'] = SampleDistance
        setup['EffectivePulseWidth'] = EffectivePulseWidth

        self.BaseDirectory = BaseDirectory


        self.call_setups(setup = setup,
                         verbose = verbose,
                         load_previous_simresults = load_previous_simresults)

    @staticmethod
    def get_previous_simreturns(setup : dict, BaseDirectory : str, verbose = False) -> pd.DataFrame:
        """Load the previous simulation results for the given setups

        Parameters
        ----------
        setup : dict
            Dict with SimulationSetups
        BaseDirectory : str
            Base directory where to look for prefix/setups
        verbose : bool, optional
            Print extra output, by default False

        Returns
        -------
        pd.DataFrame
            Previous simulation results

        Raises
        ------
        RuntimeError
            If setups cannot be found or missmatch
        """

        try:
            _setup, SimulationResults = SimulationSetup.load_simreturns(SimulationSetup.create_simulation_name(SimSetup=setup,BaseDirectory=BaseDirectory), verbose)
        except Exception as e:
            print("WARNING[get_previous_simreturns]::load_prevous_simresults Exception:", e)
            _setup = setup
            SimulationResults = pd.DataFrame()

        if _setup.keys() != setup.keys():
            raise RuntimeError(
                'ERROR[get_previous_simreturns]::load_prevous_simresults: Setup Missmatch! Keys are not the same!')

        for k in _setup.keys():
            if _setup[k] != setup[k]:
                raise RuntimeError(
                    'ERROR[get_previous_simreturns]::load_prevous_simresults: Setup Missmatch! [{}]{} != [{}]{}'.format(k,
                                                                                                            _setup[
                                                                                                                k],
                                                                                                            k,
                                                                                                            setup[
                                                                                                                k]))

        return SimulationResults


    def call_setups(self,setup : dict, load_previous_simresults : bool = True, verbose : bool = False):
        """Update the simulation setup using the given setup dict

        Parameters
        ----------
        setup : dict
            Dict with SimulationSetup parameters
        load_previous_simresults : bool, optional
            _description_, by default True
        verbose : bool, optional
            _description_, by default False
        """

        #self.AddPlots   = setup['AddPlots']
        self.NumBeams            = setup["NumBeams"] * setup['downfactor']
        self.NumElements         = setup["NumElements"] * setup['downfactor'] * setup['resfactor']
        self.SampleDistance      = setup["SampleDistance"] / setup['downfactor']
        self.EffectivePulseWidth = setup["EffectivePulseWidth"] / setup['downfactor'] / setup['resfactor'] 

        self.MaxRange = setup['MaxRange']
        self.MaxDepth = setup['MaxDepth']


        self.Multibeam = self.__init_multibeam__(setup['windowType'],
                                                 setup['equiDist'],
                                                 num_beams             = self.NumBeams,
                                                 num_elements          = self.NumElements,
                                                 sample_dist           = self.SampleDistance,
                                                 effective_pulse_length = self.EffectivePulseWidth,
                                                 max_range             = self.MaxRange,
                                                 verbose=verbose)


        self.Survey,self.MotionData = self.__init_navigation__(
            minBeamSteeringAngelRadians = self.Multibeam.beamsteeringangles_radians[0],
            maxBeamSteeringAngelRadians = self.Multibeam.beamsteeringangles_radians[-1],
            swathDistance = setup['swathdistance'],
            speedKnots = setup['surveyspeedKnots'],
            motionDataPath=self.motionDataPath,
            verbose = verbose)

        self.BubbleGenerator,self.TargetGenerator = self.__init_bubbleGeneration__(
            bubbleMode = setup['bubbleType'],
            voxelsize  = setup['voxelsize'],
            maxDepth   = self.MaxDepth,
            minBeamSteeringAngelRadians = self.Multibeam.beamsteeringangles_radians[0],
            maxBeamSteeringAngelRadians = self.Multibeam.beamsteeringangles_radians[-1]
        )

        self.Simulation = self.__init_simulation__(
                multibeam = self.Multibeam,
                blockAvg = setup['blockAvg'],
                voxelsize = setup['voxelsize'],
                voxelsizeZ = setup['voxelsizeZ'],
                survey    = self.Survey,
                useIdealizedBeampattern = setup['idealizedBeampattern'])


        self.SimSetup = setup

        if load_previous_simresults:
            self.SimulationResults = self.get_previous_simreturns(setup=setup,BaseDirectory=self.BaseDirectory)

        else:
            self.SimulationResults = pd.DataFrame()


    # internal functions
    class PGR(object):
        """Internal class for progress reporting on parallel processing
        """
        def __init__(self,sync_val,shared_end):
            self.sync_val = sync_val
            self.total = shared_end
            pass
        def reset(self,total):
            self.sync_val.value = 0
            self.total.value = total
            pass
        def update(self):
            self.sync_val.value += 1

    @staticmethod
    def __simulate_parallel__(simulation,return_list,shared_val,shared_end):
        """Internal function for parallel processing
        """
        pbar = SimulationSetup.PGR(shared_val,shared_end)

        return_list.append((simulation,simulation.simulate3DEchoesGrid(progress=False, pbar = pbar)))
        shared_val.value = -1

        return

    def simulate(self,
                 totalSimulations: int,
                 parallelSimulations: int):
        """Run the simulation

        Parameters
        ----------
        totalSimulations : int
            Number of simulations to run
        parallelSimulations : int
            Number of simulations to run in parallel
        """

        if parallelSimulations > mp.cpu_count():
            print('Warning[simulate]: You chose more cpu cores [{}] than physically exist [{}]'.format(parallelSimulations,mp.cpu_count()))

        totalSimulations = int(round(totalSimulations / parallelSimulations))

        pbars = [tqdm(desc="#" + "{}".format(pid).zfill(3), position=pid +1, total =100) for pid in range(parallelSimulations)]
        pvars = [mp.Value('i',0) for _ in range(parallelSimulations)]
        pmaxs = [mp.Value('i',100) for _ in range(parallelSimulations)]

        with tqdm(desc='Sim Progress',position=parallelSimulations +1) as custom_progress:

            with tqdm(desc='Simulations',total=totalSimulations*parallelSimulations,position=parallelSimulations +2) as simpgrogress:


                for simsnum in range(totalSimulations):
                    simulations = []

                    custom_progress.reset(total=8)
                    #print('\rSimulation: Init simulations                                 ',end='\r')
                    custom_progress.set_postfix_str('Init simulations')

                    for sim_targets in self.TargetGenerator.iterarte(parallelSimulations):

                        sim = deepcopy(self.Simulation)

                        sim_survey = self.get_survey()

                        sim_survey.setRandomOffsetX(self.SimSetup['voxelsize'])
                        sim_survey.setRandomOffsetY(self.SimSetup['voxelsize'])
                        sim_survey.setRandomOffsetZ(self.SimSetup['voxelsize'])

                        sim.setSurvey(sim_survey)
                        sim.setTargets(sim_targets)

                        simulations.append(sim)

                    try:
                        for p in self.processes:
                            p.close()
                    except:
                        pass


                    self.processes = []
                    manager = mp.Manager()
                    return_list = manager.list()

                    for pbar in pbars:
                        pbar.clear()
                        pbar.reset(total = pbar.total)

                    gc.collect()
                    custom_progress.update()
                    custom_progress.set_postfix_str('Start processes')
                    for pid in range(parallelSimulations):
                        p = mp.Process(target=self.__simulate_parallel__, args=(simulations[pid],return_list, pvars[pid],pmaxs[pid]))
                        p.start()
                        self.processes.append(p)

                    self.resulting_scatterGrids=[]

                    custom_progress.update()
                    custom_progress.set_postfix_str('Process progress')
                    while True:
                        max_reached = np.zeros(len(pvars))
                        for pid,pv in enumerate(pvars):

                            if pv.value < 0  or pv.value >= pbars[pid].total:
                                max_reached[pid] = 1
                                pbars[pid].update(pbars[pid].total-pbars[pid].n)
                                pbars[pid].refresh()
                            else:
                               if pbars[pid].total != pmaxs[pid].value:
                                   pbars[pid].reset(total=pmaxs[pid].value)
                               pbars[pid].update(pv.value - pbars[pid].n)
                               pbars[pid].refresh()


                        if np.sum(max_reached) >= len(pvars):
                            break

                        time.sleep(0.1)

                    custom_progress.update()
                    custom_progress.set_postfix_str('Joining processes')
                    for p in self.processes:
                        p.join()

                    custom_progress.update()
                    custom_progress.set_postfix_str('Closing processes')
                    for p in self.processes:
                        p.close()

                    custom_progress.update()
                    custom_progress.set_postfix_str('Retreiving simresults')
                    for ri,rl in enumerate(return_list):
                        pbars[ri].reset()
                        self.resulting_scatterGrids.append(rl)

                    custom_progress.set_postfix_str('Collecting garbage')
                    gc.collect()

                    custom_progress.update()
                    custom_progress.set_postfix_str('Processing Simresults')
                    # run the simulations in parallel than iterate through them
                    for sim,scatterGrids in self.resulting_scatterGrids:

                        simret = pd.Series(dtype=object)


                        simret['sample_ranges'] = deepcopy(sim.Multibeam.sampleranges)
                        simret['survey'] = deepcopy(sim.Survey)


                        sim_targets = sim.Targets
                        simret['target_x'] = np.mean(sim_targets.x)
                        simret['target_y'] = np.mean(sim_targets.y)
                        simret['target_z'] = np.mean(sim_targets.z)
                        simret['simTargets'] = deepcopy(sim_targets)

                        r, tx_angle, rx_angle = sim.Multibeam.get_target_range_tx_rx(sim_targets.x,
                                                                                     sim_targets.y,
                                                                                     sim_targets.z)

                        simret['target_range'] = np.mean(r)
                        simret['target_rx_angle'] = np.mean(rx_angle)

                        max_db_diffs = [np.nan,-2.5,-3,-5,-6,-7.5,-8,-9,-10,-12,-12.5,-13,-15,-16,-17.5,-18,-20,-30,-40,-50,-60,-70,-80,-90]
                        simret['max_db_diffs'] = deepcopy(max_db_diffs)

                        target_detect = {}
                        for method, scattergrid in scatterGrids.items():
                            simret[method] = scattergrid.TotalValue

                            target_detect[method] = bubbles.Targets.init_empty()

                            try:
                                max_db_diff = None
                                max_val_db = 10*math.log10(np.nanmax(scattergrid.ImageAvg))

                                #Determine target position
                                for max_db_diff in max_db_diffs:
                                        tp = scattergrid.get_target_pos(10**((max_val_db+max_db_diff)/10))

                                        target_detect[method].append(*tp,1.0)

                                        simret[method + '[' + str(max_db_diff) + ']'] = scattergrid.getTotalvalue(10**((max_val_db+max_db_diff)/10))

                                        simret['detect[' + method + '][' + str(max_db_diff) + ']_x'] = tp[0]
                                        simret['detect[' + method + '][' + str(max_db_diff) + ']_y'] = tp[1]
                                        simret['detect[' + method + '][' + str(max_db_diff) + ']_z'] = tp[2]

                                        dx = np.mean(sim_targets.x) - tp[0]
                                        dy = np.mean(sim_targets.y) - tp[1]
                                        dz = np.mean(sim_targets.z) - tp[2]
                                        dist =  math.sqrt(dx * dx + dy * dy + dz * dz)

                                        simret['detect_dist[' + method + '][' + str(max_db_diff) + ']'] = dist
                                        simret['detect_dist_x[' + method + '][' + str(max_db_diff) + ']'] = dx
                                        simret['detect_dist_y[' + method + '][' + str(max_db_diff) + ']'] = dy
                                        simret['detect_dist_z[' + method + '][' + str(max_db_diff) + ']'] = dz
                            except Exception as e:
                                pass

                        simret['methods'] = list(scatterGrids.keys())
                        simret['trueValue'] = sim_targets.sum()

                        for layer_depth,layer_size in zip(self.SimSetup['layerDepths'],self.SimSetup['layerSizes']):

                            cutScatterGrids, (
                            layer_z_extend, true_layer_size_z, layer_z_coordinates) = scatterGrids.cutDepthLayer(
                                layer_depth, layer_size)

                            layerTargets = sim_targets.cutDepthLayer(*layer_z_extend)
                            simret[str(round(layer_depth,2)) + '|' + str(round(layer_size,2)) + ' - trueValue'] = layerTargets.sum()


                            for method, cutScatterGrid in cutScatterGrids.items():
                                simret[str(round(layer_depth,2)) + '|' + str(round(layer_size,2)) + ' - ' + method] = cutScatterGrid.TotalValue

                                simret[str(round(layer_depth,2)) + '|' + str(round(layer_size,2)) + ' -layerMean- ' + method] = cutScatterGrid.TotalValueLayer

                                try:
                                    max_db_diff = None
                                    max_val_db = 10 * math.log10(np.nanmax(cutScatterGrid.ImageAvg))

                                    # threshhold values for layers
                                    for max_db_diff in max_db_diffs:
                                        simret[str(round(layer_depth,2)) + '|' + str(round(layer_size,2)) + ' - ' + method + '[' + str(max_db_diff) + ']'] = cutScatterGrid.getTotalvalue(
                                            10 ** ((max_val_db + max_db_diff) / 10))

                                        simret[str(round(layer_depth,2)) + '|' + str(round(layer_size,2)) + ' -layerMean- ' + method + '[' + str(max_db_diff) + ']'] = cutScatterGrid.getTotalvalueLayer(
                                            10 ** ((max_val_db + max_db_diff) / 10))
                                except Exception as e:
                                    pass
                                 
                        # create plots
                        if self.AddPlots:
                            for name, scattergrid in scatterGrids.items():
                                fig = plt.Figure()
                                fig.suptitle = 'resample - ' + name
                                scattergrid.plot(
                                    fig,
                                     targets_color   = [
                                        (sim_targets,'grey'),
                                        (target_detect[name],'red')
                                     ],
                                    show_wci=True,
                                    show_echo=True,
                                    show_map=True,
                                    todB=True,
                                    mindBVal=-70
                                )
                                simret['resample - ' + name] = fig

                        self.SimulationResults = self.SimulationResults.append(simret, ignore_index=True)

                    custom_progress.update()
                    custom_progress.set_postfix_str('Saving simresults')
                    self.save_simreturns(self.SimSetup,self.SimulationResults,self.get_simulation_name())

                    custom_progress.update()
                    custom_progress.set_postfix_str('Done')
                    simpgrogress.update((simsnum+1)*parallelSimulations-simpgrogress.n)

        for pbar in pbars:
            pbar.close()
            pbar.clear()

        del pbars

        del simpgrogress
        del custom_progress
        gc.collect()


    def get_simulation_name(self) -> str:
        """Create name and directories / sub directories for the simulation (based on the SimSetup)

        Returns
        -------
        str
            simulation name
        """
        return self.create_simulation_name(self.SimSetup,self.BaseDirectory)

    @staticmethod
    def create_simulation_name(SimSetup,BaseDirectory) -> str:
        """Internal function
        Create name and directories / sub directories for the simulation (based on the SimSetup)

        Returns
        -------
        str
            simulation name
        """

        dir_keys = ['prefix','downfactor','resfactor','equiDist','bubbleType']

        dir = deepcopy(BaseDirectory) + '/'
        for k in dir_keys:
            dir +=  '{}[{}]/'.format(k,str(SimSetup[k]))

        sim_name = ''
        for k, v in SimSetup.items():
            if not k in dir_keys:
                if not k in ['layerSizes', 'layerDepths']:
                    sim_name += '{}[{}]'.format(k, str(v))
                else:
                    try:
                        sim_name += '{}[{}[{}|{}]]'.format(k,len(v),min(v),max(v))
                    except:
                        sim_name += '{}[{}]'.format(k,str(v))

        os.makedirs(dir, exist_ok=True)
        sim_name = dir + sim_name + '.pd'
        sim_name = sim_name.replace(' ','')

        return sim_name

    def get_survey(self,exagHPR:float = None):
        """Internal function (called by simulate)

        Parameters
        ----------
        exagHPR : float, optional
            heave, pitch roll exageration, by default None

        Returns
        -------
        Survey
            Survey with random x,y,z offset and (if not t_Survey.IdealMotion) modified motion data (based on exagHPR)
        """
        self.Survey.setRandomOffsetX(self.SimSetup['voxelsize'])
        self.Survey.setRandomOffsetY(self.SimSetup['voxelsize'])
        self.Survey.setRandomOffsetZ(self.SimSetup['voxelsize'])

        if exagHPR is None:
            exagHPR = self.SimSetup['exagHPR']

        if self.SimSetup['surveyType'] == t_Survey.IdealMotion:
            return deepcopy(self.Survey)

        return self.MotionData.get_modified_survey(self.Survey,
                                                       #exaggerate_yaw  = exagHPR,
                                                       exaggerate_heave = exagHPR,
                                                       exaggerate_pitch = exagHPR,
                                                       exaggerate_roll  = exagHPR)

    @staticmethod
    def load_simreturns(simulation_path : str,
    verbose: bool =  False):
        """Load simulation results from file

        Parameters
        ----------
        simulation_path : str
            path to simulation file (hdf5 (.h5) or pickle (.pd))
        verbose : bool, optional
            Print extra output, by default False

        Returns
        -------
        (dict,pd.DataFrame)
            simulation setup parameters, simulation results
        """
        # initi
        simreturns = pd.DataFrame()

        if simulation_path.endswith('.h5'):
            hdf5 = True
        else:
            hdf5 = False

        if hdf5:
            setup = dict(pd.read_hdf(simulation_path, 'setup')[0])
            simreturns = pd.read_hdf(simulation_path, 'simresults')
        else:
            with open(simulation_path, 'rb') as ifi:
                setup      = pickle.load(ifi)
                simreturns = pickle.load(ifi)

        if verbose:
            print('Length of previous simreturns:',len(simreturns))

        return setup,simreturns

    @staticmethod
    def save_simreturns(
            setup : dict,
            simulation_results : pd.DataFrame,
            simulation_path: str,
            hdf5 = False,
            verbose: bool = False):
        """Save simulation results to file

        Parameters
        ----------
        setup : dict
            Setup parameters
        simulation_results : pd.DataFrame
            Simulation results
        simulation_path : str
            path to simulation file (hdf5 (.h5) or pickle (.pd))
        hdf5 : bool, optional
            Safe the file as hdf5 (.h5) or pandas dataframe, by default False
        verbose : bool, optional
            Print extra output, by default False
        """
        # initi
        simreturns = pd.DataFrame()

        if verbose:
            print('saving simulations results to:',simulation_path)

        if hdf5:
            if not simulation_path.endswith('.h5'):
                simulation_path = '.'.join(simulation_path.split('.')[:-1]) + '.h5'
            pdsetup = pd.DataFrame(pd.Series(setup))
            pdsetup.to_hdf(simulation_path, key='setup', mode='w')
            simulation_results.to_hdf(simulation_path, key='simresults', mode='a')
        else:
            if not simulation_path.endswith('.pd'):
                '.'.join(simulation_path.split('.')[:-1]) + '.pd'
            with open(simulation_path, 'wb') as ofi:
                pickle.dump(setup,ofi)
                pickle.dump(simulation_results,ofi)

    @staticmethod
    def __init_multibeam__(windowType : t_Window,
                           equiDist   : bool = True,
                           num_beams : int = 256,
                           num_elements : int = 128,
                           sample_dist : float = 0.324,
                           effective_pulse_length : float = 0.375,
                           max_range : float = 125,
                           verbose = False):
        """Internal function
        Initialize multibeam sonar with the given parameters
        """

        elements = int(num_elements)

        if windowType == t_Window.Box:
            window = signal.windows.boxcar(elements)
        elif windowType == t_Window.Hann:
            window = signal.windows.hann(elements)
        elif windowType == t_Window.Exponential:
            window = signal.windows.exponential(elements, tau=elements / 2)
        else:
            raise RuntimeError("Unknown window:", str(windowType))

        if not equiDist:
            #num_beams equiangluar beams from -60° -> 60°
            steeringangles = np.linspace(-60, 60, int(num_beams))
        else:
            #num_beams equidistant beams from -60° -> 60°
            # min,max perpendicular distance at 1 meter distance
            min_y = math.tan(math.radians(-60))
            max_y = math.tan(math.radians(60))

            # arrange 256 positions between min_y and max_y
            ys = np.linspace(min_y, max_y, num_beams)

            # compute angles for the positions
            steeringangles = np.degrees(np.arctan(ys))

        multibeam = mb.Multibeam(
            beamsteeringangles_degrees = steeringangles,
            sampleranges               = np.arange(1, max_range, sample_dist),
            effective_pulse_length      = effective_pulse_length,
            window                     = window,
            progress                   = verbose
        )

        return multibeam

    @staticmethod
    def __init_navigation__(minBeamSteeringAngelRadians,
                            maxBeamSteeringAngelRadians,
                            speedKnots,
                            swathDistance,
                            motionDataPath,
                            verbose = False):
        """Internal function
        Initialize the navigation object with the given parameters
        """
        depth = 125

        min_x = -abs(math.sin(minBeamSteeringAngelRadians) * depth)
        max_x = abs(math.sin(maxBeamSteeringAngelRadians) * depth)

        survey = nav.Survey.from_ping_distance(min_x=min_x,
                                               max_x=max_x,
                                               ping_spacing=swathDistance,
                                               speed_knots=speedKnots)
        if motionDataPath is None:
            surveyNav = survey
            MotionData = None
        else:
            try:
                MotionData = nav.MotionData(motion_data_path=motionDataPath)
                surveyNav      = MotionData.get_modified_survey(survey)
            except Exception as e:
                print("WARNING: Could not open navigation data! Exception:",e)
                surveyNav = survey
                MotionData = None

        if verbose:
            print(round(min_x, 2), round(max_x, 2))

        return survey,MotionData

    @staticmethod
    def __init_bubbleGeneration__(bubbleMode : t_Bubbles,
                                voxelsize : float,
                                maxDepth : float,
                                minBeamSteeringAngelRadians,
                                maxBeamSteeringAngelRadians):
        """Internal function
        Initialize the bubble generator with the given parameters
        """

        bubbleGenerator = bubbles.BubbleGenerator(
            sigma_val=0.3,  # variation of value
            sigma_x=1,  # for bubble stream: sigma variation in x direction in m
            sigma_y=1   # for bubble stream: sigma variation in y direction in m
        )

        if bubbleMode == t_Bubbles.SingleBubble:
            tgen = TargetGenerator(
                bubbleGenerator.generate_bubbles_within_cylindrical_section_along_path,
                start_x=-voxelsize / 2,
                end_x=voxelsize / 2,
                min_range=1,
                max_range=maxDepth,
                min_beamsteeringangle=math.degrees(minBeamSteeringAngelRadians),
                max_beamsteeringangle=math.degrees(maxBeamSteeringAngelRadians),
                nbubbles=1
            )
        elif bubbleMode == t_Bubbles.BubbleStream:
            min_y = math.atan(minBeamSteeringAngelRadians) * maxDepth
            max_y = math.atan(maxBeamSteeringAngelRadians) * maxDepth

            tgen = TargetGenerator(
                bubbleGenerator.generate_bubblestreams_within_cylindrical_section_along_path,
                start_x=-voxelsize / 2,
                end_x=voxelsize / 2,
                min_range=np.nan,
                max_range=np.nan,
                min_beamsteeringangle=np.nan,
                max_beamsteeringangle=np.nan,
                min_y=min_y,
                max_y=max_y,
                min_z=0,
                max_z=maxDepth,
                zdist_list=[0.1]
            )
        else:
            raise RuntimeError("BubbleMode: AAAaaaah " + str(bubbleMode))


        return bubbleGenerator,tgen

    @staticmethod
    def __init_simulation__(multibeam,
                            voxelsize,
                            voxelsizeZ,
                            survey,
                            useIdealizedBeampattern,
                            blockAvg):
        """Internal function
        Initialize the simulation with the given parameters
        """
        simulation = SIM.Simulation()

        simulation.setMultibeam(multibeam)
        simulation.setGridResolution(voxelsize)

        if voxelsizeZ is not None:
            simulation.setGridResolutionZ(voxelsizeZ)

        simulation.setSurvey(survey)

        simulation.setUseIdealizedBeampattern(useIdealizedBeampattern)

        if blockAvg:
            simulation.setMethods(
                [
                    SIM.GriddingMethod.sv,
                    SIM.GriddingMethod.sv_int_lin
                ])
        else:
            simulation.setMethods(
                [
                    SIM.GriddingMethod.sv_int_lin,
                ])

        return simulation



if __name__ == '__main__':
    init_angular_resolution(180 * 90)

    downfactor = 0.33
    resfactor = 1
    voxsize = 1.5

    SingleTarget = True

    beam_pattern = t_Window.Exponential

    setup = SimulationSetup(
        addPlots=True,
        prefix='examples',
        downfactor=downfactor,
        blockAvg=False,
        resfactor=resfactor,
        windowType=beam_pattern,
        idealizedBeampattern=False,
        equiDist=False,
        motionDataPath="../test_data/m143_l0154_motion.csv",

        surveyType=t_Survey.IdealMotion,
        voxelsize=voxsize / downfactor,
        voxelsizeZ=voxsize / downfactor,
        surveyspeedKnots=3,
        swathdistance=0.8 / downfactor,
        layerDepths=[],
        layerSizes=[],
        bubbleType=t_Bubbles.SingleBubble,
        exagHPR=1,
        BaseDirectory='GEOMAR_simresults',

        load_previous_simresults=False,
        verbose=True)