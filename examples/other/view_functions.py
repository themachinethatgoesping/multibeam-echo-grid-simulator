#!/home/ssd/opt/anaconda3/bin/python3

# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

from matplotlib import pyplot as plt

import math
import numpy as np
from copy import deepcopy
from sys import getsizeof
from IPython.display import clear_output

from scipy import signal
from scipy import stats

from numba import njit, prange

import random
import time
import importlib
import os
import pickle
import shutil as sh
import gc

import mbes_sim.functions.helperfunctions as hlp
import mbes_sim.functions.create_bubblesfunctions as bubbles
import mbes_sim.functions.navfunctions as nav
import mbes_sim.functions.transformfunctions as tf
import mbes_sim.functions.pulsefunctions as pf
import mbes_sim.functions.gridfunctions as gf
from mbes_sim.mbes import Multibeam
import mbes_sim.scattergrid as sg
import mbes_sim.simulation as SIM
import mbes_sim.simulationfunctions as SIMFUN

import view_functions as vf

from enum import IntEnum

# from tqdm.notebook import tqdm
from tqdm import tqdm

def find_files(simresults_path,search_for=[], print_vars=[],not_search_for=['noplots']):
    files = []

    for r, d, f in os.walk(simresults_path):
        for file in f:
            if file.endswith('.pd'):
                for sf in not_search_for:
                    if sf in file or sf in r:
                        break
                else:
                    for sf in search_for:
                        if not sf in file and not sf in r:
                            break
                    else:
                        files.append(r + '/' + file)

    files.sort()

    for i, f in enumerate(files):
        # print('({}) : {}'.format(i,f))
        print('File nr: ({})'.format(i))

        setups = f.replace('SIM', '')
        setups = setups.replace('.pd', '')
        setups = setups.split(']')

        if i == 0:
            bsetups = deepcopy(setups)

        for s in setups:

            if print_vars:
                cont = True
                for pv in print_vars:
                    if pv in s:
                        cont = False
                if cont:
                    continue
            else:
                if i > 0:
                    if s in bsetups:
                        continue

            if s.strip():
                s = s.replace('_', "")
                s = s.replace('/', "")

                print('    :', s + ']')

    return files

CACHED_FILES={}

def open_file(file, verbose=True, cache = False, no_plots=True, hdf5=True):
    global CACHED_FILES

    #print('in file', file)
    if no_plots:
        filename = file.split('/')[-1]
        filepath = '/'.join(file.split('/')[:-1])
        if not filename.startswith('noplots_'):
            plt_filename = deepcopy(filename)
            filename = 'noplots_' + filename
        else:
            plt_filename = filename.split('noplots_')[-1]

        file     = filepath + '/' + filename
        plt_file = filepath + '/' + plt_filename
        
        if hdf5:
            file = '.'.join(file.split('.')[:-1]) + '.h5'
            
        

        #print('file',file)
        #print('plt_file',plt_file)

        if not os.path.exists(file):
            setup, simreturns = SIMFUN.SimulationSetup.load_simreturns(plt_file, verbose=verbose)

            plt_keys = [k for k in simreturns.keys() if 'resample' in k]
            for k in plt_keys:
                del simreturns[k]

            SIMFUN.SimulationSetup.save_simreturns(setup = setup,
                                                   simulation_results = simreturns,
                                                   simulation_path = file,
                                                   verbose=verbose,
                                                   hdf5 = hdf5)


    if file in CACHED_FILES.keys():
        setup, simreturns = CACHED_FILES[file]
    else:
        setup, simreturns = SIMFUN.SimulationSetup.load_simreturns(file, verbose=verbose)

        if cache:
            CACHED_FILES[file] = setup, simreturns

    if verbose:
        print('-- Opened simreturns --')
        print(file)
        print(len(simreturns))

    return setup, simreturns


def cache_files(files,no_plots=True, hdf5=True):
    global CACHED_FILES;
    for f in tqdm(files, desc='caching files'):
        try:
            open_file(f, verbose=False, cache = True,no_plots=no_plots,hdf5=hdf5)
        except Exception as e:
            print('Failed for file:',f)
            print(e)

def clear_cached_files():
    CACHED_FILES = {}
    gc.collect()

PLOT_NR = 0
def plot_resampled(simreturns, main,plot_nr = None):
    global PLOT_NR

    plot_keys = []
    for k in simreturns.keys():
        if k.startswith('resample'):
            plot_keys.append(k)

    if plot_nr is None:
        PLOT_NR += 1
        if PLOT_NR >= len(simreturns):
            PLOT_NR = 0

        plot_nr = PLOT_NR

    if plot_nr >= len(simreturns):
        plot_nr = 0
    PLOT_NR = plot_nr


    for k in plot_keys:
        fig = simreturns[k][plot_nr]
        fig.axes[0].set_title(str(plot_nr) + ': ' + k)
        axes = fig.axes
        main.addfig(k, simreturns[k][plot_nr], select_figure=True)


def plot_simreturns(create_figure,df, name, sort_by=None, methods=None, methods_not=None, ax = None, hist_ax = None, hist_n = 0, kwargs_list=[],hist_kwargs_list=[]):

    if ax is None:
        fig = create_figure(name)
        ax = fig.subplots()
        ax.set_title(name)

    if methods is None:
        methods = df['methods'].to_numpy()[1].copy()

    if methods_not is not None:
        for m in methods_not:
            for i, meth in enumerate(methods):
                if meth == m:
                    del methods[i]
                    break

    if hist_n > 0:
        if hist_ax is None:
            hist_fig = create_figure('hist-'+name)
            hist_ax = hist_fig.subplots()
            hist_ax.set_title('hist-'+name)

        for nr, method in enumerate(methods):
            kwargs = {}
            if len(hist_kwargs_list) > nr:
                kwargs = hist_kwargs_list[nr]
            hist_ax.hist(df[method],hist_n, label=method, **kwargs)

        hist_ax.set_xlabel('measured_sigma')
        hist_ax.legend()

    if sort_by is None:
        for nr,method in enumerate(methods):
            kwargs={}
            if len(kwargs_list) > nr:
                kwargs = kwargs_list[nr]
            ax.plot(df.index, df[method], label=method,**kwargs)

        ax.set_xlabel('simulation nr.')

    else:
        df_copy = df.sort_values(sort_by)
        for nr,method in enumerate(methods):
            kwargs={}
            if len(kwargs_list) > nr:
                kwargs = kwargs_list[nr]
            ax.plot(df_copy[sort_by], df_copy[method], label=method,**kwargs)

        ax.set_xlabel(sort_by)

    ax.axhline(y=1,linestyle='--',c='red',linewidth=1.5)
    ax.legend(loc='upper left')

    ax.set_ylabel('measured_sigma')
    ax.set_ylim(0.8,1.45)



    return ax


def do_plot_simreturns(create_figure,simreturns,filtered_simreturns,methods_not,kwargs_list=[],hist_n = 0,hist_kwargs_list=[]):
    ax = plot_simreturns(create_figure,simreturns, 'simreturns - everything',kwargs_list=kwargs_list)
    ax = plot_simreturns(create_figure,simreturns, 'simreturns - default', methods_not=methods_not,kwargs_list=kwargs_list, hist_n=hist_n,hist_kwargs_list=hist_kwargs_list)
    ax = plot_simreturns(create_figure,simreturns, 'simreturns - target_range', 'target_range', methods_not=methods_not,kwargs_list=kwargs_list)
    ax = plot_simreturns(create_figure,simreturns, 'simreturns - target_x', 'target_x', methods_not=methods_not,kwargs_list=kwargs_list)
    ax = plot_simreturns(create_figure,simreturns, 'simreturns - target_y', 'target_y', methods_not=methods_not,kwargs_list=kwargs_list)
    ax = plot_simreturns(create_figure,simreturns, 'simreturns - target_z', 'target_z', methods_not=methods_not,kwargs_list=kwargs_list)
    ax = plot_simreturns(create_figure,simreturns, 'simreturns - target_rx_angle', 'target_rx_angle', methods_not=methods_not,kwargs_list=kwargs_list)

    ax = plot_simreturns(create_figure,filtered_simreturns, 'filtered simreturns - default', methods_not=methods_not,kwargs_list=kwargs_list, hist_n=hist_n,hist_kwargs_list=hist_kwargs_list)
    ax = plot_simreturns(create_figure,filtered_simreturns, 'filtered simreturns - target_range', 'target_range',
                         methods_not=methods_not,kwargs_list=kwargs_list)
    ax = plot_simreturns(create_figure,filtered_simreturns, 'filtered simreturns - target_x', 'target_x', methods_not=methods_not,kwargs_list=kwargs_list)
    ax = plot_simreturns(create_figure,filtered_simreturns, 'filtered simreturns - target_y', 'target_y', methods_not=methods_not,kwargs_list=kwargs_list)
    ax = plot_simreturns(create_figure,filtered_simreturns, 'filtered simreturns - target_z', 'target_z', methods_not=methods_not,kwargs_list=kwargs_list)
    ax = plot_simreturns(create_figure,filtered_simreturns, 'filtered simreturns - target_rx_angle', 'target_rx_angle',
                         methods_not=methods_not,kwargs_list=kwargs_list)

    methods = filtered_simreturns['methods'].to_numpy()[0]

    for method in methods:
        plot_simreturns(create_figure,filtered_simreturns, 'simreturns - default - ' + method, methods=[method],kwargs_list=kwargs_list, hist_n=hist_n,hist_kwargs_list=hist_kwargs_list)


def print_simreturns(df, methods=None,max_db_diffs=None, show_db = True,prefix=''):
    if methods is None:
        methods = df['methods'].to_numpy()[0]

    minMethodNameSize = max([len(m) for m in methods])

    method_keys = []
    for method in methods:
        method_keys.append(method)
        for k in df.keys():
            if k.startswith(method):
                use = False
                if max_db_diffs is not None:
                    for max_db_diff in max_db_diffs:
                        if '[{}]'.format(max_db_diff) in k:
                            use = True
                else:
                    use = True

                if use:
                    method_keys.append(k)
                    
    
    #if not show_db:
    #    print('N Measurements | Bias | Std. Error | Std. Dev | Min | Max')

    for method in method_keys:
        preprefix = prefix + 'Bubbles Grid'
        preprefix += '[{:MMMs}]'.replace('MMM', str(int(minMethodNameSize))).format(method)

        checkvalue_1 = max(np.abs([min(df["trueValue"]),max(df["trueValue"])]))
        TrueValue    = df["trueValue"].mean()
        
        #TotalValue = (df[method]/df["trueValue"]).mean()
        TotalValue      = df[method].mean()
        TotalValueDiff  = TotalValue - TrueValue
        TotalValueError = 2*np.std(df[method]) / np.sqrt(len(df))
        TotalValue_2SD  = 2*np.std(df[method])
        
        tmpd = deepcopy(df[method].to_numpy())
#         tmpd = tmpd[tmpd > TrueValue-TotalValue_2SD]
#         tmpd = tmpd[tmpd < TrueValue+TotalValue_2SD]
        tmpd = tmpd[tmpd > TotalValue-TotalValue_2SD]
        tmpd = tmpd[tmpd < TotalValue+TotalValue_2SD]
        Interval_2SD = len(tmpd)/len(df[method])
        
        TotalValue_median = df[method].median()        
        TotalValue_min    = np.min(df[method])
        TotalValue_max    = np.max(df[method])
                
        
        string = (preprefix 
                  + ': {} || {} |' 
                  + '| {:+5.1f} % | ±{:5.1f} % |'
                  + '| ±{:5.1f} % | {:4.0f} % |'
                  + '| {:+6.1f} % | {:+5.1f} % | {:+5.1f} % | {:+5.1f} % | {:+6.1f} %').format(
                  #+ '| {:+5.0f} % | {:+4.0f} % | {:+4.0f} % | {:+4.0f} % | {:+5.0f} %').format(
                checkvalue_1,
                len(df[method]),
                100 * (TotalValueDiff/TrueValue),
                100 * (TotalValueError/TrueValue),
                100 * (TotalValue_2SD/TrueValue),
                100 * Interval_2SD,
                100 * (np.percentile(df[method], 0)-TrueValue)/TrueValue,
                100 * (np.percentile(df[method], 25)-TrueValue)/TrueValue,
                100 * (np.percentile(df[method], 50)-TrueValue)/TrueValue,
                100 * (np.percentile(df[method], 75)-TrueValue)/TrueValue,
                100 * (np.percentile(df[method], 100)-TrueValue)/TrueValue,
            )
        print(string)


def plot_detect_dist(create_figure,df, name, sort_by=None, methods=None, methods_not=None, max_db_diffs=None):
    fig = create_figure(name)
    ax = fig.subplots()
    ax.set_title(name)

    if methods is None:
        methods = df['methods'].to_numpy()[1].copy()

    if methods_not is not None:
        for m in methods_not:
            for i, meth in enumerate(methods):
                if meth == m:
                    del methods[i]
                    break

    detect_dist_keys = []
    for k in df.keys():
        if 'detect_dist[' in k:
            use = False
            if methods is not None:
                for method in methods:
                    if '[{}]'.format(method) in k:

                        if max_db_diffs is not None:
                            for max_db_diff in max_db_diffs:
                                if '[{}]'.format(max_db_diff) in k:
                                    use = True
                        else:
                            use = True
            else:
                use = True

            if use:
                detect_dist_keys.append(k)

    if sort_by is None:
        for k in detect_dist_keys:
            ax.plot(df[k], label=k)
        ax.set_xlabel('simulation runs')

    else:
        df_copy = df.sort_values(sort_by)
        for k in detect_dist_keys:
            ax.plot(df_copy[sort_by], df_copy[k], label=k)
        ax.set_xlabel(sort_by)


    return ax


def do_plot_detect_dist(create_figure,simreturns,filtered_simreturns,methods_not,max_db_diffs):
    ax = plot_detect_dist(create_figure,simreturns, 'detect_dist - everything', max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,simreturns, 'detect_dist - default', methods_not=methods_not, max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,simreturns, 'detect_dist - target_range', 'target_range', methods_not=methods_not,
                          max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,simreturns, 'detect_dist - target_x', 'target_x', methods_not=methods_not,
                          max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,simreturns, 'detect_dist - target_y', 'target_y', methods_not=methods_not,
                          max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,simreturns, 'detect_dist - target_z', 'target_z', methods_not=methods_not,
                          max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,simreturns, 'detect_dist - target_rx_angle', 'target_rx_angle', methods_not=methods_not,
                          max_db_diffs=max_db_diffs)

    ax = plot_detect_dist(create_figure,filtered_simreturns, 'filtered detect_dist - default', methods_not=methods_not,
                          max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,filtered_simreturns, 'filtered detect_dist - target_range', 'target_range',
                          methods_not=methods_not, max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,filtered_simreturns, 'filtered detect_dist - target_x', 'target_x', methods_not=methods_not,
                          max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,filtered_simreturns, 'filtered detect_dist - target_y', 'target_y', methods_not=methods_not,
                          max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,filtered_simreturns, 'filtered detect_dist - target_z', 'target_z', methods_not=methods_not,
                          max_db_diffs=max_db_diffs)
    ax = plot_detect_dist(create_figure,filtered_simreturns, 'filtered detect_dist - target_rx_angle', 'target_rx_angle',
                          methods_not=methods_not, max_db_diffs=max_db_diffs)

    methods = filtered_simreturns['methods'].to_numpy()[0]

    for method in methods:
        plot_detect_dist(create_figure,filtered_simreturns, 'simreturns - default - ' + method, methods=[method])


def print_detect_dist(df, methods=None, max_db_diffs=None):
    if methods is None:
        methods = df['methods'].to_numpy()[0]

    minMethodNameSize = max([len(m) for m in methods])

    for method in methods:
        prefix = 'Bubbles Grid'
        prefix += '[{:MMMs}]'.replace('MMM', str(int(minMethodNameSize))).format(method)

        detect_dist_keys = []
        for k in df.keys():
            if 'detect_dist[' in k:
                use = False
                if '[{}]'.format(method) in k:

                    if max_db_diffs is not None:
                        for max_db_diff in max_db_diffs:
                            if '[{}]'.format(max_db_diff) in k:
                                use = True
                    else:
                        use = True

                if use:
                    detect_dist_keys.append(k)

        # print(detect_dist_keys)

        for k in detect_dist_keys:
            string = prefix + '[' + k.split('[')[-1] + ": "
            dist = df[k].mean()
            dist_std = np.std(df[k])

            dist_max = np.nanmax(df[k])

            string += "{:5.2f}  | std {:5.2f} | max: {:5.2f} ||".format(dist, dist_std, dist_max)

            print(string)


def filter_simreturns(simreturns,min_range,max_range,min_rx_angle,max_rx_angle):
    filtered_simreturns = simreturns.copy()
    filtered_simreturns = filtered_simreturns[filtered_simreturns['target_range'] <= max_range]
    filtered_simreturns = filtered_simreturns[filtered_simreturns['target_range'] >= min_range]
    filtered_simreturns = filtered_simreturns[filtered_simreturns['target_rx_angle'] >= min_rx_angle]
    filtered_simreturns = filtered_simreturns[filtered_simreturns['target_rx_angle'] <= max_rx_angle]
    return filtered_simreturns


def filter_simreturns_stream(setup,simreturns,min_target_y,max_target_y):
    simreturns['layers'] = [setup['layerDepths'] for _ in simreturns['target_y']]

    for k in simreturns.keys():
        if 'dist[' in k:
            dist_k_fields = k.split('dist[')

            dist_k_x = dist_k_fields[0] + 'dist_x[' + dist_k_fields[1]
            dist_k_y = dist_k_fields[0] + 'dist_y[' + dist_k_fields[1]
            dist_k_h = dist_k_fields[0] + 'dist_horizontal[' + dist_k_fields[1]

            simreturns[dist_k_h] = np.sqrt(
                simreturns[dist_k_x] * simreturns[dist_k_x] + simreturns[dist_k_y] * simreturns[dist_k_y])

            # print(simreturns[dist_k])

    filtered_simreturns = simreturns.copy()
    filtered_simreturns = filtered_simreturns[filtered_simreturns['target_y'] <= max_target_y]
    filtered_simreturns = filtered_simreturns[filtered_simreturns['target_y'] >= min_target_y]

    return filtered_simreturns


def print_simreturns_stream(setup, df, methods=None, layerdepths=None, layersizes=None,max_db_diffs=None,prefix='',layermethods=[' - ']):
    if layerdepths is None:
        layerdepths = np.array(setup['layerDepths'])
        layersizes = np.array(setup['layerSizes'])

    if methods is None:
        methods = df['methods'].to_numpy()[1]

    method_keys = []
    for method in methods:
        if max_db_diffs is None:
            method_keys.append(method)
        else:
            for max_db_diff in max_db_diffs:
                for k in df.keys():
                    if k.startswith(method):
                        use = False
                        if max_db_diff is not None:
                            if '[{}]'.format(max_db_diff) in k:
                                use = True
                                break
                        else:
                            use = True

                if use:
                    method_keys.append(k)
                        
    

    minMethodNameSize = max([len(m) for m in method_keys])

    for layerdepth, layersize in zip(layerdepths, layersizes):

        key_trueVal = str(layerdepth) + '|' + str(layersize) + ' - trueValue'

        TrueValue = df[key_trueVal].mean()

        #print('-- layer {}|{} --'.format(layerdepth, layersize))
        #print('True Value:', round(TrueValue, 2))
        for method in method_keys:
            #for layermethod in [' - ', ' -layerMean- ']:
            #for layermethod in [' -layerMean- ']:
            for layermethod in layermethods:
                try:
                    preprefix  = prefix
                    preprefix += method
                    preprefix += ': Depth[{:3}]m: Size[{:2}]m'.format(layerdepth,layersize) 
                    preprefix += layermethod +':'
                    for _ in range(minMethodNameSize - len(method)):
                        preprefix += ' '
                    for _ in range(13 - len(layermethod)):
                        preprefix += ' '
                

                    key_Method = str(layerdepth) + '|' + str(layersize) + layermethod + str(method)

                    measurements = (df[key_Method] / df[key_trueVal]).to_numpy()

                    TotalValue = measurements.mean()
                    TotalValue_std  = np.std(measurements)
                    TotalValueError = 2 * TotalValue_std / np.sqrt(len(measurements))
                    TotalValue_2SD  = 2 * TotalValue_std

                    tmpd = deepcopy(measurements)
                    tmpd = tmpd[tmpd > TotalValue-TotalValue_2SD]
                    tmpd = tmpd[tmpd < TotalValue+TotalValue_2SD]
                    Interval_2SD = len(tmpd)/len(measurements)  
                    TrueValue = 1
                    #print(TotalValue,TotalValue_2SD, min(tmpd),max(tmpd))

    #                 TotalValue_median = df[method].median()        
    #                 TotalValue_min    = np.min(df[method])
    #                 TotalValue_max    = np.max(df[method])


                    string = (preprefix
                              + ': {} |' 
                              + '| {:+5.1f} % | ±{:5.1f} % |'
                              + '| ±{:5.1f} % | {:4.0f} % |'
                              + '| {:+6.1f} % | {:+5.1f} % | {:+5.1f} % | {:+5.1f} % | {:+6.1f} %'
                             ).format(
                              len(df[method]),
                              100 * (TotalValue - 1),
                              100 * (TotalValueError),
                              100 * (TotalValue_2SD),
                              100 * Interval_2SD,
                              100 * (np.percentile(measurements, 0)-1),
                              100 * (np.percentile(measurements, 25)-1),
                              100 * (np.percentile(measurements, 50)-1),
                              100 * (np.percentile(measurements, 75)-1), 
                              100 * (np.percentile(measurements, 100)-1),
                        )
                    print(string)

#                 string = preprefix + '[{}]: {:10.3f} / {:6.2f} % | std: {:5.4f} / {:6.2f} % | min: {:5.4f} / {:6.2f} % / {:5.2f} dB | max: {:5.4f} / {:6.2f} % / {:5.2f} dB'.format(
#                     len(df[key_Method]),
#                     TotalValue,
#                     100 * (TotalValue - 1),
#                     TotalValue_std,
#                     100 * (TotalValue_std),
#                     TotalValue_min,
#                     100 * (TotalValue_min -1),
#                     10 * np.log10(TotalValue_min),
#                     TotalValue_max,
#                     100 * (TotalValue_max -1),
#                     10 * np.log10(TotalValue_max),
#                 )
#                 print(string)
                except:
                    pass

        #print()

def plot_simreturns_stream(create_figure, setup, df, name, sort_by=None, methods=None, layerDepths=None, layerSizes=None,
                           methods_not=None, kwargs_list=None):
    if methods is None:
        methods = df['methods'].to_numpy()[1].copy()

    if kwargs_list is None:
        kwargs_list = []

    if methods_not is not None:
        for m in methods_not:
            for i, meth in enumerate(methods):
                if meth == m:
                    del methods[i]
                    break

    #print('layerDepths',layerDepths)
    if layerDepths is None:
        layerDepths = np.array(setup['layerDepths'])
    if layerSizes is None:
        layerSizes = np.array(setup['layerSizes'])

    #print('layerDepths',layerDepths)

    nr = 0
    for method in methods:
        fig = create_figure('layers[{}]-method[{}]- '.format(len(layerDepths), method) + name)
        ax = fig.subplots()
        ax.set_title('layers[{}]-method[{}]- '.format(len(layerDepths), method) + name)
        for layerDepth,layerSize in zip(layerDepths,layerSizes):

            
            if len(kwargs_list) > nr:
                kwargs = kwargs_list[nr]
            nr +=1

            if sort_by is None:
                layer_str = '{}|{}'.format(layerDepth,layerSize)

                ax.plot(df.index, df[str(layer_str) + ' - ' + str(method)] / df[str(layer_str) + ' - trueValue'],
                        label='layer[{}]'.format(str(layer_str)), **kwargs)

            elif sort_by == 'layerDepths':
                layer_str = '{}|{}'.format(layerDepth,layerSize)
                ax.plot(df[str(layer_str) + ' - ' + str(method)] / df[str(layer_str) + ' - trueValue'],
                        [layerDepth for _ in df.index],

                        label='layer[{}]'.format(str(layer_str)), **kwargs)

            else:
                layer_str = '{}|{}'.format(layerDepth,layerSize)
                df_copy = df.sort_values(sort_by)
                ax.plot(df_copy[sort_by],
                        df_copy[str(layer_str) + ' - ' + str(method)] / df_copy[str(layer_str) + ' - trueValue'],
                        label='layer[{}]'.format(str(layer_str)), **kwargs)

            ax.legend()

        return ax


def do_plot_simreturns_stream(create_figure,setup,simreturns,filtered_simreturns,layerDepths=None, layerSizes=None,methods_not=None,max_db_diffs=None):
    #layers = filtered_simreturns['layers'].to_numpy()
    ax = plot_simreturns_stream(create_figure,setup,simreturns, 'simreturns - default', layerDepths=layerDepths,layerSizes=layerSizes, methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,simreturns, 'simreturns - target_range', 'target_range', layerDepths=layerDepths,layerSizes=layerSizes,
                                methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,simreturns, 'simreturns - target_x', 'target_x', layerDepths=layerDepths,layerSizes=layerSizes, methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,simreturns, 'simreturns - target_y', 'target_y', layerDepths=layerDepths,layerSizes=layerSizes, methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,simreturns, 'simreturns - target_z', 'target_z', layerDepths=layerDepths,layerSizes=layerSizes, methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,simreturns, 'simreturns - target_rx_angle', 'target_rx_angle', layerDepths=layerDepths,layerSizes=layerSizes,
                                methods_not=methods_not)

    ax = plot_simreturns_stream(create_figure,setup,filtered_simreturns, 'filtered_simreturns - default', layerDepths=layerDepths,layerSizes=layerSizes,
                                methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,filtered_simreturns, 'filtered_simreturns - target_range', 'target_range',
                                layerDepths=layerDepths,layerSizes=layerSizes, methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,filtered_simreturns, 'filtered_simreturns - target_x', 'target_x', layerDepths=layerDepths,layerSizes=layerSizes,
                                methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,filtered_simreturns, 'filtered_simreturns - target_y', 'target_y', layerDepths=layerDepths,layerSizes=layerSizes,
                                methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,filtered_simreturns, 'filtered_simreturns - target_z', 'target_z', layerDepths=layerDepths,layerSizes=layerSizes,
                                methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,filtered_simreturns, 'filtered_simreturns - layerDepths', 'layerDepths', layerDepths=layerDepths,layerSizes=layerSizes,
                                methods_not=methods_not)
    ax = plot_simreturns_stream(create_figure,setup,filtered_simreturns, 'filtered_simreturns - target_rx_angle', 'target_rx_angle',
                                layerDepths=layerDepths,layerSizes=layerSizes, methods_not=methods_not)

    methods = filtered_simreturns['methods'].to_numpy()[1]

    for method in methods:
        plot_simreturns_stream(create_figure,setup,filtered_simreturns, 'simreturns - default - ' + method, methods=[method], layerDepths=layerDepths,layerSizes=layerSizes)


