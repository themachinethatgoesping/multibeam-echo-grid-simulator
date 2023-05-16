# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import math

from matplotlib import pyplot as plt

import numpy as np
from tqdm.auto import tqdm

from numba import njit

from copy import deepcopy
from collections import defaultdict
from enum import IntEnum

import mbes_sim.functions.helperfunctions as hlp
import mbes_sim.functions.create_bubblesfunctions as bubbles
import mbes_sim.functions.navfunctions as nav
import mbes_sim.functions.transformfunctions as tf
import mbes_sim.functions.pulsefunctions as pf
import mbes_sim.functions.gridfunctions as gf

from collections.abc import MutableMapping

import warnings

"""Functions and classes for creating scatter grids from simulated data.

Scattergrids use the grid classes defined in gridfunctions.py and add convinience functions for averaging, integrating and plotting.
"""

@njit
def static_get_target_pos(image: np.ndarray, min_val: float = np.nan) -> (float, float, float):
    """Returns the center of weight from an image to estimate the position of the target in the image. 
    Note: this only works if only one target is present in the image.

    Parameters
    ----------
    image : np.ndarray
        3D numpy array
    min_val : float, optional
        Threshold applied before applying the function, by default np.nan

    Returns
    -------
    (float, float, float)
        x,y,z position of the target [m]
    """

    x_sum = 0
    y_sum = 0
    z_sum = 0
    x_weight = 0
    y_weight = 0
    z_weight = 0

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                if not np.isfinite(image[x][y][z]):
                    continue

                if image[x][y][z] < min_val:
                    continue

                x_sum += image[x][y][z] * x
                y_sum += image[x][y][z] * y
                z_sum += image[x][y][z] * z
                x_weight += image[x][y][z]
                y_weight += image[x][y][z]
                z_weight += image[x][y][z]

    return x_sum/x_weight, y_sum/y_weight, z_sum/z_weight


class ScatterGrid(object):
    """Class for creating and handling scatter grids. This class uses the grid classes defined in gridfunctions.py and adds convinience functions for averaging, integrating and plotting.
    """

    def __init__(self, imagesums: np.ndarray, imagenums: np.ndarray, gridder: gf.GRIDDER):
        """Creates a ScatterGrid from the given imagesums, imagenums and gridder.

        Parameters
        ----------
        imagesums : np.ndarray
            3D numpy array containing the summed backscattering values (can be created using the gridfunctions.Gridder class)
        imagenums : np.ndarray
            3D numpy array containing the number or weight of the backscattering values (can be created using the gridfunctions.Gridder class)
        gridder : gf.Gridder
            Gridder object used to create the imagesums and imagenums arrays
        """

        self.ImageSums = imagesums.copy()
        self.ImageNums = imagenums.copy()
        self.ImageAvg = np.empty(imagenums.shape,dtype=float)
        self.ImageAvg.fill(np.nan)

        self.ImageAvg[imagenums > 0] = imagesums[imagenums > 0] / imagenums[imagenums > 0]

        self.TotalValue = np.nansum(self.ImageAvg) * (gridder.xres*gridder.yres*gridder.zres)

        self.ZDiff = gridder.get_extent_z()[1] - gridder.get_extent_z()[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.TotalValueLayer = np.nansum(np.nanmean(self.ImageAvg, axis=2))*gridder.xres * gridder.yres * self.ZDiff

        #self.Gridder = gridder
        self.ExtentX = gridder.get_extent_x()
        self.ExtentY = gridder.get_extent_y()
        self.ExtentZ = gridder.get_extent_z()
        self.ResX = gridder.xres
        self.ResY = gridder.yres
        self.ResZ = gridder.zres
        self.MinX = gridder.xmin
        self.MaxX = gridder.xmax
        self.MinY = gridder.ymin
        self.MaxY = gridder.ymax
        self.MinZ = gridder.zmin
        self.MaxZ = gridder.zmax

    def get_target_pos(self,min_val: float = np.nan) -> (float, float, float):
        """Returns the center of weight from the image to estimate the position of the target in the image.
        Note: this only works if only one target is present in the image.

        Parameters
        ----------
        min_val : float, optional
            Threshold applied before applying the function, by default np.nan

        Returns
        -------
        (float, float, float)
            x,y,z position of the target [m]
        """
        xi,yi,zi = static_get_target_pos(self.ImageAvg,min_val)

        return xi * self.ResX + self.MinX,\
               yi * self.ResY + self.MinY,\
               zi * self.ResZ + self.MinZ

    def getTotalvalue(self,min_val: float) -> float:
        """Returns the total backscattering value of the internal 3D image.

        Parameters
        ----------
        min_val : float
            Apply a threshold before calculating the total value

        Returns
        -------
        float
            Total backscattering value
        """
        if not np.isfinite(min_val):
            return self.TotalValue

        return np.nansum(self.ImageAvg[self.ImageAvg >= min_val]) * (self.ResX * self.ResY * self.ResZ)

    def getTotalvalueLayer(self,min_val: float) -> float:
        """Returns the total backscattering value of the internal 3D image, averaged over the z-axis.

        Parameters
        ----------
        min_val : float
            Apply a threshold before calculating the total value

        Returns
        -------
        float
            Total backscattering value
        """

        if not np.isfinite(min_val):
            return self.TotalValueLayer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            layerAvg = np.nanmean(self.ImageAvg, axis=2)

        return np.nansum(layerAvg[layerAvg >= min_val]) * self.ResX * self.ResY * self.ZDiff

    def getGridder(self) -> gf.GRIDDER:
        """Returns a gridder object initialized with the values specified in this ScatterGrid.

        Returns
        -------
        gf.GRIDDER
            Equivalent gridder object
        """
        return gf.GRIDDER(self.ResX, self.ResY, self.ResZ,
                          self.MinX, self.MaxX,
                          self.MinY, self.MaxY,
                          self.MinZ, self.MaxZ)


    def cutDepthLayer(self,layer_z: float,layer_size: float) -> ScatterGrid:
        """Returns a new ScatterGrid object with the same values as this ScatterGrid, but only for the specified layer.

        Parameters
        ----------
        layer_z : float
            Center of the layer [m]
        layer_size : float
            Size of the layer [m]

        Returns
        -------
        ScatterGrid
            New ScatterGrid object
        """

        gridder_old = self.getGridder()

        minZ = gridder_old.get_z_grd_value(layer_z - (layer_size - gridder_old.zres) / 2)
        maxZ = gridder_old.get_z_grd_value(layer_z + (layer_size - gridder_old.zres) / 2)

        gridder = gf.GRIDDER(self.ResX,self.ResY,self.ResZ,
                             self.MinX,self.MaxX,
                             self.MinY,self.MaxY,
                             minZ,maxZ)

        iz0 = gridder_old.get_z_index(minZ)
        iz1 = gridder_old.get_z_index(maxZ)



        imagesums = self.ImageSums[:,:,iz0:iz1+1]
        imagenums = self.ImageNums[:,:,iz0:iz1+1]

        return ScatterGrid(imagesums,imagenums,gridder)

    def getDepthMeanImage(self, layer_z: float,layer_size: float) -> (np.ndarray,np.ndarray,gf.GRIDDER):
        """Returns the mean image (vertically averaged) of the specified layer.

        Parameters
        ----------
        layer_z : float
            Center of the layer [m]
        layer_size : float
            Size of the layer [m]

        Returns
        -------
        TODO: this is different than what is stated above
        (np.ndarray,np.ndarray,gf.GRIDDER)
            Mean image, number of samples, gridder
        """

        gridder_old = self.getGridder()

        minZ = gridder_old.get_z_grd_value(layer_z - (layer_size - gridder_old.zres) / 2)
        maxZ = gridder_old.get_z_grd_value(layer_z + (layer_size - gridder_old.zres) / 2)

        gridder = gf.GRIDDER(self.ResX,self.ResY,self.ResZ,
                             self.MinX,self.MaxX,
                             self.MinY,self.MaxY,
                             minZ,maxZ)

        iz0 = gridder_old.get_z_index(minZ)
        iz1 = gridder_old.get_z_index(maxZ)

        imagesums = self.ImageSums[:,:,iz0:iz1+1]
        imagenums = self.ImageNums[:,:,iz0:iz1+1]

        return imagesums,imagenums,gridder

    def getGridExtents(self) -> (np.ndarray,np.ndarray,np.ndarray):
        """Returns the extents of the 3D grid.

        Returns
        -------
        (np.ndarray,np.ndarray,np.ndarray)
            Extents of the 3D grid (x,y,z)
        """
        return self.ExtentX,self.ExtentY,self.ExtentZ

    def toString(self,TrueValue: float, methodName: str = None, minMethodNameSize:int = None) -> str:
        """Returns a string representation of the ScatterGrid object and the integration results. (For plotting or printing)

        Parameters
        ----------
        TrueValue : float
            The true value of the object (simulation input)
        methodName : str, optional
            Name of the used method, by default None
        minMethodNameSize : int, optional
            Fixed minimum number of characters of the method name (for alligning), by default None

        Returns
        -------
        str
            String representation of the ScatterGrid object and the integration results
        """

        if methodName is None:
            prefix = "TotalValue"
        else:
            prefix = 'Bubbles Grid'
            if minMethodNameSize:
                prefix += '[{:MMMs}]'.replace('MMM',str(int(minMethodNameSize))).format(methodName)
            else:
                prefix += '[{}]'.format(methodName)

        string = prefix + ': {:15.2f}  | {:5.2f} %'.format(round(self.TotalValue, 2),round(100 * (self.TotalValue / TrueValue - 1),2))
        return string

    def print(self,TrueValue: float, methodName: str = None, minMethodNameSize:int = None) -> str:
        """Print a string representation of the ScatterGrid object and the integration results. (For plotting or printing)

        Parameters
        ----------
        TrueValue : float
            The true value of the object (simulation input)
        methodName : str, optional
            Name of the used method, by default None
        minMethodNameSize : int, optional
            Fixed minimum number of characters of the method name (for alligning), by default None

        """
        print(self.toString(TrueValue, methodName,minMethodNameSize))

    def get_3DImage(self,
             todB: bool= True,
             mindBVal:float = -50) -> np.ndarray:
        """Returns an averaged 3D image of the ScatterGrid. 
        The internal images are averaged over the number of samples. 
        Depending on the parameters, the image can be converted to dB and/or the minimum dB value can be replaced by a constant.

        Parameters
        ----------
        todB : bool, optional
            convert image to dB, by default True
        mindBVal : float, optional
            minimum dB value for the conversion (to avoid log10(0)), by default -50

        Returns
        -------
        np.ndarray
            3D image
        """

        image = self.ImageAvg.copy()
        image[self.ImageNums == 0] = np.nan
        if todB:
            image[image == 0] = 0.000000000000001
            image = 10 * np.log10(image)
            image[image < mindBVal] = mindBVal

        return image

    def plot(self, figure:plt.Figure,
             targets_color: (list(bubbles.Targets), list) = None,
             target_size: float = 1,
             show_wci: bool = True,
             show_echo: bool = True,
             show_map: bool = True,
             show_colorbar: bool = True,

             todB: bool= True,
             mindBVal: float = -50,
             mindBReplace: float = None,
             xindex: int = None,
             yindex: int = None,
             zindex: int = None,
             zindeces: list = None,
             kwargs: dict=None,
             colorbar_kwargs: dict=None) -> (plt.Figure,plt.Axes, np.ndarray):
        """Plots for the ScatterGrid object.

        Parameters
        ----------
        figure : plt.Figure
            Figure used for the plotting
        targets_color : (list(bubbles.Targets), list), optional
            Add a list of Targets and corresponding colors to the plot, by default None
        target_size : float, optional
            size used for plotting targets, by default 1
        show_wci : bool, optional
            If true: plot the front view (wci view), by default True
        show_echo : bool, optional
            If true: plot the side view (echogram view), by default True
        show_map : bool, optional
            If true: plot the top view (map view), by default True
        show_colorbar : bool, optional
            If true: plot the colorbar, by default True
        todB : bool, optional
            If true: convert image to dB, by default True
        mindBVal : float, optional
            Threshold for the to dB conversion (to avoid log(0)), by default -50
        mindBReplace : float, optional
            Value used to replace values < mindBVal, by default None
        xindex : int, optional
            If set, use this index for wci view, by default None
        yindex : int, optional
            If set, use this index for echo view, by default None
        zindex : int, optional
            If set, use this index for map view, by default None
        zindeces : list, optional
            If set, use average of these indices for map view, by default None
        kwargs : dict, optional
            kwargs for the matplotlib plots, by default None
        colorbar_kwargs : dict, optional
            kwargs for the matplotlib colorbar command, by default None

        Returns
        -------
        (plt.Figure,plt.Axes, np.ndarray)
            Figure, Axes and 3D image that is plotted
        """

        figure.clear()

        nplots = sum([show_wci, show_echo, show_map])
        if kwargs is None:
            kwargs = {}
        if colorbar_kwargs is None:
            colorbar_kwargs = {}

        if nplots == 1:
            axes = [figure.subplots(ncols=nplots)]
        else:
            axes = figure.subplots(ncols=nplots)

        axit = iter(axes)
        image_extent_x, image_extent_y, image_extent_z = self.getGridExtents()

        def getNanSum(imageLin, axis,divide=1):
            #image = np.nansum(imageLin,axis=axis)
            image = np.nanmean(imageLin,axis=axis)
            num = np.nansum(self.ImageNums,axis=axis)

            if divide != 1:
                image /= divide

            image [num == 0] = np.nan

            if todB:
                image[image == 0] = 0.000000000000001
                image = 10 * np.log10(image)
                if mindBReplace is not None:
                    image[image < mindBVal] = mindBReplace
                else:
                    image[image < mindBVal] = mindBVal


            return image

        if show_wci:
            ax = next(axit)

            if xindex is None:
                image = getNanSum(self.ImageAvg.copy(),axis=0)
            else:
                image = self.ImageAvg[xindex,:,:]
                if todB:
                    image[image == 0] = 0.000000000000001
                    image = 10 * np.log10(image)
                    if mindBReplace is not None:
                        image[image < mindBVal] = mindBReplace
                    else:
                        image[image < mindBVal] = mindBVal

            mapable = ax.imshow(image.transpose(), aspect='equal',
                      extent=[image_extent_y[0], image_extent_y[1], image_extent_z[1], image_extent_z[0]],
                      **kwargs)

            if show_colorbar:
                figure.colorbar(mapable, ax=ax, **colorbar_kwargs)

            if targets_color:
                for targets,color in targets_color:
                    ax.scatter(targets.y, targets.z, c = color, s = target_size)

        if show_echo:
            ax = next(axit)

            if yindex is None:
                image = getNanSum(self.ImageAvg.copy(),axis=1)
            else:
                image = self.ImageAvg[:,yindex,:]
                if todB:
                    image[image == 0] = 0.000000000000001
                    image = 10 * np.log10(image)
                    if mindBReplace is not None:
                        image[image < mindBVal] = mindBReplace
                    else:
                        image[image < mindBVal] = mindBVal


            mapable = ax.imshow(image.transpose(), aspect='equal',
                      extent=[image_extent_x[0], image_extent_x[1], image_extent_z[1], image_extent_z[0]],
                      **kwargs)
            if targets_color:
                for targets,color in targets_color:
                    ax.scatter(targets.x, targets.z, c = color, s = target_size)


            if show_colorbar:
                figure.colorbar(mapable, ax=ax, **colorbar_kwargs)

        if show_map:
            ax = next(axit)

            if zindex is None and zindeces is None:
                image = getNanSum(self.ImageAvg.copy(),axis=2)
            elif zindex is None and zindeces is not None:
                image = getNanSum(self.ImageAvg[:,:,zindeces[0]:zindeces[1]+1],axis=2)#,divide=abs(zindeces[1]-zindeces[0]))

            else:
                image = self.ImageAvg[:,:,zindex]
                if todB:
                    image[image == 0] = 0.000000000000001
                    image = 10 * np.log10(image)
                    if mindBReplace is not None:
                        image[image < mindBVal] = mindBReplace
                    else:
                        image[image < mindBVal] = mindBVal

            mapable = ax.imshow(image, aspect='equal',
                      extent=[image_extent_y[0], image_extent_y[1], image_extent_x[1], image_extent_x[0]],
                      **kwargs)
            if targets_color:
                for targets,color in targets_color:
                    ax.scatter(targets.y, targets.x, c = color, s = target_size)


            if show_colorbar:
                figure.colorbar(mapable, ax=ax, **colorbar_kwargs)

        return figure,ax,image


class ScatterGridDict(MutableMapping):
    """A dictionary that stores ScatterGrid objects. Adds convinience functions for printing and cutting layers.
    """

    def print(self, TrueValue: float):
        """Prints the integration results of the ScatterGrid objects in the dictionary

        Parameters
        ----------
        TrueValue : float
            True backscattering value of the targets used as simulation input

        """
        maxKeyLen = max([len(k) for k in self.keys()])
        for k in self.keys():
            self.store[k].print(TrueValue, k, maxKeyLen)


    def cutDepthLayer(self, layer_z: float, layer_size: float) -> ScatterGridDict:
        """Cuts a layer of the ScatterGrid objects in the dictionary

        Parameters
        ----------
        layer_z : float
            Center of the layer [m]
        layer_size : float
            Size of the layer [m]

        Returns
        -------
        ScatterGridDict
            Dictionary containing the cut ScatterGrid objects
        """

        scd = ScatterGridDict()
        try_layer_z = None
        for k in self.keys():
            scd[k] = self[k].cutDepthLayer(layer_z,layer_size)

        z_extend = scd[k].ExtentZ
        true_layer_size_z = abs(z_extend[1]-z_extend[0])
        z_coordinates = scd[k].getGridder().get_z_coordinates()

        return scd,(z_extend,true_layer_size_z,z_coordinates)
    

    # ----- Functions to implement the MutableMapping abstract base class -----
    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        if isinstance(value, ScatterGrid):
            self.store[key] = value
        else:
            try:
                self.store[key] = ScatterGrid(*value)
            except:
                try:
                    types = [type(v) for v in value]
                except:
                    types = [type(value)]

                raise RuntimeError("Cannot initialize ScatterGrid using arguments of type:",*types)

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)



if __name__ == "__main__":
    # check the printing function

    imN = np.ones((10,10,10))
    imS = np.ones((10,10,10))
    gridder = gf.GRIDDER(1,1,1,0,1,0,1,0,1)
    sc = ScatterGrid(imN,imS,gridder)

    scd = ScatterGridDict()
    scd['test'] = sc
    scd['test2'] = imN,imS,gridder

    print(scd['test'].toString(999,'peter',10))
    print(scd['test2'].toString(1010,'peter',10))
    print(scd['test2'].toString(1020,'peter'))
    print(scd['test2'].toString(1030,))

    print('---')
    scd.print(999)

    print('done')
