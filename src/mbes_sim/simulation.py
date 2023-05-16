# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

import math

from matplotlib import pyplot as plt

import numpy as np
from tqdm.auto import tqdm

from numba import njit

from copy import deepcopy
from enum import IntEnum

import mbes_sim.functions.helperfunctions as hlp
import mbes_sim.functions.create_bubblesfunctions as bubbles
import mbes_sim.functions.navfunctions as nav
import mbes_sim.functions.transformfunctions as tf
import mbes_sim.functions.pulsefunctions as pf
import mbes_sim.functions.gridfunctions as gf
import mbes_sim.mbes as mb
import mbes_sim.scattergrid as sg

"""Class and functions to create individual simulation runs (scatter grids)
Use simulationfunctions.py for batch simulations
"""

@njit
def get_extent_vol(sampleranges, equivalent_beam_angle_tx_radians, equivalent_beam_angles_rx_radians, pulse_distance,
                   sample_size, beam_spacing_degrees):
    # sample volume
    v1 = equivalent_beam_angle_tx_radians * equivalent_beam_angles_rx_radians * pulse_distance
    v1 = sampleranges * sampleranges * v1

    extent = np.power(v1, 1 / 3)
    return extent


@njit
def get_extent_max_real(sampleranges, equivalent_beam_angle_tx_radians, equivalent_beam_angles_rx_radians, pulse_distance,
                        sample_size, beam_spacing_degrees):
    extent = np.empty(sampleranges.shape)
    for i, r in enumerate(sampleranges):
        extent[i] = np.array([r * equivalent_beam_angle_tx_radians, r * equivalent_beam_angles_rx_radians, pulse_distance]).max()
    return extent

@njit
def get_extent_max_dist(sampleranges, beam_spacing_radians):
    extent = np.empty(sampleranges.shape)
    for i, r in enumerate(sampleranges):
        extent[i] = np.array([r * beam_spacing_radians, sampleranges[1]-sampleranges[0]]).max()
    return extent



class GriddingMethod(IntEnum):
    """Enum class for gridding methods.     
    """

    ts = 0, # block mean of ts, for testing only
    sv = 1, # block mean of sv
    sv_int_lin = 2, # weighted mean of sv
    
    # block mean of different variables. For test/plotting only
    pingpos_x = 8,
    pingpos_y = 9,
    pingpos_z = 10,
    pingpos_yaw = 11,
    pingpos_pitch = 12,
    pingpos_roll = 13,
    angle_swath = 14,
    angle_beam = 15

    def to_string(self) -> str:
        """Converts the enum value to a string.

        Returns
        -------
        str
            The string representation of the enum value.
        """
        if self.value == GriddingMethod.ts: return 'ts' # block mean of ts, for testing only
        if self.value == GriddingMethod.sv: return 'sv' # block mean of sv
        if self.value == GriddingMethod.sv_int_lin: return 'sv_int_lin' # weighted mean of sv

        # block mean of different variables. For test/plotting only
        if self.value == GriddingMethod.pingpos_x: return 'pingpos_x'
        if self.value == GriddingMethod.pingpos_y: return 'pingpos_y'
        if self.value == GriddingMethod.pingpos_z: return 'pingpos_z'
        if self.value == GriddingMethod.pingpos_yaw: return 'pingpos_yaw'
        if self.value == GriddingMethod.pingpos_pitch: return 'pingpos_pitch'
        if self.value == GriddingMethod.pingpos_roll: return 'pingpos_roll'
        if self.value == GriddingMethod.angle_swath: return 'angle_swath'
        if self.value == GriddingMethod.angle_beam: return 'angle_beam'

    @staticmethod
    def print_all():
        """Prints all possible enum values as strings.
        """
        for i in range(99):
            try:
                print(GriddingMethod(i).to_string())
            except:
                pass


def grid_wci(
        methods: list(GriddingMethod),
        imagessums: dict,
        imagesnums: dict,
        gridder: gf.GRIDDER,
        multibeam: mb.Multibeam,
        ECHO: np.ndarray, TDSV: np.ndarray,
        TDX: np.ndarray, TDY: np.ndarray, TDZ: np.ndarray,
) -> (dict, dict):
    """Grids the data using the given methods.

    Parameters
    ----------
    methods : list(GriddingMethod)
        List of gridding methods (defined above)
    imagessums : dict
        dictionary of image sums (integrated backscattering values) (can be empty)
    imagesnums : dict
        dictionary of image nums (integration weights) (can be empty)
    gridder : gf.GRIDDER
        gridder object corresponding to imagesums and imagenums
    multibeam : mb.Multibeam
        Multibeam object containing the simulation parametesr (e.g. beam angles, pulse length, current position, etc.)
    ECHO : np.ndarray
        2D array of echo values (WCI / TS), used for TS gridding
    TDSV : np.ndarray
        2D array of echo values (WCI / SV), used for SV gridding
    TDX : np.ndarray
        2D array of x coordinates (result from WCI raytracing)
    TDY : np.ndarray
        2D array of y coordinates (result from WCI raytracing)
    TDZ : np.ndarray
        2D array of z coordinates (result from WCI raytracing)
           
    Returns
    -------
    (dict, dict)
        Dictionaries with np.ndarrays as values. The keys are the method names.
        - first dict is a dict of image sums (integrated backscattering values)
        - second dict is a dict of image nums (integration weights)

    Raises
    ------
    RuntimeError
        Aaaaaaah! if the given method is not implemented.
    """

    bval = ECHO.flatten()
    bsv = TDSV.flatten()
    x = TDX.flatten()
    y = TDY.flatten()
    z = TDZ.flatten()

    for method in methods:
        try:
            imagesum = imagessums[method.to_string()]
            imagenum = imagesnums[method.to_string()]
        except:
            imagesum = None
            imagenum = None

        tmp = np.empty(bval.shape)
        if method == GriddingMethod.pingpos_x:
            tmp.fill(pingpositions_x[pnr])
            imagesum, imagenum = gridder.append_sampled_image(x, y, z, tmp, imagesum=imagesum, imagenum=imagenum,
                                                              skip_invalid=True)
        elif method == GriddingMethod.pingpos_y:
            tmp.fill(pingpositions_y[pnr])
            imagesum, imagenum = gridder.append_sampled_image(x, y, z, tmp, imagesum=imagesum, imagenum=imagenum,
                                                              skip_invalid=True)
        elif method == GriddingMethod.pingpos_z:
            tmp.fill(heaves[pnr])
            imagesum, imagenum = gridder.append_sampled_image(x, y, z, tmp, imagesum=imagesum, imagenum=imagenum,
                                                              skip_invalid=True)
        elif method == GriddingMethod.pingpos_yaw:
            tmp.fill(heaves[pnr])
            imagesum, imagenum = gridder.append_sampled_image(x, y, z, tmp, imagesum=imagesum, imagenum=imagenum,
                                                              skip_invalid=True)
        elif method == GriddingMethod.pingpos_pitch:
            tmp.fill(heaves[pnr])
            imagesum, imagenum = gridder.append_sampled_image(x, y, z, tmp, imagesum=imagesum, imagenum=imagenum,
                                                              skip_invalid=True)
        elif method == GriddingMethod.pingpos_roll:
            tmp.fill(heaves[pnr])
            imagesum, imagenum = gridder.append_sampled_image(x, y, z, tmp, imagesum=imagesum, imagenum=imagenum,
                                                              skip_invalid=True)
        elif method == GriddingMethod.angle_swath:
            tmp.fill(multibeam.transmit_steeringangle_degrees)
            imagesum, imagenum = gridder.append_sampled_image(x, y, z, tmp, imagesum=imagesum, imagenum=imagenum,
                                                              skip_invalid=True)
        elif method == GriddingMethod.angle_beam:
            tmp.fill(beam_angle)
            imagesum, imagenum = gridder.append_sampled_image(x, y, z, tmp, imagesum=imagesum, imagenum=imagenum,
                                                              skip_invalid=True)

        elif method == GriddingMethod.ts:
            imagesum, imagenum = gridder.append_sampled_image(x, y, z, bval, imagesum=imagesum, imagenum=imagenum,
                                                              skip_invalid=True)
        elif method == GriddingMethod.sv:
            imagesum, imagenum = gridder.append_sampled_image(x, y, z, bsv, imagesum=imagesum, imagenum=imagenum,
                                                              skip_invalid=True)

        elif method == GriddingMethod.sv_int_lin:
            imagesum, imagenum = gridder.append_sampled_image2(x, y, z, bsv,
                                                               imagesum=imagesum,
                                                               imagenum=imagenum,
                                                               skip_invalid=True)
        else:
            raise RuntimeError('Aaaaaaah!' + GriddingMethod.to_string())

        imagessums[method.to_string()] = imagesum
        imagesnums[method.to_string()] = imagenum

    return imagessums, imagesnums



class Simulation(object):
    """Simulation class to hold all simulation parameters
    Usefull for individual simulations
    For batch simulations, have a look at simulationfunctions.py
    """

    # unitilized variables
    # These have to be initialized before the simulation can be run
    Multibeam = None
    Targets   = None
    GridResX   = None
    GridResy   = None
    GridResZ   = None
    UseIdealizedBeampattern = False
    Survey = None

    LimitXMin = np.nan
    LimitXMax = np.nan
    LimitYMin = np.nan
    LimitYMax = np.nan
    LimitZMin = np.nan
    LimitZMax = np.nan

    def __init__(self):
        """Create an uninitialized simulation object.
        Note, you need to run:
        - setMultibeam
        - setTargets
        - setGridResolution
        - setSurvey
        - setMethods
        - setGridLimits(optional)
        before you can run the simulation
        """
        pass

    def setGridLimits(self,
                      xmin: float = np.nan,
                      xmax: float = np.nan,
                      ymin: float = np.nan,
                      ymax: float = np.nan,
                      zmin: float = np.nan,
                      zmax: float = np.nan
                      ):
        """Set the limits of the grid. If not set, the limits are determined by the survey object

        Parameters
        ----------
        xmin : float, optional
            Minimum x value covered by the grid [m], by default np.nan
        xmax : float, optional
            Maximum x value covered by the grid [m], by default np.nan
        ymin : float, optional
            Minimum y value covered by the grid [m], by default np.nan
        ymax : float, optional
            Maximum y value covered by the grid [m], by default np.nan
        zmin : float, optional
            Minimum z value covered by the grid [m], by default np.nan
        zmax : float, optional
            Maximum z value covered by the grid [m], by default np.nan
        """

        self.LimitXMin = xmin
        self.LimitXMax = xmax
        self.LimitYMin = ymin
        self.LimitYMax = ymax
        self.LimitZMin = zmin
        self.LimitZMax = zmax

    def setMultibeam(self,multibeam: mb.Multibeam):
        """Set the multibeam object used for the simulation
        Note: the set_navigation function of the multibeam object will be called/initialzied during the simulation

        Parameters
        ----------
        multibeam : mb.Multibeam
            Initialzied mb.Multibeam that holds all relevant information for the simulation (except set_navigation which will be set from the Survey object)
        """
        self.Multibeam = deepcopy(multibeam)
        self.Multibeam.set_navigation(0, 0, 0, 0, 0, 0)

        tX, tY, tZ = self.Multibeam.raytrace_wci()

        self.MinY = min(tY.flat)
        self.MaxY = max(tY.flat)
        self.MinZ = min(tZ.flat)
        self.MaxZ = max(tZ.flat)

    def setTargets(self,targets: bubbles.Targets):
        """Set the targets object used for the simulation

        Parameters
        ----------
        targets : bubbles.Targets
            Targets object that holds position and strength of the simulated targets
        """

        self.Targets = targets

    def setGridResolution(self,gridres: float):
        """Set the x,y,z grid resolution for the simulation

        Parameters
        ----------
        gridres : float
            Grid cell size [m]
        """

        self.GridResX = gridres
        self.GridResY = gridres
        self.GridResZ = gridres

    def setGridResolutionX(self, gridres: float):
        """Set the x grid resolution for the simulation individually

        Parameters
        ----------
        gridres : float
            Grid cell size [m]
        """
        self.GridResX = gridres
    def setGridResolutionY(self, gridres: float):
        """Set the y grid resolution for the simulation individually

        Parameters
        ----------
        gridres : float
            Grid cell size [m]
        """
        self.GridResY = gridres
    def setGridResolutionZ(self, gridres: float):
        """Set the z grid resolution for the simulation individually

        Parameters
        ----------
        gridres : float
            Grid cell size [m]
        """
        self.GridResZ = gridres

    def getGridResolutions(self) -> (float,float,float):
        """get the x,y,z grid resolution for the simulation

        Returns
        -------
        (float,float,float)
            Grid cell size for x,y,z dimension [m]
        """
        return self.GridResX,self.GridResY,self.GridResZ

    def getGridVoxelVolume(self) -> float:
        """get the grid voxel volume

        Returns
        -------
        float
            Grid voxel volume [m^3]
        """
        return self.GridResX*self.GridResY*self.GridResZ

    def setUseIdealizedBeampattern(self,useIdealizedBeampattern : bool):
        """Set true to use idealized beampattern (rectangular)

        Parameters
        ----------
        useIdealizedBeampattern : bool
            True to use idealized beampattern (rectangular)
        """
        self.UseIdealizedBeampattern = useIdealizedBeampattern

    def setSurvey(self,survey: nav.Survey):
        """Set the survey object used for the simulation (determines the virtual path of the vessel)

        Parameters
        ----------
        survey : nav.Survey
            Survey object
        """
        self.Survey = survey

        self.MinX = min(self.Survey.pingpositions_x)
        self.MaxX = max(self.Survey.pingpositions_x)

    def setMethods(self,methods : list(GriddingMethod)):
        """Set the list of gridding methods to be used for the simulation

        Parameters
        ----------
        methods : list
            List of gridding methods to be used for the simulation
        """
        self.Methods = methods

    def getGridder(self) -> gf.GRIDDER:
        """Return a gridder object that describes the grid generated by this simulation

        Returns
        -------
        gf.GRIDDER
            Equivalent gridder object

        Raises
        ------
        RuntimeError
            If the simulation is not initialized
        """

        if self.GridResX is None: raise RuntimeError("ERROR Simulation: GridResX not initialized")
        if self.GridResY is None: raise RuntimeError("ERROR Simulation: GridResY not initialized")
        if self.GridResZ is None: raise RuntimeError("ERROR Simulation: GridResZ not initialized")
        if self.Multibeam is None: raise RuntimeError("ERROR Simulation: Multibeam not initialized")
        if self.Survey is None: raise RuntimeError("ERROR Simulation: Survey not initialized")

        xmin = self.LimitXMin if np.isfinite(self.LimitXMin) else self.MinX
        xmax = self.LimitXMax if np.isfinite(self.LimitXMax) else self.MaxX
        ymin = self.LimitYMin if np.isfinite(self.LimitYMin) else self.MinY
        ymax = self.LimitYMax if np.isfinite(self.LimitYMax) else self.MaxY
        zmin = self.LimitZMin if np.isfinite(self.LimitZMin) else self.MinZ
        zmax = self.LimitZMax if np.isfinite(self.LimitZMax) else self.MaxZ

        return gf.GRIDDER(self.GridResX, self.GridResY, self.GridResZ,
                             xmin, xmax,
                             ymin, ymax,
                             zmin, zmax)


    def simulate3DEchoesSamples(self,
                                progress: bool=True,
                                pbar=None) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray):
        """Simulate and raytrace the backscatter values for all beams and samples along the survey path over the set targets.

        Parameters
        ----------
        progress : bool, optional
            show progress bar, by default True
        pbar : tqdm.tqdm, optional
            If set: use this progressbar object to plot the progress, by default None

        Returns
        -------
        (np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray)
            Vectors of simulated backscatter values for all beams and samples allong the survey path
            - Simulated TS values
            - Simulated SV values
            - raytraced x positions of the samples
            - raytraced y positions of the samples
            - raytraced z positions of the samples

        Raises
        ------
        RuntimeError
            If the simulation is not initialized
        
        """

        if self.Survey is None: raise RuntimeError("ERROR Simulation: Survey not initialized")
        if self.Multibeam is None: raise RuntimeError("ERROR Simulation: Multibeam not initialized")
        if self.Targets is None: raise RuntimeError("ERROR Simulation: Targets not initialized")
        if self.Targets is None: raise RuntimeError("ERROR Simulation: Targets not initialized")

        if progress:
            iterator = tqdm(range(len(self.Survey)))
        else:
            iterator = range(len(self.Survey))

        if pbar is not None:
            pbar.reset(total=len(self.Survey))

        TDTS = []
        TDSV = []
        TDX = []
        TDY = []
        TDZ = []

        for pnr in iterator:
            if pbar is not None:
                pbar.update()

            # 1: because first is time, then x,y,z,yaw,pitch,roll
            self.Multibeam.set_navigation(*self.Survey[pnr][1:])

            TS, SV = self.Multibeam.create_wci(*self.Targets.xyzval_vectors(),
                                               return_ts=True,
                                               return_sv=True,
                                               idealized_beampattern=self.UseIdealizedBeampattern)

            TDTS.append(TS)
            TDSV.append(SV)
            TDX.append(X)
            TDY.append(Y)
            TDZ.append(Z)

        TDTS = np.array(TDTS)
        TDSV = np.array(TDSV)
        TDX = np.array(TDX)
        TDY = np.array(TDY)
        TDZ = np.array(TDZ)

        self.Multibeam.set_navigation(0, 0, 0, 0, 0, 0)

        return TDTS, TDSV, TDX, TDY, TDZ


    def simulate3DEchoesGrid(self,
                       progress: bool=False,
                       pbar=None,
                       return_scatterGrids: bool = True,
                       progress_position: int = 1,
                       min_sv: float=None) -> sg.ScatterGridDict:
        """Simulate, raytrace and grid the backscatter values for all beams and samples along the survey path over the set targets.
        

        Parameters
        ----------
        progress : bool, optional
            show progress bar, by default True
        pbar : tqdm.tqdm, optional
            If set: use this progressbar object to plot the progress, by default None
        progress_position : int, optional
            Position of the progressbar (tqdm parameter), by default 1
        min_sv : float, optional
            Threshhold for sv integration, by default None

        Returns
        -------
        sg.ScatterGridDict
            A ScatterGridDict that holds a scattergrid for each of the methods set in the simulation object.

        Raises
        ------
        RuntimeError
            If the simulation is not initialized
        """

        if self.Methods is None: raise RuntimeError("ERROR Simulation: Methods not initialized")
        if self.Survey is None: raise RuntimeError("ERROR Simulation: Survey not initialized")
        if self.Multibeam is None: raise RuntimeError("ERROR Simulation: Multibeam not initialized")
        if self.Targets is None: raise RuntimeError("ERROR Simulation: Targets not initialized")

        gridder = self.getGridder()

        if progress:
            iterator = tqdm(range(len(self.Survey)),position=progress_position)
        else:
            iterator = range(len(self.Survey))

        if pbar is not None:
            pbar.reset(total=len(self.Survey))

        imagessums = {}
        imagesnums = {}

        for pnr in iterator:
            if pbar is not None:
                pbar.update()

            # 1: because first is time, then x,y,z,yaw,pitch,roll
            self.Multibeam.set_navigation(*self.Survey[pnr][1:])

            TS, SV = self.Multibeam.create_wci(*self.Targets.xyzval_vectors(),
                                          return_ts=True,
                                          return_sv=True,
                                          idealized_beampattern=self.UseIdealizedBeampattern)

            X, Y, Z = self.Multibeam.raytrace_wci()

            if min_sv is not None:
                X  = X[SV > min_sv]
                Y  = Y[SV > min_sv]
                Z  = Z[SV > min_sv]
                TS = TS[SV > min_sv]
                SV = SV[SV > min_sv]

            imagessums, imagesnums = grid_wci(
                self.Methods,
                imagessums,
                imagesnums,
                gridder,
                self.Multibeam,
                TS, SV,
                X, Y, Z
            )

        self.Multibeam.set_navigation(0, 0, 0, 0, 0, 0)

        scatterGrids = sg.ScatterGridDict()
        for m in imagessums.keys():
            scatterGrids[m] = sg.ScatterGrid(imagessums[m],imagesnums[m],gridder)

        return scatterGrids


    @staticmethod
    def print_methods():
        """Print all available gridding methods"""
        GriddingMethod.print_all()
