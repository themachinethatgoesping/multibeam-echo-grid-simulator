#!/home/ssd/opt/python/anaconda3/bin/python3

# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

"""
Functions to create bubbles for the MBES simulation
 -
"""

# ------------------- Imports -------------------
import math
import numpy as np

import random

from matplotlib import pyplot as plt

import mbes_sim.functions.helperfunctions as hlp

# -------------------  Bubbles class -----------------------------
class Targets(object):
    """
    This is a container for bubble targets (x,y,z,val) that
    also provides convenient plotting functions
    """
    def __init__(self,
                 targets_x : np.ndarray,
                 targets_y : np.ndarray,
                 targets_z : np.ndarray,
                 targets_vals : np.ndarray):
        """Initialize a targets object

        Parameters
        ----------
        targets_x : np.ndarray
            target x positions in m
        targets_y : np.ndarray
            target y positions in m
        targets_z : np.ndarray
            target z positions in m
        targets_vals : np.ndarray
            target scattering strength values
        """

        self.x = targets_x
        self.y = targets_y
        self.z = targets_z
        self.val = targets_vals

    @staticmethod
    def init_empty():
        """Create an empty targets object

        Returns
        -------
        Targets
            empty targets object
        """

        return Targets(np.zeros(0,dtype=float),
                       np.zeros(0,dtype=float),
                       np.zeros(0,dtype=float),
                       np.zeros(0,dtype=float))


    def extend(self,targets):
        """extend targets to this target container

        Parameters
        ----------
        targets : Targets
            Targets to be added
        """

        self.x = np.append(self.x,targets.x)
        self.y = np.append(self.y,targets.y)
        self.z = np.append(self.z,targets.z)
        self.val = np.append(self.val,targets.val)

    def append(self, x: float, y: float, z: float, val: float):
        """Add an additional target

        Parameters
        ----------
        x : float
            x position of the target in m
        y : float
            y position of the target in m
        z : float
            z position of the target in m
        val : float
            scattering strength of the target
        """

        self.x = np.append(self.x,x)
        self.y = np.append(self.y,y)
        self.z = np.append(self.z,z)
        self.val = np.append(self.val,val)


    def cutDepthLayer(self, minZ: float, maxZ: float):
        """Return all targets within a depth range

        Parameters
        ----------
        minZ : float
            Minimum depth in m
        maxZ : float
            Maximum depth in m

        Returns
        -------
        Targets
            Targets within the depth range
        """
        indices = np.array(range(len(self.z)))

        indices[self.z < minZ] = -1
        indices[self.z > maxZ] = -1

        indices = indices[indices >= 0]

        return Targets(self.x[indices],
                       self.y[indices],
                       self.z[indices],
                       self.val[indices])

    def xyzval_vectors(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Return the targets as vectors

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            x,y,z,val vectors
        """
        return self.x,self.y,self.z,self.val

    def __len__(self) -> int:
        """Return the number of targets in the container

        Returns
        -------
        int
            Number of targets in the container
        """
        return len(self.x)

    def __getitem__(self, index: int) -> (float, float, float, float):
        """Get a target by index

        Parameters
        ----------
        index : int
            target index

        Returns
        -------
        (float, float, float, float)
            x,y,z,val of the target

        Raises
        ------
        IndexError
            If the index is out of bounds
        """
        if index < 0:
            index = len(self.x) - index

        if index > 0:
            raise IndexError("bubbles out of bounds")

        return self.x[index], self.y[index], self.z[index], self.val[index]

    def sum(self) -> float:
        """Return the sum of all target scattering strength values

        Returns
        -------
        float
            Sum of all target scattering strength values
        """
        
        return np.sum(self.val)

    def plot(self,
             front_view : bool = True,
             side_view : bool = True,
             top_view : bool = True,
             histogram : bool = True,
             name: str = None,
             fig = None,
             close_plots: bool = False):
        """Plotting function for the targets inside the container

        Parameters
        ----------
        front_view : bool, optional
            Plot target front view, by default True
        side_view : bool, optional
            Plot target side view, by default True
        top_view : bool, optional
            Plot target top view, by default True
        histogram : bool, optional
            Plot target value histogram, by default True
        name : str, optional
            Plot name, by default None
        fig : _type_, optional
            If provided: create the plot within this figure, by default None
        close_plots : bool, optional
            Close plots before creating a new function (handy when plotting inside jupyter notebooks), by default False

        Returns
        -------
        _type_
            Matplotlib axes
        """

        if fig is None:
            if name is None:
                name = 'Targets'
            else:
                name = 'Targets - ' + name
            if close_plots: plt.close(name)
            fig = plt.figure(name)
            fig.clf()


        nplots = front_view + side_view + top_view + histogram

        if nplots == 0:
            return

        if nplots <= 2:
            axes = fig.subplots(ncols = nplots)
        else:
            axes = fig.subplots(ncols=2, nrows = nplots - 2)

        try:
            axit = axes.flat
        except:
            axit = np.array([axes]).flat

        if front_view:
            ax = next(axit)
            ax.clear()
            ax.set_title('targets front view')
            ax.scatter(self.y, -self.z, c=self.val)
            ax.set_aspect('equal')
            ax.set_xlabel("y")
            ax.set_xlabel("y")
            ax.set_ylabel("z")
            ax.set_ylabel("z")

        if side_view:
            ax = next(axit)
            ax.clear()
            ax.set_title('targets side view')
            ax.scatter(self.x, -self.z, c=self.val)
            ax.set_aspect('equal')
            ax.set_xlabel("x")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_ylabel("z")
            ax.set_xlim(-220, 220)

        if top_view:
            ax = next(axit)
            ax.clear()
            ax.set_title('targets top view')
            ax.scatter(self.y[self.val > 0],
                       self.x[self.val > 0],
                       c=self.val[self.val > 0])
            ax.set_aspect('equal')
            ax.set_xlabel("y")
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            ax.set_ylabel("x")
            ax.set_ylim(-220, 220)
            ax.set_xlim(-220, 220)

        if histogram:
            ax = next(axit)
            ax.set_title('targets value histogram')
            ax.hist(self.val)

        return axes



# ------------------- Create bubbles functions -------------------
class BubbleGenerator(object):

    def __init__(self,
                 base_x: float = 0.0,
                 base_y: float = 0.0,
                 base_z: float = 0.0,
                 base_val: float = 1.0,
                 sigma_val: float = 0.0,
                 sigma_x: float = 0.2,
                 sigma_y: float = 0.2,
                 sigma_z: float = 0.5,
                 normalize_bubbles_mean: bool = True):
        """Class that allows for generating bubbles (targets) at random positions within a specified water volume

        Parameters
        ----------
        base_x : float, optional
            All returned bubble positions will be relative to this x coordinate, by default 0.0
        base_y : float, optional
            All returned bubble positions will be relative to this y coordinate, by default 0.0
        base_z : float, optional
            All returned bubble positions will be relative to this z coordinate, by default 0.0
        base_val : float, optional
            Base value for bubble scattering strength, by default 1.0
        sigma_val : float, optional
            Gauss sigma variation of bubble scattering strength (mu = base_val), by default 0.0
        sigma_x : float, optional
            Gauss sigma variation x of bubble positions relative to bubble stream center, by default 0.2
        sigma_y : float, optional
            Gauss sigma variation y of bubble positions relative to bubble stream center, by default 0.2
        sigma_z : float, optional
            Gauss sigma variation z of bubble positions relative to vertical distance between bubbles, by default 0.5
        normalize_bubbles_mean : bool, optional
            If True, the mean over all bubble value will be normalized such that the sum over all
                                       bubbles is always nbubbles*base_val. Note: for nbubbles == 1, this means that
                                       the single bubble will always have the value base_val despite the set
                                       sigma_val!, by default True
        """

        self.base_x = base_x
        self.base_y = base_y
        self.base_z = base_z

        self.base_val = base_val
        self.sigma_val = sigma_val

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z

        self.normalize_bubbles_mean = normalize_bubbles_mean

        # internal use
        self.max_tries = 1000

    def generate_bubble_val(self):
        """Generate a random scattering strength value. Could be used in the future to fit to a given bubble radii distribution

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        RuntimeError
            _description_
        """
        # TODO: This should be improved (loop to ensure positive value is not very efficient)
        for _ in range(self.max_tries):
            val = self.base_val + random.gauss(0, self.sigma_val)
            if val > 0:
                break
        else:
            # this will execute if the loop runs till the end without ever reaching a break
            raise RuntimeError('reached max tries without creating bubble value')

        return val

    def generate_bubbles_within_cylindrical_section_along_path(self,
                                                               start_x: float,
                                                               end_x: float,
                                                               min_range: float,
                                                               max_range: float,
                                                               min_beamsteeringangle: float,
                                                               max_beamsteeringangle: float,
                                                               nbubbles=1,
                                                               min_y: float = None,
                                                               max_y: float = None,
                                                               min_z: float = None,
                                                               max_z: float = None,
                                                               uniform_likelyhood_along_x_range_angle=False
                                                               ) -> Targets:
        """Creates nbubbles bubbles at a random position within a cylindrical sector that is spanned between start/end x,
        min/max range, min/max beamsteering_angle. This function can be used to generate bubbles within the observable
        water volume of a forward moving vessel. min/max x/y/z can be used to additionally constrain the allowed
        volume.

        The likelyhood/density of the bubbles is equally spread along the euclidean volume (x,y,z) unless
        normalize_likelyhood_along_XRangeAngle is set (then the likelyhood of x, r, beamsteering_angle will be uniform

        Parameters
        ----------
        start_x : float
            Start x position of the cylinder (defined positive forward)
        end_x : float
            End x position (defined positive forward)
        min_range : float
            Minimum range of cylinder
        max_range : float
            Maximum range of cylinder
        min_beamsteeringangle : float
            Minimum angle of the cylinder section [°, -90° (port) -> 90° (starboard)]
        max_beamsteeringangle : float
            Maximum angle of the cylinder section [°, -90° (port) -> 90° (starboard)]
        nbubbles : int, optional
            number of bubbles to generate, by default 1
        min_y : float, optional
            Additional minimum y constraint for bubble targets [positive starboard], by default None
        max_y : float, optional
            Additional maximum y constraint for bubble targets [positive starboard], by default None
        min_z : float, optional
            Additional minimum z constraint for bubble targets [positive downwards], by default None
        max_z : float, optional
            Additional maximum z constraint for bubble targets [positive downwards], by default None
        uniform_likelyhood_along_x_range_angle : bool, optional
            If true The likelyhood/density of bubbles is uniform along x,range,
            and angle, rather than x,y and z, by default False

        Returns
        -------
        Targets
            Container containing the generated bubble targets


        Raises
        ------
        RuntimeError
            _description_
        RuntimeError
            _description_
        """

        # create bubble stream array
        bubbles_x = np.empty(nbubbles)
        bubbles_y = np.empty(nbubbles)
        bubbles_z = np.empty(nbubbles)
        bubbles_val = np.empty(nbubbles)

        # for equal density in euclidean volume, some min/max y/x values are needed
        if not uniform_likelyhood_along_x_range_angle:
            if min_y is None:
                min_y = max_range * math.sin(math.radians(min_beamsteeringangle))
            if max_y is None:
                max_y = max_range * math.sin(math.radians(max_beamsteeringangle))
            if min_z is None:
                min_z = 0
            if max_z is None:
                max_z = max_range

        # loop through nbubbles and create bubbles
        for n in range(nbubbles):

            x = random.uniform(start_x, end_x)

            if uniform_likelyhood_along_x_range_angle:
                # regenerate angle and range until it fits within min/max y and min/max z
                for _ in range(self.max_tries):
                    angle = random.uniform(min_beamsteeringangle, max_beamsteeringangle)
                    r = random.uniform(min_range, max_range)

                    y = r * math.sin(math.radians(angle))
                    z = r * math.cos(math.radians(angle))

                    if y < min_y or y > max_y:
                        continue
                    if z < min_z or z > max_z:
                        continue

                    break
                else:
                    # this will execute if the loop runs till the end without ever reaching a break
                    raise RuntimeError('reached max tries without creating bubble')
            else:
                # regenerate x and y until it fits within the cylindrical section
                for _ in range(self.max_tries):
                    z = random.uniform(min_z, max_z)
                    y = random.uniform(min_y, max_y)

                    r = math.sqrt(z * z + y * y)
                    if r == 0:
                        continue

                    angle = math.degrees(math.asin(y / r))

                    if r < min_range or r > max_range:
                        continue
                    if angle < min_beamsteeringangle or angle > max_beamsteeringangle:
                        continue

                    break
                else:
                    # this will execute if the loop runs till the end without ever reaching a break
                    raise RuntimeError('reached max tries without creating bubble stream')

            bubbles_x[n] = x + self.base_x
            bubbles_y[n] = y + self.base_y
            bubbles_z[n] = z + self.base_z
            bubbles_val[n] = self.generate_bubble_val()

        # normalize sum
        if self.normalize_bubbles_mean:
            bubbles_val *= self.base_val / bubbles_val.mean()

        return Targets(bubbles_x, bubbles_y, bubbles_z, bubbles_val)

    def generate_bubblestreams_within_cylindrical_section_along_path(self,
                                                                     start_x: float,
                                                                     end_x: float,
                                                                     min_range: float,
                                                                     max_range: float,
                                                                     min_beamsteeringangle: float,
                                                                     max_beamsteeringangle: float,
                                                                     zdist_list=None,
                                                                     min_x: float = np.nan,
                                                                     max_x: float = np.nan,
                                                                     min_y: float = np.nan,
                                                                     max_y: float = np.nan,
                                                                     min_z: float = np.nan,
                                                                     max_z: float = np.nan,
                                                                     return_zero_bubbles: bool = False
                                                                     ) -> Targets:
        """Creates len(nbubbles_list) bubble streams with nbubbles_list[i] bubbles at a random position within a
        cylindrical sector that is spanned between start/end x,
        min/max range, min/max beamsteering_angle. This function can be used to generate bubbles within the observable
        water volume of a forward moving vessel. min/max x/y/z can be used to additionally constrain the allowed
        volume.

        The likelyhood/density of the bubbles is equally spread along the euclidean volume (x,y,z) unless
        normalize_likelyhood_along_XRangeAngle is set (then the likelyhood of x, r, beamsteering_angle will be uniform

        Parameters
        ----------
        start_x : float
            Start x position of the cylinder (defined positive forward)
        end_x : float
            End x position (defined positive forward)
        min_range : float
            Minimum range of cylinder
        max_range : float
            Maximum range of cylinder
        min_beamsteeringangle : float
            Minimum angle of the cylinder section [°, -90° (port) -> 90° (starboard)]
        max_beamsteeringangle : float
            Maximum angle of the cylinder section [°, -90° (port) -> 90° (starboard)]
        zdist_list : list
            List of z distances for bubble streams. len(zdist_list) == number of bubble streams
            zdist is the vertical distance of the bubbles in [m]
        min_x : float, optional
            Additional minimum x constraint for bubble targets [positive starboard], by default None
        max_x : float, optional
            Additional maximum x constraint for bubble targets [positive starboard], by default None
        min_y : float, optional
            Additional minimum y constraint for bubble targets [positive starboard], by default None
        max_y : float, optional
            Additional maximum y constraint for bubble targets [positive starboard], by default None
        min_z : float, optional
            Additional minimum z constraint for bubble targets [positive downwards], by default None
        max_z : float, optional
            Additional maximum z constraint for bubble targets [positive downwards], by default None

        Returns
        -------
        Targets
            Container containing the generated bubble targets

        Raises
        ------
        RuntimeError
            _description_
        """

        if zdist_list is None:
            zdist_list = [0.1]

        bubble_stream_x = []
        bubble_stream_y = []
        bubble_stream_z = []
        bubble_stream_val = []

        if not np.isfinite(min_y):
            min_y = max_range * math.sin(math.radians(min_beamsteeringangle))
        if not np.isfinite(max_y):
            max_y = max_range * math.sin(math.radians(max_beamsteeringangle))
        if not np.isfinite(min_z):
            min_z = 0
        if not np.isfinite(max_z):
            max_z = max_range

        for n_stream, zdist in enumerate(zdist_list):

            nbubbles = math.floor(abs((max_z - min_z)) / zdist + 1)
            
            assert (nbubbles > 1), "zdist is too large! (nbubbles < 2)"

            # try to create a bubble stream with at least one valid bubble
            for _ in range(self.max_tries):

                stream_z = np.linspace(min_z, max_z, nbubbles)

                stream_x = np.zeros(stream_z.shape)
                stream_y = np.zeros(stream_z.shape)
                stream_val = np.ones(stream_z.shape)

                stream_x += random.uniform(start_x, end_x)
                stream_y += random.uniform(min_y, max_y)

                for n in range(nbubbles):

                    stream_x[n] += random.gauss(0, self.sigma_x) + self.base_x
                    stream_y[n] += random.gauss(0, self.sigma_y) + self.base_y
                    stream_z[n] += random.uniform(-zdist * self.sigma_z, +zdist * self.sigma_z) + self.base_z
                    stream_val[n] += self.generate_bubble_val()

                    r = math.sqrt(stream_z[n] * stream_z[n] + stream_y[n] * stream_y[n])
                    angle = math.degrees(math.asin(stream_y[n] / r))

                    if max_range < r or r < min_range \
                            or max_beamsteeringangle < angle or angle < min_beamsteeringangle \
                            or max_x < stream_x[n] or stream_x[n] < min_x \
                            or max_y < stream_y[n] or stream_y[n] < min_y \
                            or max_z < stream_z[n] or stream_z[n] < min_z:
                        stream_val[n] = 0

                if stream_val.mean() > 0:
                    break
            else:
                # this will execute if the loop runs till the end without ever reaching a break
                raise RuntimeError('reached max tries without creating bubble stream')

            # normalize sum for individual stream
            if self.normalize_bubbles_mean:
                stream_val *= self.base_val / stream_val[stream_val > 0].mean()

            if not return_zero_bubbles:
                stream_x = stream_x[stream_val > 0]
                stream_y = stream_y[stream_val > 0]
                stream_z = stream_z[stream_val > 0]
                stream_val = stream_val[stream_val > 0]

            bubble_stream_x.extend(stream_x)
            bubble_stream_y.extend(stream_y)
            bubble_stream_z.extend(stream_z)
            bubble_stream_val.extend(stream_val)

        return Targets(np.asarray(bubble_stream_x, dtype=float), np.asarray(bubble_stream_y, dtype=float),
                       np.asarray(bubble_stream_z, dtype=float), np.asarray(bubble_stream_val, dtype=float))

    def generate_horizontal_bubblestreams_within_cylindrical_section_along_path(self,
                                                                     start_x: float,
                                                                     end_x: float,
                                                                     min_range: float,
                                                                     max_range: float,
                                                                     min_beamsteeringangle: float,
                                                                     max_beamsteeringangle: float,
                                                                     zdist_list=None,
                                                                     min_x: float = np.nan,
                                                                     max_x: float = np.nan,
                                                                     min_y: float = np.nan,
                                                                     max_y: float = np.nan,
                                                                     min_z: float = np.nan,
                                                                     max_z: float = np.nan,
                                                                     return_zero_bubbles: bool = False
                                                                     ) -> Targets:
        """Creates len(nbubbles_list) bubble streams with nbubbles_list[i] bubbles at a random position within a
        cylindrical sector that is spanned between start/end x,
        min/max range, min/max beamsteering_angle. This function can be used to generate bubbles within the observable
        water volume of a forward moving vessel. min/max x/y/z can be used to additionally constrain the allowed
        volume.

        The likelyhood/density of the bubbles is equally spread along the euclidean volume (x,y,z) unless
        normalize_likelyhood_along_XRangeAngle is set (then the likelyhood of x, r, beamsteering_angle will be uniform

        Parameters
        ----------
        start_x : float
            Start x position of the cylinder (defined positive forward)
        end_x : float
            End x position (defined positive forward)
        min_range : float
            Minimum range of cylinder
        max_range : float
            Maximum range of cylinder
        min_beamsteeringangle : float
            Minimum angle of the cylinder section [°, -90° (port) -> 90° (starboard)]
        max_beamsteeringangle : float
            Maximum angle of the cylinder section [°, -90° (port) -> 90° (starboard)]
        zdist_list : list, optional
            List of z distances for bubble streams. len(zdist_list) == number of bubble streams
                           zdist is the vertical distance of the bubbles in [m], by default None
        min_x : float, optional
            Additional minimum x constraint for bubble targets [positive starboard], by default None
        max_x : float, optional
            Additional maximum x constraint for bubble targets [positive starboard], by default None
        min_y : float, optional
            Additional minimum y constraint for bubble targets [positive starboard], by default None
        max_y : float, optional
            Additional maximum y constraint for bubble targets [positive starboard], by default None
        min_z : float, optional
            Additional minimum z constraint for bubble targets [positive downwards], by default None
        max_z : float, optional
            Additional maximum z constraint for bubble targets [positive downwards], by default None
        return_zero_bubbles : bool, optional
            If True, bubbles of stream outside min/max range and min/max beamsteering angle
            are returned with value 0, otherwise they are excluded, by default False

        Returns
        -------
        Targets
            Container containing the generated bubble targets

        Raises
        ------
        RuntimeError
            _description_
        """

        if zdist_list is None:
            zdist_list = [0.1]

        bubble_stream_x = []
        bubble_stream_y = []
        bubble_stream_z = []
        bubble_stream_val = []

        if not np.isfinite(min_x):
            min_x = start_x - (end_x-start_x)*0.25
        if not np.isfinite(max_x):
            max_x = end_x + (end_x-start_x)*0.25
        if not np.isfinite(min_y):
            min_y = max_range * math.sin(math.radians(min_beamsteeringangle))
        if not np.isfinite(max_y):
            max_y = max_range * math.sin(math.radians(max_beamsteeringangle))
        if not np.isfinite(min_z):
            min_z = 0
        if not np.isfinite(max_z):
            max_z = max_range


        for n_stream, zdist in enumerate(zdist_list):

            nbubbles = math.floor(abs((max_x - min_x)) / zdist + 1)

            assert (nbubbles > 1), "zdist is too large! (nbubbles < 2)"

            # try to create a bubble stream with at least one valid bubble
            for _ in range(self.max_tries):

                stream_x = np.linspace(min_x, max_x, nbubbles)

                stream_z = np.zeros(stream_x.shape)
                stream_y = np.zeros(stream_x.shape)
                stream_val = np.ones(stream_x.shape)

                #stream_x += random.uniform(start_x, end_x)
                stream_y += random.uniform(min_y, max_y)
                stream_z += random.uniform(min_z, max_z)

                for n in range(nbubbles):

                    stream_x[n] += random.uniform(-zdist * self.sigma_z, +zdist * self.sigma_z) + self.base_x
                    #stream_x[n] += random.uniform(-zdist * self.sigma_z, +zdist * self.sigma_z) + 0
                    stream_y[n] += random.gauss(0, self.sigma_y) + self.base_y
                    stream_z[n] += random.gauss(0, self.sigma_x) + self.base_z
                    stream_val[n] += self.generate_bubble_val()

                    r = math.sqrt(stream_z[n] * stream_z[n] + stream_y[n] * stream_y[n])
                    angle = math.degrees(math.asin(stream_y[n] / r))

                    if max_range < r or r < min_range \
                            or max_beamsteeringangle < angle or angle < min_beamsteeringangle \
                            or max_x < stream_x[n] or stream_x[n] < min_x \
                            or max_y < stream_y[n] or stream_y[n] < min_y \
                            or max_z < stream_z[n] or stream_z[n] < min_z:
                        stream_val[n] = 0

                if stream_val.mean() > 0:
                    break
            else:
                # this will execute if the loop runs till the end without ever reaching a break
                raise RuntimeError('reached max tries without creating bubble stream')

            # normalize sum for individual stream
            if self.normalize_bubbles_mean:
                stream_val *= self.base_val / stream_val[stream_val > 0].mean()

            if not return_zero_bubbles:
                stream_x = stream_x[stream_val > 0]
                stream_y = stream_y[stream_val > 0]
                stream_z = stream_z[stream_val > 0]
                stream_val = stream_val[stream_val > 0]

            bubble_stream_x.extend(stream_x)
            bubble_stream_y.extend(stream_y)
            bubble_stream_z.extend(stream_z)
            bubble_stream_val.extend(stream_val)

        return Targets(np.asarray(bubble_stream_x, dtype=float), np.asarray(bubble_stream_y, dtype=float),
                       np.asarray(bubble_stream_z, dtype=float), np.asarray(bubble_stream_val, dtype=float))

if __name__ == "__main__":

    import matplotlib as mpl

    mpl.rcParams['figure.dpi'] = 200

    #mbes = MBES(120, 1, 1)

    bubbleGenerator = BubbleGenerator(sigma_val=0.3,
                                      sigma_x=1,
                                      sigma_y=1)


    targets = \
        bubbleGenerator.generate_bubblestreams_within_cylindrical_section_along_path(
            start_x=-200,
            end_x=200,
            min_range=10,
            max_range=120,
            min_beamsteeringangle=-60,
            max_beamsteeringangle=60,
            min_z=8,
            # max_z=90,
            min_y=-90,
            max_y=60,
            zdist_list=[0.1, 1, 5],
            return_zero_bubbles=True
        )
    targets.plot(name = 'Bubblestreams')


    plt.show()
