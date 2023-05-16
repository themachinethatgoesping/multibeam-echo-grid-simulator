# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

"""
Functions and classes for gridding scatter point values to a regular grid
"""

from numba import njit, prange
import numba.types as ntypes
import numpy as np
import math
import numpy as np

import mbes_sim.functions.helperfunctions as hlp

@njit
def get_minmax(sx: np.ndarray, sy: np.ndarray, sz: np.ndarray) -> (float, float, float, float, float, float):
    """Numba accelerated function to get the minimum and maximum values of given x, y and z coordinate arrays

    Parameters
    ----------
    sx : np.ndarray
        x coordinates
    sy : np.ndarray
        y coordinates
    sz : np.ndarray
        z.coordinates

    Returns
    -------
    (float, float, float, float, float, float)
        minx, maxx, miny, maxy, minz, maxz
    """
    minx = np.nan
    maxx = np.nan
    miny = np.nan
    maxy = np.nan
    minz = np.nan
    maxz = np.nan

    for i in range(len(sx)):
        x = sx[i]
        y = sy[i]
        z = sz[i]

        if not x > minx: minx = x
        if not x < maxx: maxx = x
        if not y > miny: miny = y
        if not y < maxy: maxy = y
        if not z > minz: minz = z
        if not z < maxz: maxz = z

    return minx, maxx, miny, maxy, minz, maxz



@njit
def get_index(val: float, grd_val_min: float, grd_res: float) -> int:
    """Get the nearest index for a specified value in a grid with the specified minimum value and resolution

    Parameters
    ----------
    val : float
        Coordinate value
    grd_val_min : float
        coordinate of the first grid cell
    grd_res : float
        resolution of the grid

    Returns
    -------
    int
        index of the nearest grid cell
    """
    return hlp.round_int((val - grd_val_min) / grd_res)

@njit
def get_index_fraction(val: float, grd_val_min: float, grd_res: float) -> float:
    """Convert the specified coordinate value to a fractional index in a grid within the specified minimum value and resolution

    Parameters
    ----------
    val : float
        Coordinate value
    grd_val_min : float
        coordinate of the first grid cell
    grd_res : float
        resolution of the grid

    Returns
    -------
    float
        fractional index within the grid
    """
    return (val - grd_val_min) / grd_res


@njit
def get_value(index: float, grd_val_min: float, grd_res: float) -> float:
    """Convert a grid index (int or fractional) to a coordinate value
    The grid is defined by the minimum value and the resolution

    Parameters
    ----------
    index : float or int
        Grid index (int or fractional)
    grd_val_min : float
        coordinate of the first grid cell
    grd_res : float
        resolution of the grid

    Returns
    -------
    float
        coordinate value
    """

    return grd_val_min + grd_res * float(index)


@njit
def get_grd_value(value: float, grd_val_min: float, grd_res: float) -> float:
    """Get the coordinate of the nearest grid cell for a specified value in a grid with the specified minimum value and resolution

    Parameters
    ----------
    value : float
        Coordinate value
    grd_val_min : float
        coordinate of the first grid cell
    grd_res : float
        resolution of the grid

    Returns
    -------
    float
        Coordinate of the nearest grid cell
    """
    return get_value(get_index(value, grd_val_min, grd_res), grd_val_min, grd_res)



@njit
def get_index_vals(fraction_index_x : float,
                   fraction_index_y : float,
                   fraction_index_z : float) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):

    """Return a vector with fraction and weights for the neighboring grid cells.
    This allows for a linear/weighted mean interpolation 
    :return: 

    Parameters
    ----------
    fraction_index_x : float
        fractional x index (e.g 4.2)
    fraction_index_y : float
        fractional y index (e.g 4.2)
    fraction_index_z : float
        fractional z index (e.g 4.2)

    Returns
    -------
    (np.ndarray,np.ndarray,np.ndarray,np.ndarray)
        - vec X (x indices as int): all indices "touched" by the fractional point
        - vec Y (Y indices as int): all indices "touched" by the fractional point
        - vec Z (Z indices as int): all indices "touched" by the fractional point
        - vec Weights (Weights indices as float): weights
    """

    ifraction_x = fraction_index_x % 1
    ifraction_y = fraction_index_y % 1
    ifraction_z = fraction_index_z % 1

    fraction_x = 1 - ifraction_x
    fraction_y = 1 - ifraction_y
    fraction_z = 1 - ifraction_z

    ix1 = math.floor(fraction_index_x)
    ix2 = math.ceil(fraction_index_x)
    iy1 = math.floor(fraction_index_y)
    iy2 = math.ceil(fraction_index_y)
    iz1 = math.floor(fraction_index_z)
    iz2 = math.ceil(fraction_index_z)

    X = np.array([ix1, ix1, ix1, ix1, ix2, ix2, ix2, ix2])
    Y = np.array([iy1, iy1, iy2, iy2, iy1, iy1, iy2, iy2])
    Z = np.array([iz1, iz2, iz1, iz2, iz1, iz2, iz1, iz2])

    vx = 1 * fraction_x
    vxy = vx * fraction_y
    vxiy = vx * ifraction_y

    vix = 1 * ifraction_x
    vixy = vix * fraction_y
    vixiy = vix * ifraction_y

    WEIGHT = np.array([
        vxy * fraction_z,
        vxy * ifraction_z,
        vxiy * fraction_z,
        vxiy * ifraction_z,
        vixy * fraction_z,
        vixy * ifraction_z,
        vixiy * fraction_z,
        vixiy * ifraction_z
    ])

    return X, Y, Z, WEIGHT


@njit()
def get_sampled_image2(sx: np.ndarray, sy: np.ndarray, sz: np.ndarray, sv:np.ndarray,
                       xmin: float, xres: float, nx: int,
                       ymin: float, yres: float, ny: int,
                       zmin: float, zres: float, nz: int,
                       imagenum: np.ndarray,
                       imagesum: np.ndarray,
                       skip_invalid:bool=True) -> (np.ndarray, np.ndarray):
    """Gridd 3D scatter points to a given grid. This functino implements a weighted mean gridding.
    The grid is defined by the minimum value and the resolution. 
    Imagenum and Imagesum are used and returned

    Parameters
    ----------
    sx : np.ndarray
        x coordinates of the scatter points
    sy : np.ndarray
        y coordinates of the scatter points
    sz : np.ndarray
        z coordinates of the scatter points
    sv : np.ndarray
        scattering values of the scatter points
    xmin : float
        minimum x coordinate of the grid
    xres : float
        x resolution of the grid
    nx : int
        number of grid cells in x direction
    ymin : float
        minimum y coordinate of the grid
    yres : float
        y resolution of the grid
    ny : int
        number of grid cells in y direction
    zmin : float
        minimum z coordinate of the grid
    zres : float
        z resolution of the grid
    nz : int
        number of grid cells in z direction
    imagenum : np.ndarray
        3D array with the weighted number of scatter points in each grid cell
    imagesum : np.ndarray
        3D array with the sum of the scattering values in each grid cell
    skip_invalid : bool, optional
        _description_, by default True

    Returns
    -------
    (np.ndarray, np.ndarray)
        - imagenum: 3D array with the weighted number of scatter points in each grid cell
        - imagesum: 3D array with the sum of the scattering values in each grid cell
    """

    for i in range(len(sx)):
        x = sx[i]
        y = sy[i]
        z = sz[i]
        v = sv[i]

        IX, IY, IZ, WEIGHT = get_index_vals(
            get_index_fraction(x, xmin, xres),
            get_index_fraction(y, ymin, yres),
            get_index_fraction(z, zmin, zres)
        )

        # for ix,iy,iz,w in zip(IX,IY,IZ,WEIGHT):
        for i_ in range(len(IX)):
            ix = int(IX[i_])
            iy = int(IY[i_])
            iz = int(IZ[i_])
            w = WEIGHT[i_]

            if w == 0:
                continue

            if not skip_invalid:
                if ix < 0: ix = 0
                if iy < 0: iy = 0
                if iz < 0: iz = 0

                if abs(ix) >= nx: ix = nx - 1
                if abs(iy) >= ny: iy = ny - 1
                if abs(iz) >= nz: iz = nz - 1
            else:
                if ix < 0: continue
                if iy < 0: continue
                if iz < 0: continue

                if abs(ix) >= nx: continue
                if abs(iy) >= ny: continue
                if abs(iz) >= nz: continue

            # print(ix,iy,iz,v,w)
            if v >= 0:
                imagesum[ix][iy][iz] += v * w
                imagenum[ix][iy][iz] += w


    return imagesum, imagenum


@njit
def get_sampled_image(sx : np.ndarray, sy: np.ndarray, sz: np.ndarray, sv: np.ndarray,
                      xmin: float, xres: float, nx: int,
                      ymin: float, yres: float, ny: int,
                      zmin: float, zres: float, nz: int,
                      imagenum: np.ndarray,
                      imagesum: np.ndarray,
                      skip_invalid: bool = True):
    """Gridd 3D scatter points to a given grid. This functino implements a block mean gridding.
    The grid is defined by the minimum value and the resolution. 
    Imagenum and Imagesum are used and returned

    Parameters
    ----------
    sx : np.ndarray
        x coordinates of the scatter points
    sy : np.ndarray
        y coordinates of the scatter points
    sz : np.ndarray
        z coordinates of the scatter points
    sv : np.ndarray
        scattering values of the scatter points
    xmin : float
        minimum x coordinate of the grid
    xres : float
        x resolution of the grid
    nx : int
        number of grid cells in x direction
    ymin : float
        minimum y coordinate of the grid
    yres : float
        y resolution of the grid
    ny : int
        number of grid cells in y direction
    zmin : float
        minimum z coordinate of the grid
    zres : float
        z resolution of the grid
    nz : int
        number of grid cells in z direction
    imagenum : np.ndarray
        3D array with the number of scatter points in each grid cell
    imagesum : np.ndarray
        3D array with the sum of the scattering values in each grid cell
    skip_invalid : bool, optional
        _description_, by default True

    Returns
    -------
    (np.ndarray, np.ndarray)
        - imagenum: 3D array with the number of scatter points in each grid cell
        - imagesum: 3D array with the sum of the scattering values in each grid cell
    """

    for i in range(len(sx)):
        x = sx[i]
        y = sy[i]
        z = sz[i]
        v = sv[i]

        ix = get_index(x, xmin, xres)
        iy = get_index(y, ymin, yres)
        iz = get_index(z, zmin, zres)

        if not skip_invalid:
            if ix < 0: ix = 0
            if iy < 0: iy = 0
            if iz < 0: iz = 0

            if abs(ix) >= nx: ix = nx - 1
            if abs(iy) >= ny: iy = ny - 1
            if abs(iz) >= nz: iz = nz - 1
        else:
            if ix < 0: continue
            if iy < 0: continue
            if iz < 0: continue

            if abs(ix) >= nx: continue
            if abs(iy) >= ny: continue
            if abs(iz) >= nz: continue

        if v >= 0:
            imagesum[ix][iy][iz] += v
            imagenum[ix][iy][iz] += 1

    return imagesum, imagenum

class GRIDDER(object):
    """Class to define a grid and grid a set of points
    """

    def __init__(self, xres: float, yres: float, zres: float,
                 min_x: float, max_x: float,
                 min_y: float, max_y: float,
                 min_z: float, max_z: float,
                 xbase: float = 0.,
                 ybase: float = 0.,
                 zbase: float = 0.):
        """Construct a grid object with the given resolution that covers the given extent
        The minimum coordinates of the grid cells are computed to be a multiple of the resolution + base

        The default base is half the resolution which means that the grid cells are centered on the coordinates
        Due to this computation, the grid will typically cover a slightly larger extent than indicated by the min/max coordinates

        Forcing the grid to align with the base and resolution can be useful when combining multiple grids

        Parameters
        ----------
        xres : float
            x resolution in meter
        yres : float
            y resolution in meter
        zres : float
            z resolution in meter
        min_x : float
            smallest x coordinate that should be covered by the grid
        max_x : float
            largest x coordinate that should be covered by the grid
        min_y : float
            smallest y coordinate that should be covered by the grid
        max_y : float
            largest y coordinate that should be covered by the grid
        min_z : float
            smallest z coordinate that should be covered by the grid
        max_z : float
            largest z coordinate that should be covered by the grid
        xbase : float, optional
            base x coordinate for the grid, by default 0
        ybase : float, optional
            base y coordinate for the grid, by default 0
        zbase : float, optional
            base z coordinate for the grid, by default 0
        """
        
        # resolution in meter
        self.xres = xres
        self.yres = yres
        self.zres = zres
        self.xbase = xbase
        self.ybase = ybase
        self.zbase = zbase

        # compute the minimum grid coordinates that are a multiples of the resolution and cover the extent
        nx = math.floor((min_x - self.xbase) / xres)
        ny = math.floor((min_y - self.ybase) / yres)
        nz = math.floor((min_z - self.zbase) / zres)

        self.xmin = nx * xres + self.xbase
        self.ymin = ny * yres + self.ybase
        self.zmin = nz * zres + self.zbase

        # compute the maximum coordinates that are a multiple of the resolution and cover the extent
        nx = math.ceil((max_x - self.xmin) / xres)
        ny = math.ceil((max_y - self.ymin) / yres)
        nz = math.ceil((max_z - self.zmin) / zres)

        self.xmax = nx * xres + self.xmin
        self.ymax = ny * yres + self.ymin
        self.zmax = nz * zres + self.zmin

        # with round, the rounding error will be eliminated which cause res=0.3 to crash
        self.nx = math.floor(round(((self.xmax - self.xmin) / self.xres), 8)) + 1  # num of elements x
        self.ny = math.floor(round(((self.ymax - self.ymin) / self.yres), 8)) + 1  # num of elements y
        self.nz = math.floor(round(((self.zmax - self.zmin) / self.zres), 8)) + 1  # num of elements z

        # borders
        self.border_xmin = self.xmin - xres / 2.0
        self.border_xmax = self.xmax + xres / 2.0
        self.border_ymin = self.ymin - yres / 2.0
        self.border_ymax = self.ymax + yres / 2.0
        self.border_zmin = self.zmin - zres / 2.0
        self.border_zmax = self.zmax + zres / 2.0

    def get_x_index(self, val: float) -> int:
        """Get the x index of the grid cell that contains the given x coordinate

        Parameters
        ----------
        val : float
            x coordinate in meter

        Returns
        -------
        int
            x index
        """
        return get_index(val, self.xmin, self.xres)

    def get_y_index(self, val: float) -> int:
        """Get the y index of the grid cell that contains the given y coordinate

        Parameters
        ----------
        val : float
            y coordinate in meters

        Returns
        -------
        int
            y index
        """
        return get_index(val, self.ymin, self.yres)

    def get_z_index(self, val: float) -> int:
        """Get the z index of the grid cell that contains the given z coordinate

        Parameters
        ----------
        val : float
            z coordinate in meters

        Returns
        -------
        int
            z index
        """
        return get_index(val, self.zmin, self.zres)

    def get_x_index_fraction(self, val: float) -> float:
        """Get the fractional index of the grid cell that contains the given x coordinate

        Parameters
        ----------
        val : float
            x coordinate in meters

        Returns
        -------
        float
            fractional x index
        """
        return get_index_fraction(val, self.xmin, self.xres)

    def get_y_index_fraction(self, val: float) -> float:
        """Get the fractional index of the grid cell that contains the given y coordinate

        Parameters
        ----------
        val : float
            y coordinate in meters

        Returns
        -------
        float
            fractional y index
        """
        return get_index_fraction(val, self.ymin, self.yres)

    def get_z_index_fraction(self, val: float) -> float:
        """Get the fractional index of the grid cell that contains the given z coordinate

        Parameters
        ----------
        val : float
            z coordinate in meters

        Returns
        -------
        float
            fractional z index
        """
        return get_index_fraction(val, self.zmin, self.zres)

    def get_x_value(self, index: float) -> float:
        """get the x coordinate of the grid cell with the given x index

        Parameters
        ----------
        index : int or float
            x index (int or fractional)

        Returns
        -------
        float
            x coordinate in meters
        """
        return get_value(index, self.xmin, self.xres)

    def get_y_value(self, index: float) -> float:
        """get the y coordinate of the grid cell with the given y index

        Parameters
        ----------
        index : float
            y index (int or fractional)

        Returns
        -------
        float
            y coordinate in meters
        """
        return get_value(index, self.ymin, self.yres)

    def get_z_value(self, index: float) -> float:
        """get the z coordinate of the grid cell with the given z index

        Parameters
        ----------
        index : float
            z index (int or fractional)

        Returns
        -------
        float
            z coordinate in meters
        """
        return get_value(index, self.zmin, self.zres)

    def get_x_grd_value(self, value: float) -> float:
        """get the x coordinate of the grid cell that contains the given x coordinate

        Parameters
        ----------
        value : float
            x coordinate in meters

        Returns
        -------
        float
            x coordinate in meters
        """
        return self.get_x_value(self.get_x_index(value))

    def get_y_grd_value(self, value: float) -> float:
        """get the y coordinate of the grid cell that contains the given y coordinate

        Parameters
        ----------
        value : float
            y coordinate in meters

        Returns
        -------
        float
            y coordinate in meters
        """
        return self.get_y_value(self.get_y_index(value))

    def get_z_grd_value(self, value: float) -> float:
        """get the z coordinate of the grid cell that contains the given z coordinate

        Parameters
        ----------
        value : float
            z coordinate in meters

        Returns
        -------
        float
            z coordinate in meters
        """
        return self.get_z_value(self.get_z_index(value))

    def get_extent_x(self) -> (float, float):
        """Get the x extent of the grid (x values of the borders of the first and last grid cell)

        Returns
        -------
        (float, float)
            x values of the borders of the first and last grid cell
        """
        return [self.border_xmin, self.border_xmax]

    def get_extent_y(self) -> (float, float):
        """Get the y extent of the grid (y values of the borders of the first and last grid cell)

        Returns
        -------
        (float, float)
            y values of the borders of the first and last grid cell
        """
        return [self.border_ymin, self.border_ymax]

    def get_extent_z(self) -> (float, float):
        """Get the z extent of the grid (z values of the borders of the first and last grid cell)

        Returns
        -------
        (float, float)
            z values of the borders of the first and last grid cell
        """
        return [self.border_zmin, self.border_zmax]

    def get_min_and_offset(self) -> (float, float, float, float, float, float, float, float, float):
        """Get the minimum values and offsets of the grid (used in internal functions)

        Returns
        -------
        (float, float, float, float, float, float, float, float, float)
            xmin, xres, nx, ymin, yres, ny, zmin, zres, nz
        """
        return self.xmin, self.xres, self.nx, self.ymin, self.yres, self.ny, self.zmin, self.zres, self.nz

    def get_x_coordinates(self) -> list:
        """Get all x coordinates of the grid

        Returns
        -------
        list
            list of x coordinates
        """

        coordinates = []
        for i in range(self.nx):
            coordinates.append(self.get_x_value(i))

        return coordinates

    def get_y_coordinates(self) -> list:
        """Get all y coordinates of the grid

        Returns
        -------
        list
            list of y coordinates
        """

        coordinates = []
        for i in range(self.ny):
            coordinates.append(self.get_y_value(i))

        return coordinates

    def get_z_coordinates(self) -> list:
        """Get all z coordinates of the grid

        Returns
        -------
        list
            list of z coordinates
        """

        coordinates = []
        for i in range(self.nz):
            coordinates.append(self.get_z_value(i))

        return coordinates

    def append_sampled_image(self,sx: np.ndarray, sy: np.ndarray, sz: np.ndarray, s_val: np.ndarray,
                             imagesum: np.ndarray, imagenum: np.ndarray,
                             skip_invalid:bool = True):
        """Grid the given sample data onto a given image (imagesum and imagenum).
        If imagesum and imagenum are not given, they are created with the size of the grid and filled with zeros. 
        This function uses nearest neighbor interpolation / block averaging. 

        Parameters
        ----------
        sx : np.ndarray
            x coordinates of the scatter points
        sy : np.ndarray
            y coordinates of the scatter points
        sz : np.ndarray
            z coordinates of the scatter points
        s_val : np.ndarray
            scatter values of the scatter points
        imagesum : np.ndarray, optional
            3D image with sum of values per grid cell (e.g. from a previous call to get_sampled_image)
        imagenum : np.ndarray, optional
            3D image with number of values per grid cell (e.g. from a previous call to get_sampled_image)
        skip_invalid : bool, optional
            skip points that are outside the grid, by default True

        Returns
        -------
        (np.ndarray,np.ndarray)
            image with sum of- and image with number of values per grid cell, dimensions: (nx, ny, nz)
        """

        if imagenum is None:
            imagenum = np.zeros((self.nx, self.ny, self.nz)).astype(np.int64)
        if imagesum is None:
            imagesum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)
        return get_sampled_image(sx, sy, sz, s_val, *self.get_min_and_offset(),imagenum = imagenum, imagesum = imagesum,skip_invalid = skip_invalid)

    def append_sampled_image2(self,sx, sy, sz, s_val,
                             imagesum, imagenum,
                             skip_invalid = True
                              ):
        """Grid the given sample data onto a given image (imagesum and imagenumb). 
        If imagesum and imagenum are not given, they are created with the size of the grid and filled with zeros. 
        This function uses nearest linear interpolation / weighted mean averaging.

        Parameters
        ----------
        sx : np.ndarray
            x coordinates of the scatter points
        sy : np.ndarray
            y coordinates of the scatter points
        sz : np.ndarray
            z coordinates of the scatter points
        s_val : np.ndarray
            scatter values of the scatter points
        imagesum : np.ndarray, optional
            3D image with sum of values per grid cell (e.g. from a previous call to get_sampled_image)
        imagenum : np.ndarray, optional
            3D image with weighted number of values per grid cell (e.g. from a previous call to get_sampled_image)
        skip_invalid : bool, optional
            skip points that are outside the grid, by default True

        Returns
        -------
        (np.ndarray,np.ndarray)
            image with sum of- and image with number of values per grid cell, dimensions: (nx, ny, nz)
        """

        if imagenum is None:
            imagenum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)
        if imagesum is None:
            imagesum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)

        return get_sampled_image2(sx, sy, sz, s_val, *self.get_min_and_offset(), imagenum=imagenum, imagesum=imagesum,
                                  skip_invalid=skip_invalid)


if __name__ == '__main__':
    minx = -22
    maxx =  22
    miny = -22
    maxy =  22
    minz = -120
    maxz = 0

    xres = 1
    yres = 1
    zres = 1

    gridder = GRIDDER(xres, yres, zres,
                      minx, maxx,
                      miny, maxy,
                      minz, maxz,
                      xbase=0.5)

    print(minx, gridder.get_x_grd_value(minx), gridder.get_x_index(minx))
    print(maxx, gridder.get_x_grd_value(maxx), gridder.get_x_index(maxx))
    print()
    print(xres, gridder.get_x_value(1) - gridder.get_x_value(0))
    print()
    index = 10
    print(index, gridder.get_x_grd_value(index))

    print(gridder.get_extent_x())

