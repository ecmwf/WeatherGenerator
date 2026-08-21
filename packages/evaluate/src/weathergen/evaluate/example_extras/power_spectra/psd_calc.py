import iris
import iris.cube
import numpy as np


def psd(ht: np.typing.NDArray) -> np.typing.NDArray:
    """
    Returns a power spectrum density for the positive non-zero frequencies
    Assumes ht has an even number of points
    """
    n = len(ht)
    # Hf      = np.fft.fft(ht, norm='forward')
    hf = np.fft.rfft(ht, norm="forward")
    psd = np.abs(hf[1 : round(n / 2 + 1)]) ** 2
    # Compensate for positive frequencies only
    psd *= 2.0
    return psd


def cubepsd(
    cubes: iris.cube.CubeList | iris.cube.Cube, dimension: str = "longitude"
) -> np.typing.NDArray:
    """
    Returns a power spectrum density for a cube
    Assumes that cube.data has an even number of points in dimension dim
    """

    if isinstance(cubes, iris.cube.CubeList):
        # being passed a cube list
        npoints = len(cubes[0].coord(dimension).points)
    else:
        # Assume it is just a cube
        npoints = len(cubes.coord(dimension).points)

    field_psd = np.zeros([round(npoints / 2)])

    nloc = 0
    if isinstance(cubes, iris.cube.CubeList):
        for cube in cubes:
            for field_slice in cube.slices([dimension]):
                nloc += 1
                field_psd += psd(field_slice.data)
    else:
        for field_slice in cubes.slices([dimension]):
            nloc += 1
            field_psd += psd(field_slice.data)

    field_psd /= nloc

    return field_psd


def calcposfreq(cube: iris.cube.Cube, dimension: str = "longitude") -> np.typing.NDArray:
    """
    Given a cube and dimension returns the positive frequencies
    Assumes gridpoints are evenly spaced
    """
    npoints = len(cube.coord(dimension).points)

    # Create frequencies
    freq = np.fft.fftfreq(npoints, d=360.0 / npoints)

    # Positive half
    posfreq = np.absolute(freq[1 : round(npoints / 2 + 1)])

    return posfreq
