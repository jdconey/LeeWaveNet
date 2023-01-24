import numpy as np
import os

#this code has been adapted from a Jupyter Notebook created by Leif Denby

train_or_test = 'train'
no_of_examples = 10

root= '~/data/synthetic/'+train_or_test+'/'

if not os.path.isdir(root+'data'):
    os.makedirs(root+'data')
    os.makedirs(root+'amplitude')
    os.makedirs(root+'wavelength')
    os.makedirs(root+'orientation')


r0 = 50
r_sigma = 1.0

def get_grid_coordinates(coords):
    return np.floor(coords).astype("int")


def modified_poisson_disk_sampling(N=100, r0=10, r_sigma=1, k=50, radiusType="default"):
    """
    Implementation of the Poisson Disk Sampling algorithm, but modified so that for each
    point added the radius between points is sampled from a Gaussian distribution

    :param N: grid size in number of pixels (assumed square domain)
    :param r0: length-scale for distance between points
    :param r_sigma: std div for distance between points
    :param k: Number of iterations to find a new particle in an annulus between radius r and 2r from a sample particle.
    :param radiusType: Method to determine the distance to newly spawned particles. 'default' follows the algorithm of
                       Bridson (2007) and generates particles uniformly in the annulus between radius r and 2r.
                       'normDist' instead creates new particles at distances drawn from a normal distribution centered
                       around 1.5r with a dispersion of 0.2r.
    :return: nParticle: Number of particles in the sampling.
             particleCoordinates: 2d array containing the coordinates of the created particles.
             radii: radii for the sampled particles

    based on https://gitlab.com/abittner/poissondisksampling/-/blob/master/poissonDiskSampling/bridsonVariableRadius.py
    """

    def _gen_radius():
        return np.random.normal(r0, r_sigma)

    n = 0

    # Set-up background grid
    gridHeight = gridWidth = N
    grid = np.zeros((gridHeight, gridWidth))

    # Pick initial (active) point
    coords = (np.random.random() * gridHeight, np.random.random() * gridWidth)
    idx = get_grid_coordinates(coords)
    nParticle = 1
    grid[idx[0], idx[1]] = nParticle

    # Initialise active queue
    queue = [
        coords
    ]  # Appending to list is much quicker than to numpy array, if you do it very often
    particleCoordinates = [
        coords
    ]  # List containing the exact positions of the final particles
    radii = [_gen_radius()]
    activeRadii = [radii[0]]

    # Continue iteration while there is still points in active list
    while queue:

        # Pick random element in active queue
        idx = np.random.randint(len(queue))
        r_active = activeRadii[idx]
        activeCoords = queue[idx]
        activeGridCoords = get_grid_coordinates(activeCoords)

        success = False
        for _ in range(k):
            if radiusType == "default":
                # Pick radius for new sample particle ranging between 1 and 2 times the local radius
                newRadius = r_active * (np.random.random() + 1)
            elif radiusType == "normDist":
                # Pick radius for new sample particle from a normal distribution around 1.5 times the local radius
                newRadius = r_active * np.random.normal(1.5, 0.2)

            # Pick the angle to the sample particle and determine its coordinates
            angle = 2 * np.pi * np.random.random()
            newCoords = np.zeros(2)
            newCoords[0] = activeCoords[0] + newRadius * np.sin(angle)
            newCoords[1] = activeCoords[1] + newRadius * np.cos(angle)

            # Prevent that the new particle is outside of the grid
            if not (0 <= newCoords[1] <= gridWidth and 0 <= newCoords[0] <= gridHeight):
                continue

            # Check that particle is not too close to other particle
            newGridCoords = get_grid_coordinates((newCoords[1], newCoords[0]))

            radiusThere = (
                _gen_radius()
            )  # np.ceil(radius[newGridCoords[1], newGridCoords[0]])
            gridRangeX = (
                np.max([newGridCoords[0] - radiusThere, 0]).astype("int"),
                np.min([newGridCoords[0] + radiusThere + 1, gridWidth]).astype("int"),
            )
            gridRangeY = (
                np.max([newGridCoords[1] - radiusThere, 0]).astype("int"),
                np.min([newGridCoords[1] + radiusThere + 1, gridHeight]).astype("int"),
            )

            searchGrid = grid[
                slice(gridRangeY[0], gridRangeY[1]), slice(gridRangeX[0], gridRangeX[1])
            ]
            conflicts = np.where(searchGrid > 0)
      #      conflicts=[[],[]]

            if len(conflicts[0]) == 0 and len(conflicts[1]) == 0:
                # No conflicts detected. Create a new particle at this position!
                queue.append(newCoords)
                activeRadii.append(radiusThere)
                radii.append(radiusThere)
                particleCoordinates.append(newCoords)
                nParticle += 1
                grid[newGridCoords[1], newGridCoords[0]] = nParticle
                success = True

            else:
                # There is a conflict. Do NOT create a new particle at this position!
                continue

        if success == False:
            # No new particle could be associated to the currently active particle.
            # Remove current particle from the active queue!
            del queue[idx]
            del activeRadii[idx]

    return (nParticle, np.array(particleCoordinates), np.array(radii))


def make_gaussian_2d(x_center=0, y_center=0, theta=0, sigma_x=10, sigma_y=10):
    # x_center and y_center will be the center of the gaussian, theta will be the rotation angle
    # sigma_x and sigma_y will be the stdevs in the x and y axis before rotation
    # x_size and y_size give the size of the frame
    sx = sigma_x
    sy = sigma_y
    x0 = x_center
    y0 = y_center

    theta = theta * np.pi / 180

    def fn(x, y):
        a = np.cos(theta) * x - np.sin(theta) * y
        b = np.sin(theta) * x + np.cos(theta) * y
        a0 = np.cos(theta) * x0 - np.sin(theta) * y0
        b0 = np.sin(theta) * x0 + np.cos(theta) * y0

        return np.exp(
            -(((a - a0) ** 2) / (2 * (sx ** 2)) + ((b - b0) ** 2) / (2 * (sy ** 2)))
        )

    return fn


def make_gaussian_2d_area(x_center=0, y_center=0, theta=0, sigma_x=10, sigma_y=10):
    # x_center and y_center will be the center of the gaussian, theta will be the rotation angle
    # sigma_x and sigma_y will be the stdevs in the x and y axis before rotation
    # x_size and y_size give the size of the frame
    sx = sigma_x
    sy = sigma_y
    x0 = x_center
    y0 = y_center

    theta = theta * np.pi / 180

    def fn(x, y):
        a = np.cos(theta) * x - np.sin(theta) * y
        b = np.sin(theta) * x + np.cos(theta) * y
        a0 = np.cos(theta) * x0 - np.sin(theta) * y0
        b0 = np.sin(theta) * x0 + np.cos(theta) * y0

        return ((a - a0) ** 2) / (2 * (sx ** 2)) + ((b - b0) ** 2) / (2 * (sy ** 2))
        # return a, a0, sx, b, b0, sy

    return fn


def make_carrier_wave_2d(theta, lw):
    theta = theta * np.pi / 180

    def fn(x, y):
        x_ = np.cos(theta) * x - np.sin(theta) * y
        return np.cos(2 * np.pi / lw * x_)

    return fn



def make_2d_wavepacket(x0, y0, lx, ly, theta, lw):
    """
    (x0, y0): position of wavepacket
    (lx, ly): x- and y- length-scale of wavepacket
    theta: orientation [deg]
    lw: length-scale of gravity "carrier wave"
    """
    fn_envelope = make_gaussian_2d(
        x_center=x0, y_center=y0, theta=theta, sigma_x=lx / 2, sigma_y=ly / 2
    )
    fn_carrier = make_carrier_wave_2d(theta=theta, lw=lw)
    

    def fn(x, y):
        return fn_envelope(x, y) * fn_carrier(x, y)

    return fn


def make_2d_wavepacket_area(x0, y0, lx, ly, theta, lw):
    """
    (x0, y0): position of wavepacket
    (lx, ly): x- and y- length-scale of wavepacket
    theta: orientation [deg]
    lw: length-scale of gravity "carrier wave"
    """
    fn_envelope = make_gaussian_2d(
        x_center=x0, y_center=y0, theta=theta, sigma_x=lx / 2, sigma_y=ly / 2
    )
    # fn_carrier = make_carrier_wave_2d(theta=theta, lw=lw)

    def fn(x, y):
        return fn_envelope(x, y)

    return fn


for k in range(no_of_examples):
    N = 512
    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")

    r0 = 100
    n_pts, pts, radii = modified_poisson_disk_sampling(N=N, r0=r0, r_sigma=r_sigma)

    angle_envelope = np.zeros((N, N))
    amp_envelope = np.zeros((N,N))
    wavelength_envelope = np.zeros((N, N))
    phi = np.zeros((N, N))
    phi_envelope = np.zeros((N, N))
    donut_envelope = np.zeros((N,N))
    angle_threshold=0.1

    for pt, r in zip(pts, radii):
        theta = 180 * np.random.uniform()
        lw = 3.5 * np.random.chisquare(2) + 5
        a = 0.5 + 0.5 * np.random.uniform()

        lx = r / 2
        ly = a * r / 2
        x0, y0 = pt

        fn_wavepacket = make_2d_wavepacket(x0=x0, y0=y0, lx=lx, ly=ly, theta=theta, lw=lw)
        fn_wavepacket_area = make_2d_wavepacket_area(
          x0=x0, y0=y0, lx=lx, ly=ly, theta=theta, lw=lw
        )
        fn_envelope = make_gaussian_2d(
          x_center=x0, y_center=y0, theta=theta, sigma_x=lx / 2, sigma_y=ly / 2
       )
        angle = fn_wavepacket_area(x, y)
        donut = np.logical_and(angle < (1), angle > (angle_threshold))
      
        #for variable amplitude, set randn to something. For the paper, we used a
        #value drawn from a uniform distribution between 1 and 5.
        #For wavelength and orientation, we left amplitude as 1.
        
        #randn = np.random.uniform(1,5)
        randn = 1
        
        
        
        phi += fn_wavepacket(x, y)*randn
        phi_envelope += fn_envelope(x, y)
        noise = np.random.normal(size=(512,512))
        phi = phi
        angle_envelope += donut * theta
        donut_envelope += donut
        amp_envelope +=fn_envelope(x,y)*randn
        wavelength_envelope += donut * 2000 * np.abs(lw)

    np.save(root+'data/'+str(k)+'.npy',phi)
    np.save(root+'orientation/'+str(k)+'.npy',(angle_envelope-90)*donut_envelope)
    np.save(root+'wavelength/'+str(k)+'.npy',wavelength_envelope)
    np.save(root+'amplitude/'+str(k)+'.npy',amp_envelope)