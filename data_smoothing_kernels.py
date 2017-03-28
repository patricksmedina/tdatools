import scipy.spatial.kdtree as kd
import matplotlib.pyplot as plt
import numpy as np

class _spatial_smoother_2d(object):
    """
    Stores domain parameters for all spatially adaptive kernel density estimators.

    Arguments:

        1. num_neighbors: The number of neighbors to use for the adaptive smoothing parameter.
        2. domx_params: A tuple specifying the parameters for the x-axis. Elements containg lower bound; upper bound; number of steps.
        3. domy_params: A tuple specifying the parameters for the y-axis. Elements containg lower bound; upper bound; number of steps.

    """
    def __init__(self,
                 num_neighbors,
                 domx_params,
                 domy_params):

        self.k = num_neighbors
        self.domx_params = domx_params
        self.domy_params = domy_params

    def _construct_domain(self):
        # construct domain over x-axis
        domx = np.linspace( self.domx_params[0],
                            self.domx_params[1],
                            self.domx_params[2])
        # domain over the y-axis
        domy = np.linspace( self.domy_params[0],
                            self.domy_params[1],
                            self.domy_params[2])

        # return grid coordinates
        return(np.meshgrid(domx, domy))

    def plot_kernel(self,
                    f,
                    x = None,
                    title = "",
                    data_alpha = 1.0,
                    filename = None):

        """
        Plotting tools for spatially adaptive kernel smoothing functions.

        Arguments:

            1. f: Matrix containing the kernel smoothed function.
            2. x: Data used by the kernel smoothing the function. (optional)
            3. title: Plot title. (optional)

        Returns:

            1. Plots kernel smoothed data.

        """
        domx, domy = self._construct_domain()

        plt.figure()
        ctf = plt.contourf(domx, domy, f, cmap = "Blues", levels = np.linspace(np.min(f), np.max(f), 31))
        plt.colorbar(ctf, shrink=0.9, format = '%.01e')
        plt.title(title)

        # plot the data if provided
        if x is not None:
            plt.scatter(x[:, 0], x[:, 1], facecolor = 'red', edgecolor = "black", zorder = 1, alpha = data_alpha)

        if filename is not None:
            plt.savefig(filename, dpi = 300)
        else:
            plt.show()


class knn_gaussian_2d(_spatial_smoother_2d):

    def __init__(self,          \
                 num_neighbors, \
                 domx_params,   \
                 domy_params):

        """
        Parameters and functions for k nearest neighbor adaptive density estimator
        with a Gaussian kernel.

        Arguments:

            1. domx_params: A tuple specifying the parameters for the x-axis. Elements containg lower bound; upper bound; number of steps.
            2. domy_params: A tuple specifying the parameters for the y-axis. Elements containg lower bound; upper bound; number of steps.

        Example:

            domx = (0, 10, 100)     # start at 0; end at 10; 100 steps
            domy = (-10, 10, 100)   # start at -10; end at 10; 100 steps

            # initialize smoother with using 10 neighbors
            gaussian_smoother = knn_gaussian_2d(num_neighbors = 10, domx_params = domx, domy_params = domy)
        """

        _spatial_smoother_2d.__init__(self, num_neighbors, domx_params, domy_params)

    def smooth(self, x):
        """
        Apply K Nearest Neighbors adaptive Gaussian smoother to a provided 2D dataset.

        Arguments:

            1. x: 2D dataset to be smoothed. It is assumed that the rows of
                  the data matrix are the sample points.

        Returns:

            1. Smoothed function over the specified domain.

        Example:

            TODO: Write sample code...
        """
        domx, domy = self._construct_domain()
        dom = np.vstack((domx.ravel(), domy.ravel())).T

        # construct KD tree on data
        tree = kd.KDTree(x)

        # get k nearest neighbors
        dist = tree.query(dom, k = self.k)[0]
        tp_knn = dist[:, self.k - 1].reshape(-1, 1)

        ### ADAPTIVE KERNEL SMOOTHING
        # pairwise subtraction between grid points and each data point
        # reshape from tensor to matrix (K x 2)
        Fxy = np.subtract(dom[:, np.newaxis, :], x[np.newaxis, :, :]).reshape(-1, 2)
        Fxy = np.square(np.linalg.norm(Fxy, axis = 1)).reshape(dom.shape[0],-1)
        Fxy = np.divide(Fxy, -2 * tp_knn ** 2)
        Fxy = np.divide(np.exp(Fxy), 2 * np.pi * tp_knn ** 2)
        Fxy = Fxy.mean(axis = 1)

        return(Fxy.reshape(self.domy_params[2], self.domx_params[2]))

class knn_density_estimator(_spatial_smoother_2d):

    def __init__(self, num_neighbors, domx_params, domy_params):

        """
        Parameters and functions for k-nearest neighbor adaptive density estimator.

        Arguments:

            1. num_neighbors:   The number of neighbors (k) used to estimate the density.
            1. domx_params:     A tuple specifying the parameters for the x-axis. Elements containg lower bound; upper bound; number of steps.
            2. domy_params:     A tuple specifying the parameters for the y-axis. Elements containg lower bound; upper bound; number of steps.

        Example:

            k = 10                  # Number of neighbors to consider.
            domx = (0, 10, 100)     # start at 0; end at 10; 100 steps.
            domy = (-10, 10, 100)   # start at -10; end at 10; 100 steps.

            # initialize smoother with using 10 neighbors
            gaussian_smoother = knn_gaussian_2d(num_neighbors = 10, domx_params = domx, domy_params = domy)
        """

        _spatial_smoother_2d.__init__(self, num_neighbors, domx_params, domy_params)

    def smooth(self, x):
        """
        Apply K Nearest Neighbors density estimator over a grid.

        Arguments:

            1. x: 2D dataset to be smoothed. It is assumed that the rows of
                  the data matrix are the sample points.

        Returns:

            1. Smoothed function over the specified domain.

        Example:

            TODO: Write sample code...
        """
        domx, domy = self._construct_domain()
        dom = np.vstack((domx.ravel(), domy.ravel())).T

        # construct KD tree on data
        tree = kd.KDTree(x)

        # get k^{th} nearest neighbors to each point in the domain
        dist = tree.query(dom, k = self.k, p = 2)[0]
        dist_knn = dist[:, self.k - 1].reshape(self.domy_params[2], self.domx_params[2])
        dist_knn = np.divide(self.k / (x.shape[0] * np.pi), dist_knn ** 2)

        # KNN density estimator
        return(dist_knn)

class distance_to_measure(_spatial_smoother_2d):
    def __init__(self, domx_params, domy_params, num_neighbors = None, tau = None):
        """Description here"""
        assert 0 < tau < 1 or tau is None, \
               "Parameter tau must be a numerical value in (0, 1)."

        assert len(domx_params) == 3 and len(domy_params) == 3, \
               "Domain parameter tuples must contain three elements."

        assert isinstance(domx_params, tuple) and isinstance(domy_params, tuple), \
                "Domain parameters must be of type 'tuple'."


        if tau is None:
            _spatial_smoother_2d.__init__(self,
                                          num_neighbors = num_neighbors,
                                          domx_params = domx_params,
                                          domy_params = domy_params)
        else:
            _spatial_smoother_2d.__init__(self,
                                          num_neighbors = None,
                                          domx_params = domx_params,
                                          domy_params = domy_params)

        self.tau = tau

    def smooth(self, x):
        k = np.ceil(self.tau * x.shape[0]) if self.tau is not None else self.k
        k = int(k)
        domx, domy = self._construct_domain()
        dom = np.vstack((domx.ravel(), domy.ravel())).T

        tree = kd.KDTree(x)
        knn = tree.query(dom, k = k, p = 2)[1]

        #
        Fxy = np.subtract(dom[:, np.newaxis, :], x[knn, :]).reshape(-1, 2)
        Fxy = np.linalg.norm(Fxy, axis = 1).reshape(-1, k)
        Fxy = np.sqrt(Fxy.mean(axis = 1))

        return(Fxy.reshape(self.domy_params[2], self.domx_params[2]))




# class knn_uniform_2d(_spatial_smoother_2d):
#     """
#     Parameters and functions for k nearest neighbor adaptive density estimator
#     with a uniform kernel.
#     Arguments:
#         1. domx_params: A tuple specifying the parameters for the x-axis. Elements containg lower bound; upper bound; number of steps.
#         2. domy_params: A tuple specifying the parameters for the y-axis. Elements containg lower bound; upper bound; number of steps.
#     Example:
#         TODO: Write example codeblock
#     """
#
#     def __init__(self, num_neighbors, domx_params, domy_params):
#         _spatial_smoother_2d.__init__(self, num_neighbors, domx_params, domy_params)
#
#     def smooth(self, x):
#         """
#         Apply K Nearest Neighbors adaptive smoother with a uniform kernel to a
#         given 2D dataset.
#         Arguments:
#             1. x: 2D dataset to be smoothed.  It is assumed that the rows of
#                   the data matrix are the sample points.
#         Returns:
#             1. Smoothed function over the specified domain.
#         Example:
#             TODO: Write sample code...
#         """
#
#         domx, domy = self._construct_domain()
#         dom = np.vstack((domx.ravel(), domy.ravel())).T
#
#         # construct KD tree on data
#         tree = kd.KDTree(x)
#
#         # get k+1 nearest neighbors
#         #! the point itself is always the first nearest neighbor
#         dist = tree.query(x, k = self.k)[0]
#         tp_knn = dist[:, self.k - 1].reshape(-1, 1)
#         tp_knn = np.hstack([tp_knn, tp_knn])
#
#         # ADAPTIVE KERNEL SMOOTHING
#         # pairwise subtraction between each grid point and each point in the data
#         Fxy = np.subtract(dom[:, np.newaxis, :], x[np.newaxis, :, :])

        ## divide each data point by its relative weight
        #tp_knn_big = np.tile(tp_knn, (dom.shape[0], 1)).reshape(dom.shape[0], -1, 2)
        #Fxy = np.divide(Fxy, tp_knn_big)

        ## find where values satisfy kernel condition
        #Fxy = (np.abs(Fxy) < 1.0) * 1.0

        ## get columns where both x, y coordinates are satisfied
        #Fxy = np.prod(Fxy, axis = 2)
        #Fxy = np.divide(Fxy.T, 4.0 * np.prod(tp_knn, 1).reshape(-1,1)).T
        #Fxy = np.mean(Fxy, axis = 1)

        #return(Fxy.reshape(domx.shape[0], domy.shape[0]))

if __name__ == "__main__":
    """For testing and illustrative purposes only"""
    import tdaw.examples.annulus_data as ad

    x = ad.sample_paired_annuli(R1 = 60,
                                r1 = 40,
                                R2 = 40,
                                r2 = 20,
                                center_modifier = 50,
                                samples_from_shape = 500)

    domx_params = (np.min(x[:, 0]) - 10, np.max(x[:, 0]) + 10, 100)
    domy_params = (np.min(x[:, 1]) - 10, np.max(x[:, 1]) + 10, 100)

    dtm = distance_to_measure( tau = 0.10,
                               domx_params = domx_params,
                               domy_params = domy_params)
    dtm.plot_kernel(f = dtm.smooth(x), x = x)

    #gaussian_de = knn_gaussian_2d(num_neighbors = 8,
    #                              domx_params = ,
    #                              domy_params = )

    #gaussian_de.plot_kernel(f = gaussian_de.smooth(x), x = x, title = "{} KNN with Gaussian Kernel".format(gaussian_de.k))

    #from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #X, Y = dtm._construct_domain()
    #surf = ax.plot_surface(X, Y, dtm.smooth(x), cmap = "Blues", linewidth=0, antialiased=False)
    #fig.colorbar(surf, shrink=0.25, aspect=5)
    #plt.show()

    # knn_de = knn_density_2d(num_neighbors = 8,
    #                         domx_params = (np.min(x[:, 0]) - 10, np.max(x[:, 0]) + 10, 50),
    #                         domy_params = (np.min(x[:, 1]) - 10, np.max(x[:, 1]) + 10, 100))

    #print knn_de.smooth(x).shape
    #knn_de.plot_kernel(f = knn_de.smooth(x), x = x, title = "{} KNN Density Estimator".format(knn_de.k))

    #adaptive_knn.plot_kernel(f = adaptive_knn.smooth(x), x = x, title = "{} KNN with Gaussian Kernel".format(adaptive_knn.k))
