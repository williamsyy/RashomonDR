"""
Helper functions for Autoencoders.
"""

import torch
import numpy as np

from parampacmap.diffgeo.manifolds import RiemannianManifold
from parampacmap.diffgeo.metrics import PullbackMetric
from parampacmap.diffgeo.connections import LeviCivitaConnection


class Loss:
    """
    A Basis class for custom loss functions
    """
    def __init__(self, model=None, device=torch.device("cuda")):
        # a manifold object
        self.model = model
        pbm = PullbackMetric(2, model.immersion)
        lcc = LeviCivitaConnection(2, pbm)
        self.manifold = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)
        self.device = device

    @staticmethod
    def sample_points(latent_activations=None, outputs=None, num_samples=1):
        """
        Randomly sample points from batch in latent space and map it to the output space
        :param outputs: batch in output space
        :param latent_activations: batch in latent space
        :param num_samples: number of samples to take
        :return: (points in latent space, points in output space)
        """

        # randomly sample the points
        rand_choice = torch.randperm(latent_activations.shape[0])[:num_samples]

        pimg_origin = latent_activations[rand_choice, :]

        if outputs is None:
            return pimg_origin

        img_origin = outputs[rand_choice, :]

        return img_origin, pimg_origin


class DeterminantLoss(Loss):
    def __call__(self, epoch=0, *args, **kwargs):
        """
        GeoAE Determinant Regularizer.
        Args:
            epoch: current epoch
        Returns:

        """

        # here you can control whether the regularizer should only be switched on after a certain epoch
        if epoch >= 0:
            loss_det = self.determinant_loss()
        else:
            loss_det = torch.tensor([0.], device=self.device, requires_grad=True)

        return loss_det

    def determinant_loss(self):
        """
        Calculate the actual loss
        Returns:
            The determinant loss
        """

        # calculate the logarithm of the generalized jacobian determinant
        log_dets = self.manifold.metric_logdet(base_point=self.model.latent_activations)

        # replace nan values with a small number
        #EPSILON = 1e-9
        #torch.nan_to_num(log_dets, nan=EPSILON, posinf=EPSILON, neginf=EPSILON)
        torch.nan_to_num(log_dets, nan=1., posinf=1., neginf=1.)

        # calculate the variance of the logarithm of the generalized jacobian determinant
        raw_loss = torch.var(log_dets)

        return raw_loss




class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, n_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''

        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistentHomologyCalculation:
    def __call__(self, matrix):

        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')

        # 1st dimension: 'source' vertex index of edge
        # 2nd dimension: 'target' vertex index of edge
        persistence_pairs = []

        for edge_index, edge_weight in \
                zip(edge_indices, edge_weights[edge_indices]):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Not an edge of the MST, so skip it
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            if u < v:
                persistence_pairs.append((u, v))
            else:
                persistence_pairs.append((v, u))

        # Return empty cycles component
        return np.array(persistence_pairs), np.array([])

