#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2021 Gabriele Orlando
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import torch, math
import pickle

PADDING_INDEX = 999.0
NONEXISTING_HASH = -1
MISSING_ATOM = -999
NON_ACCEPTABLE_ENERGY = 999
NON_ACCEPTABLE_DISTANCE = 999
MISSING_INDEX = -1

EPS = 0.00001


class Voxels(torch.nn.Module):
    def __init__(self, device=torch.device("cpu"), sparse=True):
        """
        Constructor for the Voxels class, which builds the main PyUUL object.

        Parameters
        ----------

        device : torch.device
            The device on which the model should run. E.g. torch.device("cuda") or torch.device("cpu:0")
        sparse : bool
            Use sparse tensors calculation when possible

        Returns
        -------
        """
        super(Voxels, self).__init__()

        self.sparse = sparse
        self.boxsize = None
        self.dev = device

    def __transform_coordinates(self, coords, radius=None):
        """
        Private function that transform the coordinates to fit them in the 3d box.
        It also takes care of the resolution.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates of the atoms. Shape ( batch, numberOfAtoms, 3 )
        radius : torch.Tensor or None
            Radius of the atoms. Shape ( batch, numberOfAtoms )

        Returns
        -------
        coords : torch.Tensor
            transformed coordinates

        """
        coords = (coords * self.dilatation) - self.translation
        if not radius is None:
            radius = radius * self.dilatation
            return coords, radius
        else:
            return coords

    '''
    def get_coords_voxel(self, voxel_indices, resolution):
        """
        returns the coordinates of the center of the voxel provided its indices.

        Parameters
        ----------
        voxel_indices : torch.Tensor
            Coordinates of the atoms. Shape ( batch, numberOfAtoms, 3 ) 
        resolution : torch.Tensor or None
            Radius of the atoms. Shape ( batch, numberOfAtoms )

        Returns
        -------
        """
        #voxel_indices is a n,3 long tensor
        centersCoords = voxel_indices + 0.5*resolution
        return (centersCoords + self.translation)/self.dilatation
    '''

    def __define_spatial_conformation(
        self, mincoords, cubes_around_atoms_dim, resolution
    ):
        """
        Private function that defines the space of the volume. Takes resolution and margins into consideration.

        Parameters
        ----------
        mincoords : torch.Tensor
            minimum coordinates of each macromolecule of the batch. Shape ( batch, 3 )
        cubes_around_atoms_dim : int
            maximum distance in number of voxels to check for atom contribution to occupancy of a voxel
        resolution : float
            side in A of a voxel. The lower this value is the higher the resolution of the final representation will be
        Returns
        -------
        """
        self.translation = (mincoords - (cubes_around_atoms_dim)).unsqueeze(1)
        self.dilatation = 1.0 / resolution

    """
    def find_cubes_indices(self,coords):
        coords_scaled = self.transform_coordinates(coords)
        return torch.trunc(coords_scaled.data).long()
    """

    def forward(
        self,
        coords,
        radius,
        channels,
        numberchannels=None,
        resolution=1,
        cubes_around_atoms_dim=5,
        steepness=10,
        function="gaussian",
    ):
        """
        Voxels representation of the macromolecules

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates of the atoms. Shape ( batch, numberOfAtoms, 3 ). Can be calculated from a PDB file using utils.parsePDB
        radius : torch.Tensor
            Radius of the atoms. Shape ( batch, numberOfAtoms ). Can be calculated from a PDB file using utils.parsePDB and utils.atomlistToRadius
        channels: torch.LongTensor
            channels of the atoms. Atoms of the same type shold belong to the same channel. Shape ( batch, numberOfAtoms ). Can be calculated from a PDB file using utils.parsePDB and utils.atomlistToChannels
        numberchannels : int or None
            maximum number of channels. if None, max(atNameHashing) + 1 is used

        cubes_around_atoms_dim : int
            maximum distance in number of voxels for which the contribution to occupancy is taken into consideration. Every atom that is farer than cubes_around_atoms_dim voxels from the center of a voxel does no give any contribution to the relative voxel occupancy
        resolution : float
            side in A of a voxel. The lower this value is the higher the resolution of the final representation will be

        steepness : float or int
            steepness of the sigmoid occupancy function.

        function : "sigmoid" or "gaussian"
            occupancy function to use. Can be sigmoid (every atom has a sigmoid shaped occupancy function) or gaussian (based on Li et al. 2014)
        Returns
        -------
        volume : torch.Tensor
            voxel representation of the macromolecules in the batch. Shape ( batch, channels, x,y,z), where x,y,z are the size of the 3D volume in which the macromolecules have been represented

        """
        padding_mask = ~channels.eq(PADDING_INDEX)
        if numberchannels is None:
            numberchannels = int(channels[padding_mask].max().cpu().data + 1)
        self.featureVectorSize = numberchannels
        self.function = function

        arange_type = torch.int16

        cell_row = torch.arange(
            -cubes_around_atoms_dim,
            cubes_around_atoms_dim + 1,
            device=self.dev,
            dtype=arange_type,
        )
        self.lato = cell_row.shape[0]
        self.standard_cube = mycube = torch.cartesian_prod(
            cell_row, cell_row, cell_row
        ).reshape(11, 11, 11, 3)

        ### definition of the box ###
        # you take the maximum and min coord on each dimension (every prot in the
        # batch shares the same box. In the future we can pack, but I think this is not the bottleneck)
        # I scale by resolution
        # I add the cubes in which I define the gradient. One in the beginning and one at the end --> 2*

        maxcoords = torch.max(coords[padding_mask], dim=0)[0] / resolution
        self.mincoords = torch.min(coords[:, :, :], dim=1)[0] / resolution
        pad = 2 * cubes_around_atoms_dim + 1
        boxdims = maxcoords - self.mincoords + pad
        self.boxsize = tuple(torch.squeeze(boxdims.to(int), 0).tolist())
        #############################

        self.__define_spatial_conformation(
            self.mincoords, cubes_around_atoms_dim, resolution
        )  # define the spatial transforms to coordinates
        self.coords_scaled, radius = self.__transform_coordinates(coords, radius)

        # selecting best types for indexing
        if max(self.boxsize) < 256:  # i can use byte tensor
            self.dtype_indices = torch.uint8
        else:
            self.dtype_indices = torch.int16

        if self.function == "sigmoid":
            volume = self.__forward_actual_calculation(
                self.coords_scaled,
                self.boxsize,
                radius,
                channels,
                padding_mask,
                steepness,
                resolution,
            )
        elif self.function == "gaussian":
            volume = self.__forward_actual_calculationGaussian(
                self.coords_scaled,
                self.boxsize,
                radius,
                channels,
                padding_mask,
                resolution,
            )
        return volume

    def __forward_actual_calculationGaussian(
        self, coords_scaled, boxsize, radius, atNameHashing, padding_mask, resolution
    ):
        """
        private function for the calculation of the gaussian voxel occupancy

        Parameters
        ----------
        coords_scaled : torch.LongTensor
            Discrete Coordinates of the atoms. Shape ( batch, numberOfAtoms, 3 )
        boxsize : torch.LongTensor
            The size of the box in which the macromolecules are represented
        radius : torch.Tensor
            Radius of the atoms. Shape ( batch, numberOfAtoms ). Can be calculated from a PDB file using utils.parsePDB and utils.atomlistToRadius
        atNameHashing: torch.LongTensor
            channels of the atoms. Atoms of the same type shold belong to the same channel. Shape ( batch, numberOfAtoms ). Can be calculated from a PDB file using utils.parsePDB and utils.atomlistToChannels
        resolution : float
            side in A of a voxel. The lower this value is the higher the resolution of the final representation will be
        padding_mask : torch.BoolTensor
            tensor to mask the padding. Shape (batch, numberOfAtoms)
        Returns
        -------
        volume : torch.Tensor
            voxel representation of the macromolecules in the batch with Gaussian occupancy function. Shape ( batch, channels, x,y,z), where x,y,z are the size of the 3D volume in which the macromolecules have been represented

        """
        batch = coords_scaled.shape[0]
        dev = self.dev
        L = coords_scaled.shape[1]

        self.discrete_coordinates = torch.trunc(coords_scaled.data).to(
            self.dtype_indices
        )

        #### making everything in the volume shape

        # implicit_cube_formation
        radius = radius.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        atNameHashing = atNameHashing.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        coords_scaled = coords_scaled.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        self.discrete_coordinates = self.discrete_coordinates[:, :, None, None, None, :]
        # distmat_standard_cube has a cube for each atom.
        # The sides of the cube are of length = 2*cube_around_atoms_dim + 1
        # Each element of the cube represents the distance between the center of the cube
        # and its corresponding atom.
        self.distmat_standard_cube = torch.norm(
            coords_scaled
            - ((self.discrete_coordinates + self.standard_cube + 1) + 0.5 * resolution),
            dim=-1,
        ).to(coords_scaled.dtype)

        atNameHashing = atNameHashing.long()

        ### from doi: 10.1142/S0219633614400021 eq 1
        sigma = 0.93
        exponent = -(self.distmat_standard_cube[padding_mask] ** 2) / (
            sigma**2 * radius[padding_mask] ** 2
        )
        exp_mask = exponent.ge(10)
        exponent = torch.masked_fill(exponent, exp_mask, 10)
        # `volume_cubes` has a cube for each atom, made of ncubes^3 voxels, where:
        # ncubes = 2 * cubes_around_atoms_dim + 1
        # Each cube represents the gaussian density filled by its corresponding atom
        volume_cubes = torch.exp(exponent)

        #### index_put everything ###
        batch_list = (
            torch.arange(batch, device=dev)
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(batch, L, self.lato, self.lato, self.lato)
        )

        self.cubes_coords = (
            self.discrete_coordinates[padding_mask] + self.standard_cube.squeeze(0) + 1
        )[~exp_mask]
        atNameHashing = atNameHashing[padding_mask].expand(
            -1, self.lato, self.lato, self.lato
        )
        if self.sparse:
            index_tens = torch.cat(
                [
                    batch_list[padding_mask][~exp_mask].view(-1).unsqueeze(0),
                    atNameHashing[~exp_mask].unsqueeze(0),
                    self.cubes_coords[:, 0].unsqueeze(0),
                    self.cubes_coords[:, 1].unsqueeze(0),
                    self.cubes_coords[:, 2].unsqueeze(0),
                ]
            )
            volume_cubes = volume_cubes[~exp_mask].view(-1)
            volume_cubes = torch.log(1 - volume_cubes.contiguous())
            # powOrExpIsNotImplementedInSparse
            volume = torch.sparse_coo_tensor(
                indices=index_tens,
                values=volume_cubes.exp(),
                size=[
                    batch,
                    self.featureVectorSize,
                    boxsize[0],
                    boxsize[1],
                    boxsize[2],
                ],
            ).coalesce()
            volume = torch.sparse_coo_tensor(
                volume.indices(), 1 - volume.values(), volume.shape
            )

        else:
            volume = torch.zeros(
                batch,
                boxsize[0] + 1,
                boxsize[1] + 1,
                boxsize[2] + 1,
                self.featureVectorSize,
                device=dev,
                dtype=torch.float,
            )
            # index = (batch_list[padding_mask].view(-1),cubes_coords[padding_mask][:,:,:,:,0].view(-1), cubes_coords[padding_mask][:,:,:,:,1].view(-1), cubes_coords[padding_mask][:,:,:,:,2].view(-1), atNameHashing[padding_mask].view(-1) )
            index = (
                batch_list[padding_mask][~exp_mask].view(-1).long(),
                self.cubes_coords[:, 0].long(),
                self.cubes_coords[:, 1].long(),
                self.cubes_coords[:, 2].long(),
                atNameHashing[~exp_mask],
            )
            volume_cubes = volume_cubes[~exp_mask].view(-1)

            with open("index.pkl", "wb") as f:
                pickle.dump(index, f)

            volume_cubes = torch.log(1 - volume_cubes.contiguous())
            volume = 1 - torch.exp(
                volume.index_put(index, volume_cubes, accumulate=True)
            )
            volume = volume.permute(0, 4, 1, 2, 3)

        return volume

    def __sparseClamp(self, volume, minv, maxv):
        vals = volume.values()
        ind = volume.indices()

        vals = vals.clamp(minv, maxv)
        volume = torch.sparse_coo_tensor(
            indices=ind, values=vals, size=volume.shape
        ).coalesce()
        return volume

    def __forward_actual_calculation(
        self,
        coords_scaled,
        boxsize,
        radius,
        atNameHashing,
        padding_mask,
        steepness,
        resolution,
    ):
        """
        private function for the calculation of the gaussian voxel occupancy

        Parameters
        ----------
        coords_scaled : torch.LongTensor
            Discrete Coordinates of the atoms. Shape ( batch, numberOfAtoms, 3 )
        boxsize : torch.LongTensor
            The size of the box in which the macromolecules are represented
        radius : torch.Tensor
            Radius of the atoms. Shape ( batch, numberOfAtoms ). Can be calculated from a PDB file using utils.parsePDB and utils.atomlistToRadius
        atNameHashing: torch.LongTensor
            channels of the atoms. Atoms of the same type shold belong to the same channel. Shape ( batch, numberOfAtoms ). Can be calculated from a PDB file using utils.parsePDB and utils.atomlistToChannels
        resolution : float
            side in A of a voxel. The lower this value is the higher the resolution of the final representation will be
        padding_mask : torch.BoolTensor
            tensor to mask the padding. Shape (batch, numberOfAtoms)
        steepness : float
            steepness of the sigmoid function (coefficient of the exponent)

        Returns
        -------
        volume : torch.Tensor
            voxel representation of the macromolecules in the batch with Sigmoid occupancy function. Shape ( batch, channels, x,y,z), where x,y,z are the size of the 3D volume in which the macromolecules have been represented

        """
        batch = coords_scaled.shape[0]
        dev = self.dev
        L = coords_scaled.shape[1]

        discrete_coordinates = torch.trunc(coords_scaled.data).to(self.dtype_indices)

        #### making everything in the volume shape

        # implicit_cube_formation
        radius = radius.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        atNameHashing = atNameHashing.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        coords_scaled = coords_scaled.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        discrete_coordinates = (
            discrete_coordinates.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        )
        distmat_standard_cube = torch.norm(
            coords_scaled
            - ((discrete_coordinates + self.standard_cube + 1) + 0.5 * resolution),
            dim=-1,
        ).to(coords_scaled.dtype)

        atNameHashing = atNameHashing.long()

        exponent = steepness * (
            distmat_standard_cube[padding_mask] - radius[padding_mask]
        )
        del distmat_standard_cube
        exp_mask = exponent.ge(10)
        exponent = torch.masked_fill(exponent, exp_mask, 10)

        volume_cubes = 1.0 / (1.0 + torch.exp(exponent))

        #### index_put everything ###
        batch_list = (
            torch.arange(batch, device=dev)
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(batch, L, self.lato, self.lato, self.lato)
        )

        # cubes_coords = coords_scaled + self.standard_cube + 1
        cubes_coords = (
            discrete_coordinates[padding_mask] + self.standard_cube.squeeze(0) + 1
        )[~exp_mask]
        atNameHashing = atNameHashing[padding_mask].expand(
            -1, self.lato, self.lato, self.lato
        )
        if self.sparse:
            index_tens = torch.cat(
                [
                    batch_list[padding_mask][~exp_mask].view(-1).unsqueeze(0),
                    atNameHashing[~exp_mask].unsqueeze(0),
                    cubes_coords[:, 0].unsqueeze(0),
                    cubes_coords[:, 1].unsqueeze(0),
                    cubes_coords[:, 2].unsqueeze(0),
                ]
            )
            # index_tens = torch.cat(index)
            volume = torch.sparse_coo_tensor(
                indices=index_tens,
                values=volume_cubes[~exp_mask].view(-1),
                size=[
                    batch,
                    self.featureVectorSize,
                    boxsize[0],
                    boxsize[1],
                    boxsize[2],
                ],
            ).coalesce()
            volume = self.__sparseClamp(volume, 0, 1)

        else:
            volume = torch.zeros(
                batch,
                boxsize[0] + 1,
                boxsize[1] + 1,
                boxsize[2] + 1,
                self.featureVectorSize,
                device=dev,
                dtype=torch.float,
            )
            # index = (batch_list[padding_mask].view(-1),cubes_coords[padding_mask][:,:,:,:,0].view(-1), cubes_coords[padding_mask][:,:,:,:,1].view(-1), cubes_coords[padding_mask][:,:,:,:,2].view(-1), atNameHashing[padding_mask].view(-1) )
            index = (
                batch_list[padding_mask][~exp_mask].view(-1).long(),
                cubes_coords[:, 0].long(),
                cubes_coords[:, 1].long(),
                cubes_coords[:, 2].long(),
                atNameHashing[~exp_mask],
            )
            volume_cubes = volume_cubes[~exp_mask].view(-1)

            volume = volume.index_put(index, volume_cubes.view(-1), accumulate=True)

            volume = -torch.nn.functional.threshold(-volume, -1, -1)
            volume = volume.permute(0, 4, 1, 2, 3)

        return volume

    """
    mesh will be added as soon as pytorch3d becomes a little more stable
    def mesh(self,coords, radius,threshSurface = 0.01):

        atNameHashing= torch.zeros(radius.shape).to(self.dev)
        mask  = radius.eq(PADDING_INDEX)
        atNameHashing = atNameHashing.masked_fill_(mask,PADDING_INDEX)
        vol = self(coords,radius,atNameHashing).to_dense()
        mesh = cubifyNOALIGN(vol.sum(-1),thresh=threshSurface)# creates pytorch 3d mesh from cubes. It uses a MODIFIED version of pytorch3d with no align
        return mesh
    """

    def get_voxels_coordinates(self, channels, resolution=1, cubes_around_atoms_dim=5):
        ncubes = 2 * cubes_around_atoms_dim + 1
        center = ncubes // 2
        padding_mask = ~channels.eq(PADDING_INDEX)

        # Unflatten the cubes coordinates
        natoms = self.discrete_coordinates[padding_mask].shape[0]
        cubes_coords_3d = self.cubes_coords.view(
            [natoms] + list(self.standard_cube.squeeze(0).shape)
        )

        # Get the voxel coordinates for the 1st atom
        at0_voxel = (
            cubes_coords_3d[0][center, center, center] - 1
        ) * resolution + self.translation
        # Get the coordinates of the voxel 0,0,0
        voxel_0 = at0_voxel - cubes_coords_3d[0][center, center, center] * resolution

        # Generate the coordinates for the rest of the voxels
        voxels_coordinates = torch.stack(
            [
                (voxel_0 + displacement).squeeze()
                for displacement in torch.cartesian_prod(
                    *[torch.arange(d + 1) for d in self.boxsize]
                )
            ]
        )

        # Reshape them into the 3D format
        self.voxels_coordinates = voxels_coordinates.view(
            [d + 1 for d in self.boxsize] + [3]
        )

        return self.voxels_coordinates


class PointCloudSurface(torch.nn.Module):
    def __init__(self, device="cpu"):
        """
        Constructor for the CloudPointSurface class, which builds the main PyUUL object for cloud surface.

        Parameters
        ----------
        device : torch.device
            The device on which the model should run. E.g. torch.device("cuda") or torch.device("cpu:0")


        Returns
        -------
        """
        super(PointCloudSurface, self).__init__()

        self.device = device

    def __buildStandardSphere(self, npoints=50):  # Fibonacci lattice
        goldenRatio = (1 + 5**0.5) / 2
        i = torch.arange(0, npoints, device=self.device)
        theta = 2 * math.pi * i / goldenRatio
        phi = torch.acos(1 - 2 * (i + 0.5) / npoints)

        x, y, z = (
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        )

        coords = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
        # plot_volume(False,20*coords.unsqueeze(0))

        return coords

    def forward(self, coords, radius, maxpoints=5000, external_radius_factor=1.4):
        """
        Function to calculate the surface cloud point representation of macromolecules

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates of the atoms. Shape ( batch, numberOfAtoms, 3 ). Can be calculated from a PDB file using utils.parsePDB
        radius : torch.Tensor
            Radius of the atoms. Shape ( batch, numberOfAtoms ). Can be calculated from a PDB file using utils.parsePDB and utils.atomlistToRadius
        maxpoints : int
            number of points per macromolecule in the batch
        external_radius_factor=1.4
            multiplicative factor of the radius in order ot define the place to sample the points around each atom. The higher this value is, the smoother the surface will be
        Returns
        -------
        surfacePointCloud : torch.Tensor
            surface point cloud representation of the macromolecules in the batch. Shape ( batch, channels, numberOfAtoms, 3)

        """
        padding_mask = ~radius.eq(PADDING_INDEX)

        batch = coords.shape[0]
        npoints = (
            maxpoints // padding_mask.sum(-1).min() + 1
        ) * 2  # we ensure that the smallest protein has at least maxpoints points

        sphere = self.__buildStandardSphere(npoints)
        finalPoints = []

        for b in range(batch):
            distmat = torch.cdist(
                coords[b][padding_mask[b]].unsqueeze(0),
                coords[b][padding_mask[b]].unsqueeze(0),
            )
            L = distmat.shape[1]
            AtomSelfContributionMask = torch.eye(
                L, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            triangular_mask = ~torch.tril(
                torch.ones((L, L), dtype=torch.bool, device=self.device), diagonal=-1
            ).unsqueeze(0)

            # todoMask = (distmat[b].le(5) & (~AtomSelfContributionMask) & triangular_mask).squeeze(0)
            external_radius = radius * external_radius_factor
            todoMask = (distmat[0].le(5) & (~AtomSelfContributionMask)).squeeze(0)
            points = coords[b][padding_mask[b]].unsqueeze(0).unsqueeze(
                -2
            ) - sphere.unsqueeze(0).unsqueeze(1) * external_radius[b][
                padding_mask[b]
            ].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            p = points.expand(L, L, npoints, 3)[todoMask]
            c = (
                coords[b][padding_mask[b]]
                .unsqueeze(1)
                .unsqueeze(-2)
                .expand(L, L, points.shape[2], 3)[todoMask]
            )
            r = (
                radius[b][padding_mask[b]]
                .unsqueeze(1)
                .unsqueeze(-2)
                .expand(L, L, points.shape[2])[todoMask]
            )
            occupancy = self.__occupancy(p, c, r)

            point_index = (
                torch.arange(0, L * npoints, device=self.device)
                .view(L, npoints)
                .unsqueeze(0)
                .expand(L, L, npoints)[todoMask]
            )
            point_occupancy = torch.zeros(
                (L * npoints), dtype=torch.float, device=self.device
            )
            point_occupancy = point_occupancy.index_put_(
                [point_index.view(-1)], occupancy.view(-1), accumulate=True
            )
            point_occupancy = 1 - torch.exp(point_occupancy)

            points_on_surfaceMask = point_occupancy.le(0.5)

            points = (
                points.permute(0, 3, 1, 2)
                .view(3, -1)
                .transpose(0, 1)[points_on_surfaceMask]
            )
            random_indices = torch.randint(
                0, points.shape[0], [maxpoints], device=self.device
            )
            sampled_points = points[random_indices, :]

            finalPoints += [sampled_points]

        return torch.cat(finalPoints, dim=0)

    def __occupancy(self, points, coords, radius):
        dist = torch.norm(points - coords, dim=-1)

        sigma = 0.93
        exponent = -(dist**2) / (sigma**2 * radius**2)
        exp_mask = exponent.ge(10)
        exponent = torch.masked_fill(exponent, exp_mask, 10)

        occupancy_on_points = torch.exp(exponent)
        return torch.log(1 - occupancy_on_points)
        return occupancy_on_points
        del exponent

        AtomSelfContributionMask = (
            torch.eye(L, dtype=torch.bool, device=self.device)
            .unsqueeze(0)
            .expand(batch, L, L)
        )
        occupancy_on_points[AtomSelfContributionMask] = 0.0

        occupancy = 1 - torch.exp(
            torch.log(1 - occupancy_on_points).sum(2)
        )  # .sum(dim=-1)/npoints
        # if log_correction:
        # occupancy = -torch.log(occupancy + 1) # log scaling
        return occupancy


class PointCloudVolume(torch.nn.Module):
    def __init__(self, device="cpu"):
        """
        Constructor for the CloudPointSurface class, which builds the main PyUUL object for volumetric point cloud.

        Parameters
        ----------
        device : torch.device
            The device on which the model should run. E.g. torch.device("cuda") or torch.device("cpu:0")


        Returns
        -------
        """
        super(PointCloudVolume, self).__init__()

        self.device = device

    def forward(self, coords, radius, maxpoints=5000):
        """
        Function to calculate the volumetric cloud point representation of macromolecules

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates of the atoms. Shape ( batch, numberOfAtoms, 3 ). Can be calculated from a PDB file using utils.parsePDB
        radius : torch.Tensor
            Radius of the atoms. Shape ( batch, numberOfAtoms ). Can be calculated from a PDB file using utils.parsePDB and utils.atomlistToRadius
        maxpoints : int
            number of points per macromolecule in the batch

        Returns
        -------
        PointCloudVolume : torch.Tensor
            volume point cloud representation of the macromolecules in the batch. Shape ( batch, channels, numberOfAtoms, 3)

        """

        padding_mask = ~radius.eq(PADDING_INDEX)

        # npoints = torch.div(maxpoints, padding_mask.sum(-1).min()) + 1 # we ensure that the smallest protein has at least 5000 points

        batch = coords.shape[0]
        L = coords.shape[1]

        batched = []
        for i in range(batch):
            mean = coords[i][padding_mask[i]]

            sampled = (
                radius[i][padding_mask[i]].sqrt().unsqueeze(-1)
                * torch.randn((mean.size()), device=self.device)
                + mean
            )
            p = sampled.view(-1, 3)
            random_indices = torch.randint(
                0, p.shape[0], [maxpoints], device=self.device
            )
            batched += [p[random_indices].unsqueeze(0)]

        batched = torch.cat(batched, dim=0)
        return batched
