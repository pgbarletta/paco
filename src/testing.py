import sys
from pathlib import Path
from Bio.PDB import PDBParser, Selection, Superimposer, PDBIO, Atom, Residue, Structure
import numpy as np
from pyuul import VolumeMaker # the main PyUUL module
from pyuul import utils
from myvolmaker import Voxels

casa = Path("/home/pbarletta/labo/23/paco")
dlp = Path("/home/pbarletta/labo/22/locuaz/rebin/dlpacker")
weights_path = Path(dlp, "DLPacker_weights.h5")
lib_path = Path(dlp, "DLPacker/library.npz")
charges_path = Path(dlp, "DLPacker/charges.rtp")

input_pdb_fn = Path(casa, "c.pdb")
coords, atname = utils.parsePDB(input_pdb_fn)
atoms_channel = utils.atomlistToChannels(atname)
radius = utils.atomlistToRadius(atname)

volmaker = Voxels(device="cpu",sparse=False)
voxelized_volume = volmaker(coords, radius, atoms_channel, resolution=1)
voxels = voxelized_volume.sum(1)[:, None, :, :]






# dlp = DLPacker(str(input_pdb_fn), weights_path=weights_path, lib_path=lib_path, charges_path=charges_path)
#
# dlp.mutate_sequence((27, "B", "ALA"), "ARG")
#
# targets = dlp.get_targets(target=(27, "B", "ARG"), radius=10)
# out_path = Path(input_pdb_fn.parent, "init_mutated.pdb")
#
# dlp.reconstruct_region(targets=targets, order='sequence', output_filename=str(out_path))
