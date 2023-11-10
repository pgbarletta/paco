from pathlib import Path
from typing import Tuple

from Bio.PDB import PDBParser, Selection, Superimposer, PDBIO, Atom, Residue, Structure
import numpy as np
from numpy.typing import NDArray

THE20 = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}
SCH_ATOMS = {
    "ALA": 1,
    "ARG": 7,
    "ASN": 4,
    "ASP": 4,
    "CYS": 2,
    "GLN": 5,
    "GLU": 5,
    "GLY": 0,
    "HIS": 6,
    "ILE": 4,
    "LEU": 4,
    "LYS": 5,
    "MET": 4,
    "PHE": 7,
    "PRO": 3,
    "SER": 2,
    "THR": 3,
    "TRP": 10,
    "TYR": 8,
    "VAL": 3,
}
BB_ATOMS = ["C", "CA", "N", "O"]
SIDE_CHAINS = {
    "MET": ["CB", "CE", "CG", "SD"],
    "ILE": ["CB", "CD1", "CG1", "CG2"],
    "LEU": ["CB", "CD1", "CD2", "CG"],
    "VAL": ["CB", "CG1", "CG2"],
    "THR": ["CB", "CG2", "OG1"],
    "ALA": ["CB"],
    "ARG": ["CB", "CD", "CG", "CZ", "NE", "NH1", "NH2"],
    "SER": ["CB", "OG"],
    "LYS": ["CB", "CD", "CE", "CG", "NZ"],
    "HIS": ["CB", "CD2", "CE1", "CG", "ND1", "NE2"],
    "GLU": ["CB", "CD", "CG", "OE1", "OE2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "PRO": ["CB", "CD", "CG"],
    "GLN": ["CB", "CD", "CG", "NE2", "OE1"],
    "TYR": ["CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "OH"],
    "TRP": ["CB", "CD1", "CD2", "CE2", "CE3", "CG", "CH2", "CZ2", "CZ3", "NE1"],
    "CYS": ["CB", "SG"],
    "ASN": ["CB", "CG", "ND2", "OD1"],
    "PHE": ["CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ"],
}


casa = Path("/home/pbarletta/labo/23/paco")
dlp = Path("/home/pbarletta/labo/22/locuaz/rebin/dlpacker")
weights_path = Path(dlp, "DLPacker_weights.h5")
lib_path = Path(dlp, "DLPacker/library.npz")
charges_path = Path(dlp, "DLPacker/charges.rtp")

input_pdb_fn = Path(casa, "d11.pdb")
#
# dlp = DLPacker(str(input_pdb_fn), weights_path=weights_path, lib_path=lib_path, charges_path=charges_path)
#
# dlp.mutate_sequence((27, "B", "ALA"), "ARG")
#
# targets = dlp.get_targets(target=(27, "B", "ARG"), radius=10)
# out_path = Path(input_pdb_fn.parent, "init_mutated.pdb")
#
# dlp.reconstruct_region(targets=targets, order='sequence', output_filename=str(out_path))


def _align_residue(self, residue: Residue):
    # In order to generate input box properly
    # we first need to align selected residue
    # to reference atoms from reference.pdb
    if not residue.has_id("N") or not residue.has_id("C") or not residue.has_id("CA"):
        print("Missing backbone atoms: residue", self._get_residue_tuple(residue))
        return False
    r = list(self.reference.get_atoms())
    s = [residue["N"], residue["CA"], residue["C"]]
    self.sup.set_atoms(r, s)
    self.sup.apply(self._get_parent_structure(residue))
    return True


def _get_box_atoms(self, residue: Residue):
    # Alighns selected residue to reference positions
    # and selects atoms that lie withing some cube.
    # Cube size is 10 angstroms by default and an
    # additional offset of 1 angstrom is used
    # to include all atoms bc even if the atom is
    # slightly outside the cube, due to gaussian kernel
    # blur, some of the density might still be within
    # 10 angstrom
    aligned = self._align_residue(residue)
    if not aligned:
        return []
    atoms = []
    b = self.box_size + 1  # one angstrom offset to include more atoms
    for a in self._get_parent_structure(residue).get_atoms():
        xyz = a.coord
        if (
            xyz[0] < b
            and xyz[0] > -b
            and xyz[1] < b
            and xyz[1] > -b
            and xyz[2] < b
            and xyz[2] > -b
        ):
            atoms.append(a)
    return atoms


def _genetare_input_box(self, residue: Residue, allow_missing_atoms: bool = False):
    # Takes a residue and generates a special
    # dictionary that is then given to InputReader,
    # which uses this dictionary to generate the actual input
    # for the neural network
    # Input:
    # residue             - the residue we want to restore
    # allow_missing_atoms - boolean flag that allows or disallows
    #                       missing sidechain atoms
    atoms = self._get_box_atoms(residue)
    if not atoms:
        return None

    r, s, n = self._get_residue_tuple(residue)

    exclude, types, resnames = [], [], []
    segids, positions, names = [], [], []
    resids = []

    for i, a in enumerate(atoms):
        p = a.get_parent()
        a_tuple = (p.get_id()[1], p.get_full_id()[2], p.get_resname())
        if a.get_name() not in BB_ATOMS and (r, s, n) == a_tuple:
            exclude.append(i)

        types.append(a.element)
        resnames.append(a.get_parent().get_resname())
        segids.append(a.get_parent().get_full_id()[2])
        positions.append(a.coord)
        names.append(a.get_name())
        resids.append(a.get_parent().get_id()[1])

    d = {
        "target": {"id": int(r), "segid": s, "name": n, "atomids": exclude},
        "types": np.array(types),
        "resnames": np.array(resnames),
        "segids": np.array(segids),
        "positions": np.array(positions, dtype=np.float16),
        "names": np.array(names),
        "resids": np.array(resids),
    }

    if allow_missing_atoms or len(exclude) == SCH_ATOMS[n]:
        return d
    else:
        return None


