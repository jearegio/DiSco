# future use

# import numpy as np
# import rdkit
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import openbabel as ob
# import featurizer
# import model

# def read_xyz_file(filepath):
#     """Read an .xyz file and convert it to an RDKit molecule with 3D coordinates."""
#     obConversion = ob.OBConversion()
#     obConversion.SetInAndOutFormats("xyz", "mdl")
#     mol = ob.OBMol()
#     if not obConversion.ReadFile(mol, filepath):
#         raise IOError("Could not read the .xyz file.")
#     rdkit_mol = rdkit.Chem.MolFromMolBlock(obConversion.WriteString(mol))
#     if rdkit_mol is None:
#         raise ValueError("Failed to convert to RDKit molecule.")
#     AllChem.EmbedMolecule(rdkit_mol, useRandomCoords=True)  # Ensure 3D coords are OK
#     return rdkit_mol

# def prepare_and_infer(model, filepath):
#     """Prepare the molecule from .xyz file and run the model inference."""
#     rdkit_mol = read_xyz_file(filepath)
#     featurizer_obj = featurizer.DownstreamTransformFn(is_inference=True)
#     graph_data = featurizer_obj({'smiles': rdkit_mol})  # Modified to take RDKit Mol directly
    
#     collate_fn = featurizer.DownstreamCollateFn(
#         atom_names=[],  # List the names of atom features
#         bond_names=[],  # List the names of bond features
#         bond_float_names=[],  # List names of bond float features
#         bond_angle_float_names=[],  # List names of bond angle float features
#         task_type='class',  # Assuming classification task
#         is_inference=True
#     )
    
#     # Prepare graphs for model input
#     atom_bond_graph, bond_angle_graph = collate_fn([graph_data])
#     # Predict using the model
#     predictions = model(atom_bond_graph, bond_angle_graph)
#     return predictions

# # Load your model here (assuming it has been initialized elsewhere in your script)
# # my_model = model.DownstreamModel(model_config, compound_encoder)
# # predictions = prepare_and_infer(my_model, "path/to/your/file.xyz")
