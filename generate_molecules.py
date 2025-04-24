import random
import torch
import dgl
import pandas as pd

from model import VGAE, n_hidden_z
from utils import postprocess_decoded_graph, graph_to_smiles

# Output CSV path
generated_smiles_save_path = "/content/zinc_50_generated_smiles.csv"

# Number of molecules to generate
nb_molecules_to_generate = 50
min_nb_atoms = 5
max_nb_atoms = 20

nb_node_types = 14
nb_edge_types = 3

# Load trained model
model = VGAE(nb_node_types, nb_edge_types)
model.load_state_dict(torch.load("/content/Molecular-Graph-Generation/trained_model.pkl"))
model.eval()

# Generate molecules and convert to SMILES
generated_smiles = []
with torch.no_grad():
    for i in range(nb_molecules_to_generate):
        nb_atoms = random.randint(min_nb_atoms, max_nb_atoms)
        # sample latent vectors
        z = torch.randn(nb_atoms, n_hidden_z)
        gz = z.mean(dim=0, keepdim=True)

        # decode graph
        decoded_node_types, decoded_edges, _ = model.decode(z, gz, [nb_atoms])[0]
        node_types, edges_src, edges_dst, edge_types = postprocess_decoded_graph(
            decoded_node_types, decoded_edges
        )
        smiles = graph_to_smiles(node_types, edges_src, edges_dst, edge_types)
        generated_smiles.append(smiles)

# Save generated SMILES to CSV
df = pd.DataFrame({"smiles": generated_smiles})
df.to_csv(generated_smiles_save_path, index=False)

print(f"Saved {len(generated_smiles)} generated SMILES to {generated_smiles_save_path}")
