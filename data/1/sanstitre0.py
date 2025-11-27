# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 20:04:20 2025

@author: boubo
"""

import re
import numpy as np

nodes = []

with open("beam_2d.fem", "r") as f:
    for line in f:
        # Cherche les lignes qui contiennent des noeuds (ex: "GRID" ou "*NODE" selon ton format)
        if line.strip().startswith("GRID"):
            # capture tous les nombres présents sur la ligne (ID, X, Y, Z, etc.)
            vals = re.findall(r"[-+]?\d*\.?\d+", line)
            if len(vals) >= 4:
                # ID, X, Y, Z
                node_id, x, y, z = map(float, vals[:4])
                # Ajout de l'ID en première colonne
                nodes.append([int(node_id), x, y])

# Conversion en array numpy (N, 3) : [ID, X, Y]
nodes = np.array(nodes)

print("Nombre de noeuds lus :", len(nodes))
print("Premier noeud :", nodes[:10])