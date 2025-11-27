# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 20:35:08 2025

@author: boubo
"""

import numpy as np
import re

ids = []
coords = []

with open("beam_2d.fem", "r") as f:
    for line in f:
        if line.strip().startswith("GRID"):
            vals = re.findall(r"[-+]?\d*\.?\d+", line)
            if len(vals) >= 4:
                node_id, x, y, z = map(float, vals[:4])
                ids.append(int(node_id))
                coords.append([x, y])

nodes = np.array(coords)
node_ids = np.array(ids, dtype=int)

print("nodes:", nodes.shape)
print("node_ids:", node_ids[:10])