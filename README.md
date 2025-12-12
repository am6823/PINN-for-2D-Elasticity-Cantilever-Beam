# PINN-for-2D-Elasticity-Cantilever-Beam

This repository contains a complete implementation of a **Physics-Informed Neural Network (PINN)** for the classical **2D cantilever beam** benchmark in linear elasticity.

Both stages of the project are implemented and fully functional in the notebook:

- **Stage 1 — Physics-Only PINN**  
  The model learns the entire elasticity problem (PDE, boundary conditions, constitutive law, bending/moment constraints, beam-theory priors) **without any labeled data**.

- **Stage 2 — Hybrid PINN (Physics + Sparse FEA Data)**  
  A second training mode incorporates **selected FEA displacements/stresses** to refine accuracy, while still enforcing all physical constraints.

The notebook reproduces displacement fields, stress distributions, moment diagrams, neutral-axis behavior, and compares the results against a reference FEA solution.

---

## Current Status

**Physics-only PINN: complete**  
**Hybrid PINN with FEA data: complete**  
**Google Colab notebook: full workflow implemented**  
**Modular `src/` code:** partially implemented, will be finalized later  
**Full documentation:** coming soon  

The main development and experimentation currently occur in the notebook, which serves as the authoritative and complete version of the method.

---


## Problem Summary

We solve a 2D cantilever beam under transverse loading using linear elasticity:

- Plane-stress constitutive law  
- Governing PDE:  
  ∇ · σ = 0  
- Clamped boundary at x = x_min  
- Free traction boundaries on top/bottom  
- Shear resultant matching the applied load  
- Bending moment consistency  

The hybrid version additionally integrates selected FEA nodal quantities.

---

## Usage

Run the full solution through the notebook: CantileverBeam_PINN.ipynb


It includes:

- mesh loading  
- geometry/boundary extraction  
- physics-only and hybrid PINN training  
- convergence plots  
- displacement/stress fields  
- comparison to FEA reference  

---

## Notes

The `src/` directory already contains early drafts of the modular architecture.  
They will be completed and connected to the notebook at a later stage.

---




