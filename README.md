# Neuro-ODE in Modeling Hub

This repository contains materials for implementing and documenting **ODE-based epidemic models** developed in alignment with the **COVID-19 Modeling Hub** framework, with an emphasis on extensibility toward **Neural ODE (Neuro-ODE)** approaches.

---

## Repository Structure

### `ODEv1.0/`

The `ODEv1.0` folder includes all components required to reproduce results from **ODE version 1.0**:

- **`data/`**  
  Input datasets used for model calibration and evaluation, including hospitalization data and vaccination.

- **`code/`**  
  Source code implementing the deterministic and stochastic ODE models, parameter estimation procedures, and sensitivity analyses.

- **`ODEv1.0.html/`**  
  Final HTML documentation generated for ODE version 1.0, describing model structure, assumptions, parameter definitions, and results.  

---

## Model Description

ODE version 1.0 implements a mechanistic epidemic modeling framework designed to be compatible with COVID-19 Modeling Hub standards. Key features include:

- A compartmental ODE-based transmission model  
- Transmission rate estimation using a least-squares fitting approach
- Deterministric and stochastic model results in fitting and projection
- Sensitivity analysis across key epidemiological parameters  
- Reproducible outputs summarized in a standalone HTML report  

This version serves as the **baseline ODE implementation**, providing a foundation for future extensions.

---

## Purpose

The goals of this repository are to:

- Provide a reproducible ODE-based modeling pipeline  
- Align model implementation and reporting with COVID-19 Modeling Hub conventions  
- Serve as a modular starting point for integrating **Neural ODE (Neuro-ODE)** methods in later versions  

