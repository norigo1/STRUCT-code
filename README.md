# Harmonising forest structural metrics from LiDAR and passive optical instruments across scales (STRUCT) / STRUCT-code

![DOI: pending](https://img.shields.io/badge/DOI-pending-lightgrey)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

> **Brief summary:** The software presented here gives users a suite of tools to set-up radiative transfer simulations and process their outputs to retrieve forest structural metrics.

---

## Overview & Scientific Context

This software was developed as part of the **STRUCT** project (*Harmonising forest structural metrics from LiDAR and passive optical instruments across scales*), a Marie Skłodowska-Curie Actions (MSCA) Postdoctoral Fellowship hosted at **Ghent University**. 

* **Purpose:** The library presented here is a suite of tools to set up and process radiative transfer simulations that assess the structure of forests. This deals with the amendment of existing 3D models of forests to correspond to 'true' forest structural parameters as well as the simulation and processing to produce the expected outputs from measurement instruments that would be used in the field.
* **Related Publications:** To follow

<!-- Link to preprints, peer-reviewed papers, or datasets on [Open Research Europe](https://open-research-europe.ec.europa.eu/ "Open Research Europe") or [Zenodo](https://zenodo.org "Zenodo") that this code validates. -->

---

## Installation & System Requirements

Detail the exact environment required to reproduce your results:

* **OS / Hardware:** Tested on MacOS Tahoe 26 / ARM
* **Dependencies:** [`requirements.txt`](requirements.txt)

```bash
# Example installation steps
git clone git@github.com:norigo1/STRUCT-code.git
cd STRUCT-code
pip install -r requirements.txt
```

---

## Usage & Reproducibility
See [CODE_EXAMPLES.md](./CODE_EXAMPLES.md) for examples covering HELIOS scene and
survey preparation, occlusion and point-cloud analysis, and scene LAI/PAI
calculations.

---

## How to Cite
To follow
<!If you use this software in your research, please cite both the software archive and the accompanying paper (see `CITATION.cff` for metadata):>

```bibtex
@software{Origoetal_2026_STRUCT,
  author       = {Origo, Niall and Massaro, Simone and Van den Broeck, Wouter and Calders, Kim},
  title        = {{STRUCT-code}},
  month        = mon,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org}
}
```

---

## License

This project is licensed under the **GPL-3.0** License — see the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.en.html) file for details.

---
## AI usage

This package used Microsoft Copilot in the generation of the code in this library.

----
## Funding Acknowledgment

<img src="./normal-reproduction-low-resolution.jpg" alt="Funded by the European Union" width="220"/>

This project has received funding from the European Union’s Horizon Europe research and innovation programme under the Marie Skłodowska-Curie grant agreement No **101264077** (**STRUCT**). 

*Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.*
