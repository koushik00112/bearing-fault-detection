# Data

Checked 2026-09-24 against the official pages. Re-check the terms before publishing
anything derived from the data.

## Paderborn University (KAt) bearing dataset: main dataset

- **Source:** https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter. The files are
  served from `https://groups.uni-paderborn.de/kat/BearingDataCenter/` (32 `.rar`
  archives of about 150–180 MB each). An unofficial Zenodo re-upload exists but is missing
  some bearings, so use the official server.
- **Licence:** Creative Commons **Attribution-NonCommercial 4.0** (CC BY-NC 4.0).
  Non-commercial academic use is allowed with a citation; commercial use needs the authors'
  permission.
- **Required citation:**
  > Christian Lessmeier et al., KAt-DataCenter: mb.uni-paderborn.de/kat/forschung/bearing-datacenter,
  > Chair of Design and Drive Technology, Paderborn University.

  Reference paper: C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro, "Condition Monitoring
  of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of
  Electric Motors: A Benchmark Data Set for Data-Driven Classification", European
  Conference of the PHM Society, 2016.

### What this repo commits and what it doesn't
- **Never committed:** raw files, processed windows (`data/`), or trained models
  (`models/`, `results/**/models/`). They are derived from the licensed data.
- **Committed:** code, aggregate results (`summary.md`, figures), and the manifest hash
  that identifies which raw files produced them.

### Contents used here
- 6 healthy bearings (K001–K006), 12 with artificial damage, and 14 with real damage from
  accelerated lifetime tests. The table is in `bearing/data/paderborn.py`.
- **Classes:** healthy / inner race / outer race. The 3 bearings with combined damage
  (KB23, KB24, KB27) are excluded, because there is no artificial counterpart to train on
  in scenario D.
- **Operating conditions:** N15_M07_F10 (1500 rpm, 0.7 Nm, 1000 N), N09_M07_F10 (900 rpm),
  N15_M01_F10 (0.1 Nm), N15_M07_F04 (400 N). There are 20 recordings of 4 s per bearing
  and condition.
- **Channels:** `vibration_1`, `phase_current_1`, `phase_current_2`, recorded at 64 kHz.

### Preprocessing (`python -m bearing.prepare paderborn`)
- Zero-phase FIR decimation by 4 gives **16 kHz**, which drops content above 8 kHz (see
  [ADR 0002](adr/0002-preprocessing.md)).
- Windows of 2048 samples (128 ms) with a stride of 1024 (50% overlap). A contiguous block
  of 16 windows is taken from the middle of each recording.
- Files that fail to load are skipped and listed in `dataset.json` (`skipped_files`).

### Verify the loader on real files first
The `.mat` layout (a struct with field `Y`, whose entries have `Name` and `Data`) follows
the KAt documentation. The loader is tested against files built to that layout, not yet
against the real files. Before the full download, run:
```bash
make download-sample      # K001 + KA01, ~330 MB
.venv/bin/python -c "from pathlib import Path; from bearing.data import paderborn as p; \
r,s=p.load_all(Path('data/raw/paderborn')); print(len(r),'loaded',len(s),'skipped'); print(r[0])"
```
You should see 160 recordings loaded and 0 skipped. If anything is skipped, the error
lists the channel names the file actually contains.

## CWRU bearing data: sanity check only

- **Source:** https://engineering.case.edu/bearingdatacenter (12k drive-end fault data plus
  normal baseline, 40 files). The site shows no explicit licence; cite the data centre and
  don't redistribute the files.
- **Why sanity check only:** CWRU is widely reported to be easy for ML models, and simple
  splits leak. See W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics
  using the Case Western Reserve University data: A benchmark study", *MSSP* 64–65 (2015),
  and J. Hendriks, P. Dumond, D. A. Knox, "Towards better benchmarking using the CWRU
  bearing fault dataset", *MSSP* 169 (2022). A high score here shows the pipeline runs,
  nothing more.
- There's only one file per (fault, load), so a recording-level split isn't possible. Use
  scenarios A and C (load as the condition) only.

## Synthetic data

`bearing/data/synthetic.py` generates signals for tests, CI and demos. Everything it
produces has `source = "synthetic"`, and any report built from it carries a banner saying
the numbers mean nothing. It is **never** used for reported results.
