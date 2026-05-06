# Phase 1 MVP Task List
## GNN-Based Intrusion Detection System for SDN

**Project:** Major Project — 6th Semester, Batch B24  
**Purpose:** Step-by-step implementation plan for the Phase 1 MVP  
**Reference:** `Docs/GNN_IDS_MVP_Phase1.md`

---

## How to use this file

- Follow the tasks in order.
- Do not move to the next milestone until the current one is stable.
- Each task includes an expected output so progress is easy to verify.
- Phase 1 ends when the offline MVP pipeline works and results are documented.

---

## Milestone 0 — Setup and Workspace Readiness
**Goal:** Make the project runnable and organized before any model work starts.

### 0.1 Repository structure
- [ ] Create a clean project structure for `data/`, `notebooks/`, `preprocessing/`, `models/`, `results/`, and `docs/`
- [ ] Add or update `.gitignore` to exclude datasets, caches, checkpoints, and notebook outputs
- [ ] Create a short `README.md` with project title, team members, and Phase 1 objective

**Expected output:** A workspace that is ready for dataset and code development.

### 0.2 Environment setup
- [ ] Set up the Python environment with required packages for data processing, ML, and GNN work
- [ ] Verify imports for `pandas`, `numpy`, `scikit-learn`, `networkx`, `torch`, and `torch_geometric`
- [ ] Confirm that the team can run notebooks or scripts in the same environment

**Expected output:** A reproducible environment that supports preprocessing and model training.

### 0.3 Folder conventions
- [ ] Create dataset subfolders for raw and cleaned files
- [ ] Create an output folder for plots, metrics, and saved models
- [ ] Decide file naming conventions for graphs, checkpoints, and reports

**Expected output:** A consistent folder layout that prevents confusion during implementation.

---

## Milestone 1 — Dataset Acquisition
**Goal:** Obtain the datasets needed for offline MVP development.

### 1.1 Download datasets
- [ ] Download CICIDS2017 from the official source
- [ ] Download InSDN for SDN-specific evaluation
- [ ] Store raw data separately from cleaned data

**Expected output:** Raw copies of both datasets available locally.

### 1.2 Verify data quality
- [ ] Inspect file formats, column names, and label fields
- [ ] Check whether both datasets contain missing values, infinite values, and duplicates
- [ ] Record any dataset-specific issues that may affect preprocessing

**Expected output:** A short note describing dataset structure and known issues.

### 1.3 Define label mapping
- [ ] Decide whether labels will be binary for MVP or multi-class for later expansion
- [ ] Map label names consistently across datasets
- [ ] Document the final label scheme in the docs folder

**Expected output:** A stable label mapping that can be reused in preprocessing and training.

---

## Milestone 2 — Data Cleaning and EDA
**Goal:** Turn raw data into a usable, documented training dataset.

### 2.1 Exploratory data analysis
- [ ] Load the dataset into a notebook for inspection
- [ ] Check dimensions, dtypes, and summary statistics
- [ ] Plot class distribution for benign vs. attack records
- [ ] Identify the most important flow features for the MVP

**Expected output:** An EDA notebook or summary document with visual and statistical insights.

### 2.2 Cleaning pipeline
- [ ] Replace infinite values with `NaN`
- [ ] Handle missing values using a clear rule such as median imputation or row removal
- [ ] Drop duplicate rows
- [ ] Standardize whitespace and label casing
- [ ] Remove clearly unusable columns such as identifiers that should not be learned directly if needed

**Expected output:** Cleaned CSV files ready for baseline training and graph conversion.

### 2.3 Save cleaned dataset
- [ ] Save cleaned CICIDS2017 data to a dedicated folder
- [ ] Save cleaned InSDN data to a dedicated folder
- [ ] Keep a record of the cleaning steps so they can be reproduced

**Expected output:** Reproducible cleaned datasets with documented preprocessing rules.

---

## Milestone 3 — Baseline Model Pipeline
**Goal:** Build a baseline benchmark before training the GNN.

### 3.1 Feature selection
- [ ] Select a compact set of useful flow features for baseline training
- [ ] Exclude raw identifiers that do not help generalization
- [ ] Confirm the selected features are available in both datasets or clearly separate the dataset-specific pipeline

**Expected output:** A finalized baseline feature set.

### 3.2 Train-test split
- [ ] Split data into train, validation, and test sets
- [ ] Use stratified splitting to preserve class balance
- [ ] Fit preprocessing only on training data to avoid leakage

**Expected output:** Stable data splits with no leakage.

### 3.3 Train baseline models
- [ ] Train Random Forest on cleaned flow features
- [ ] Train XGBoost on the same feature set
- [ ] Tune basic hyperparameters only if needed for a fair baseline comparison

**Expected output:** Baseline models trained on the same data split.

### 3.4 Baseline evaluation
- [ ] Compute Accuracy, Precision, Recall, F1-score, and ROC-AUC
- [ ] Generate confusion matrices for both models
- [ ] Record baseline scores in a results document

**Expected output:** Baseline performance values that the MVP GNN must compare against.

---

## Milestone 4 — Graph Construction Pipeline
**Goal:** Convert cleaned flow records into graph snapshots for the GNN.

### 4.1 Define graph schema
- [ ] Define what a node represents in the MVP graph
- [ ] Define what an edge represents in the MVP graph
- [ ] Decide which features belong to nodes and which belong to edges
- [ ] Write down the graph schema before coding

**Expected output:** A clear graph design specification.

### 4.2 Implement feature extraction
- [ ] Implement flow feature extraction for edge attributes
- [ ] Implement aggregated node feature extraction for each window
- [ ] Handle edge cases such as empty windows or missing node pairs

**Expected output:** Reusable feature extraction functions.

### 4.3 Implement sliding window graph builder
- [ ] Choose a window size for graph snapshots
- [ ] Build a function that groups flows by time window
- [ ] Create one graph snapshot per window
- [ ] Assign graph labels using the chosen label strategy

**Expected output:** A working graph builder that transforms flow tables into graph objects.

### 4.4 Validate graph output
- [ ] Print number of nodes, edges, and graph labels for sample windows
- [ ] Visualize a few graphs to confirm they look sensible
- [ ] Save a small graph sample for debugging and presentation

**Expected output:** Verified graph snapshots that can be used for model training.

### 4.5 Dataset packaging
- [ ] Convert graph snapshots into a list or dataset object usable by PyTorch Geometric
- [ ] Save the processed graphs to disk
- [ ] Add a loader that can read the saved graphs later

**Expected output:** A reusable graph dataset pipeline.

---

## Milestone 5 — GNN Model Prototype
**Goal:** Train an initial graph model on the processed data.

### 5.1 Pick model type
- [ ] Choose a first GNN architecture such as GAT or GraphSAGE
- [ ] Decide whether the MVP will focus on graph-level or edge-level classification
- [ ] Keep the architecture simple enough to train and debug quickly

**Expected output:** A clear model choice for the MVP.

### 5.2 Implement the model
- [ ] Create the GNN model class
- [ ] Add input layers for node and edge features if required
- [ ] Add classification head for benign vs. attack output
- [ ] Verify the forward pass works on a dummy graph

**Expected output:** A functioning GNN model definition.

### 5.3 Train the model
- [ ] Build a training loop
- [ ] Use class weighting if the dataset is imbalanced
- [ ] Track train and validation loss
- [ ] Save the best checkpoint during training

**Expected output:** A trained GNN checkpoint from the best validation epoch.

### 5.4 Evaluate the model
- [ ] Compute test-set metrics
- [ ] Compare the GNN to baseline models
- [ ] Identify whether graph structure improved detection on attack-heavy windows

**Expected output:** Final GNN performance numbers for Phase 1.

---

## Milestone 6 — Result Analysis and Documentation
**Goal:** Convert the MVP work into something ready for guide review.

### 6.1 Comparison report
- [ ] Prepare a table comparing Random Forest, XGBoost, and the GNN
- [ ] Summarize where the GNN performs well and where it needs improvement
- [ ] Mention any limitations of the offline MVP

**Expected output:** A concise comparison report for the guide.

### 6.2 Visuals
- [ ] Generate confusion matrices for all models
- [ ] Plot class distribution and feature insights from EDA
- [ ] Add one or two graph visualizations to explain the concept

**Expected output:** A small set of presentation-ready visuals.

### 6.3 Documentation
- [ ] Update `Docs/GNN_IDS_MVP_Phase1.md` if any implementation choices changed
- [ ] Write a short results summary in markdown
- [ ] Note what will move to Phase 2, especially SDN live integration

**Expected output:** Clean documentation that explains the MVP clearly.

---

## Milestone 7 — Phase 1 Review Readiness
**Goal:** Prepare the final material for the major project guide evaluation.

### 7.1 Final checks
- [ ] Verify that the full preprocessing-to-model pipeline can run without manual fixes
- [ ] Verify that baseline and GNN results are saved
- [ ] Verify that all important code files and notebooks are organized

**Expected output:** A stable demo-ready offline MVP.

### 7.2 Presentation support
- [ ] Prepare a 1-page summary of the problem, approach, and MVP results
- [ ] Prepare a simple architecture diagram for the guide meeting
- [ ] Prepare a short explanation of why graph learning is better suited to SDN traffic

**Expected output:** Materials that can be shown directly during Phase 1 evaluation.

### 7.3 Submission checklist
- [ ] Clean notebook outputs if needed
- [ ] Ensure results are reproducible from the documented steps
- [ ] Keep all Phase 1 artifacts in the appropriate folders

**Expected output:** A tidy project state ready for review.

---

## Recommended Order of Execution

1. Set up environment and folder structure
2. Download and inspect datasets
3. Clean data and complete EDA
4. Train baseline models
5. Build graph construction pipeline
6. Train the first GNN model
7. Compare results and document findings
8. Prepare review material for the guide

---

## Phase 1 Done Criteria

Phase 1 can be considered complete when all of the following are true:
- Cleaned datasets are available and documented
- EDA findings are summarized
- Baseline models are trained and evaluated
- Graph snapshots are being generated correctly
- A GNN prototype is trained and tested
- Results are written in a markdown report
- The project is ready to present to the guide

---

## Notes for Phase 2

The following items are intentionally postponed and should not block Phase 1:
- Mininet deployment
- Ryu controller integration
- Live traffic monitoring
- Automated mitigation rules
- Real-time performance tuning

These items should be planned after the offline MVP is accepted.
