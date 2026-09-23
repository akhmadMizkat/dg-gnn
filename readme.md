# DG-GNN

**Dual-Gated Graph Neural Network for Financial Fraud Detection**

This repository contains the implementation of **DG-GNN (Dual-Gated Graph Neural Network)** proposed for fraud detection on large-scale social financial graphs.

The model introduces two gating mechanisms:

* **Edge-wise Gating** — controls incoming messages during neighborhood aggregation.
* **Dynamic Gated Update** — adaptively controls the integration between neighborhood information and the previous node representation.

The experiments are conducted on the **DGraphFin** financial fraud detection dataset.

---

## Requirements

The implementation requires Python and the following main libraries:

* PyTorch
* PyTorch Geometric
* NumPy
* scikit-learn
* pandas
* matplotlib

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## Dataset

This implementation uses the **DGraphFin** dataset.

The dataset is not included in this repository. Please download the dataset from its official source and place the required files in the expected data directory.

The DGraph dataset contains approximately:

* **3.7 million nodes**
* **4.3 million edges**

The experiments use the official training, validation, and test masks provided with the dataset.

---

## Running the Experiment

Clone the repository:

```bash
git clone https://github.com/akhmadMizkat/dg-gnn.git
cd dg-gnn
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Prepare the DGraphFin dataset according to the expected directory structure.

Then run the training script:

```bash
python main.py
```

The experiment trains DG-GNN using the selected hyperparameter configuration and evaluates the model on the validation and test sets.

---

## Hyperparameters

The implementation supports experimentation with the following main hyperparameters.

| Hyperparameter          | Options                            |
| ----------------------- | ---------------------------------- |
| Number of layers        | 3, 4, 5, 6                         |
| Hidden dimension        | 30, 40, 45, 50, 60, 70, 80         |
| Dropout                 | 0.2, 0.3                           |
| Learning rate           | 1e-4, 1e-3, 1e-2                   |
| Weight decay            | 1e-3, 2e-4, 1e-4, 7e-5, 1e-5, 1e-6 |
| Activation              | ELU, ReLU                          |
| Gating heads            | 1, 2, 3                            |
| Maximum epochs          | 500                                |
| Early stopping patience | 100                                |

The configuration used for the reported DG-GNN result is:

| Parameter               | Value |
| ----------------------- | ----: |
| Number of layers        |     3 |
| Hidden dimension        |    40 |
| Dropout                 |   0.3 |
| Learning rate           |  0.01 |
| Weight decay            |  7e-5 |
| Activation              |   ELU |
| Gating heads            |     1 |
| Maximum epochs          |   500 |
| Early stopping patience |   100 |

---

## Training Configuration

The training procedure uses the following settings:

* Official DGraphFin train/validation/test masks.
* 1:1 fraud-to-normal undersampling during training.
* A 3-hop training subgraph for the default 3-layer configuration.
* Full-graph evaluation on the validation set.
* Early stopping based on validation ROC-AUC.
* The best validation checkpoint is used for final test evaluation.
* Each configuration is evaluated over **10 independent runs**.

---

## Experimental Results

The reported results are obtained from 10 independent runs.

### Validation ROC-AUC

```text
0.8453166400560634 ± 0.0010472710760313232
```

Rounded:

**0.8453 ± 0.0010**

### Test ROC-AUC

```text
0.8537609340783966 ± 0.0008838184768272502
```

Rounded:

**0.8538 ± 0.0009**

| Metric                  | Mean ± Standard Deviation |
| ----------------------- | ------------------------: |
| Best Validation ROC-AUC |       **0.8453 ± 0.0010** |
| Best Test ROC-AUC       |       **0.8538 ± 0.0009** |

---

## Reproducibility

The reported result is based on repeated experiments rather than a single training run.

Each experiment is conducted using **10 independent runs**, and the reported value is the mean and standard deviation across the runs.

Due to GPU-based graph computation and sampling operations, small variations may occur depending on the hardware and software environment.

---

## Model Architecture

The DG-GNN architecture consists of two main gating stages:

```text
Node Features
      │
      ▼
Feature Engineering
      │
      ▼
Temporal Representation
      │
      ▼
Edge-wise Gating
      │
      ▼
Message Aggregation
      │
      ▼
Dynamic Gated Update
      │
      ▼
Node Representation
      │
      ▼
Fraud Classification
```

The edge-wise gate controls the contribution of incoming messages, while the dynamic gated update controls the balance between newly aggregated neighborhood information and the previous node representation.

---

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{almahdidggnn,
  title={Fraud Detection Model on Social Financial Graph Using Dual-Gated Graph Neural Network (DG-GNN)},
  author={Al Mahdi, Muhamad Misykat Ali and Munir, Rinaldi and Mahayana, Dimitri},
  journal={Journal of ICT Research and Applications},
  year={2026}
}
```

---

## License

Please refer to the `LICENSE` file for licensing information.
