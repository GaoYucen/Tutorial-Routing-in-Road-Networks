# DRNCS

## 📁 Project Structure

### 🔧 Preprocessing Scripts

These scripts handle preprocessing of input datasets, particularly resolving non-strong connectivity issues in trajectory or map graphs:

* `data_preprocess_for_map.py`: Preprocesses map datasets for embedding.
* `data_preprocess_for_traj.py`: Preprocesses trajectory datasets.
* `data_preprocess_for_traj_BottomUp.py`: Bottom-up variant for trajectory preprocessing.
* `data_preprocess_for_traj_CH.py`: Contraction Hierarchies-based preprocessing for trajectories.

### 🧠 Model Architecture

* `model.py`: Main model structure used for training node embeddings.
* `model_general.py`: A more generalized model architecture with modular structure.

### 🚇 Shortcut Search

* `shortcut_dynamic_generate.py`: Generates shortcuts based on probabilistic negative log-weighted graphs.
* `shortcut_select.py`: Searches for useful shortcuts within historical trajectory data.

### 🏋️ Training Scripts

* `train.py`: Trains node embeddings on the original graph.
* `train_hierarchy.py`: Trains embeddings on a sparse, hierarchy-optimized graph.

### ✅ Validation

* `valid.py`: The origin version of the validation code.
* `valid_hierarchy.py`: Validation for the hierarchical training mode.

### 🔁 Other Utilities

* `Node2Vec.py`: The base Node2Vec implementation or wrapper.
* `compare.py`: Compares the shortest road and the most-likely path .
* `config.py`: Configuration settings.
* `__init__.py`: Package initializer.

---

## 🧩 Usage

To preprocess your dataset and train a model, follow the steps below:

```bash
# Step 1: Preprocess the dataset
python data_preprocess_for_map.py  # or other preprocess scripts depending on the task

# Step 2: Train the model
python train.py  # for the original graph
# or
python train_hierarchy.py  # for the sparse graph

# Step 3: Validate the trained model
python valid.py  # or valid_hierarchy.py
```

---

## 📌 Notes

* Make sure your input data is formatted correctly before preprocessing.
* Shortcut generation can significantly reduce training and inference time in large graphs.
* Supports flexible configurations through `config.py`.

---

## 📬 Contact

For questions or contributions, feel free to open an issue or submit a pull request.

---

