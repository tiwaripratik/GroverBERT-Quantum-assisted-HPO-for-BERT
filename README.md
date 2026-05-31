# GroverBERT — Quantum-assisted HPO for BERT

This repository (GroverBERT) demonstrates using quantum-inspired optimization (QAOA/VQE/Grover) to tune hyperparameters for fine-tuning a BERT sequence classification model. The primary artifact is the Jupyter notebook [Quantum.ipynb](Quantum.ipynb) which contains experiments, toy QUBO formulations, and a full fine-tuning pipeline.

**Notebook overview**
- Environment and package installs (multiple `pip` calls; the notebook pins several Qiskit and Transformers versions).
- Data loading: expects a CSV with columns `question` and `topic` (the notebook refers to `/content/questions-data-new.csv`).
- Quantum optimization: builds QUBO problems, runs VQE / QAOA / Grover workflows to produce optimized continuous hyperparameters (learning rate, weight decay, adam_epsilon, max_grad_norm) and discrete choices (batch size, epochs, scheduler).
- Training: tokenization (`bert-base-uncased`), dataset preparation, `Trainer` training/evaluation using the quantum-optimized hyperparameters.
- Save/load: saves fine-tuned model and tokenizer to `./vqe_optimized_bert_final` (and similar folders).

**Key files**
- [Quantum.ipynb](Quantum.ipynb): main notebook with experiments and runnable cells.
- `.gitignore`, `.gitattributes`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE` — repository metadata and contribution guidance.

**Dependencies (recommended)**
The notebook installs several packages and pins specific versions in different cells. To reproduce a working environment, create a fresh virtualenv and install the following (matching the notebook choices):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "qiskit==1.2.4" "qiskit-algorithms==0.3.0" "qiskit-optimization==0.6.1" \
	"transformers==4.44.2" "datasets==2.19.1" "accelerate>=0.26.0" "evaluate==0.4.2" matplotlib scikit-learn optuna
```

Note: the notebook contains multiple alternative `pip` commands and uninstall/install sequences to work around Qiskit compatibility — match one consistent set above.

**Data**
Place your dataset CSV at `./questions-data-new.csv` (or update the notebook path). The CSV must contain `question` and `topic` columns. The notebook maps `topic` values to numeric labels internally.

**Run instructions**
1. Create and activate a Python virtual environment (see commands above).
2. Ensure the dataset CSV is available at the configured path.
3. Open `Quantum.ipynb` in Jupyter or VS Code and run cells sequentially. The notebook runs quantum optimizer steps then fine-tunes a BERT model using the derived hyperparameters.

Minimal commands to start:

```bash
jupyter lab  # or jupyter notebook
# open Quantum.ipynb and run
```

**Outputs**
- Fine-tuned model and tokenizer are saved to `./vqe_optimized_bert_final` (or `./quantum_annealed_bert_final` depending on the cell run).
- Training logs and evaluation metrics are printed in-cell; training artifacts are in the specified `output_dir` in `TrainingArguments`.

**Caveats & notes**
- Qiskit and related packages are version-sensitive; if you encounter import errors, try adjusting Qiskit package versions as shown in the notebook cells.
- The notebook contains toy/simulated functions (e.g., `bert_eval_loss`) and randomized placeholders — results are illustrative and may require adaptation for production experiments.

If you'd like, I can create a `requirements.txt` with the pinned versions shown above and/or run `git init` and make an initial commit for you.
