# Molecular Property Prediction - Inference Guide

## Overview

The inference pipeline allows you to predict molecular toxicity properties for any molecule using trained models. The system can predict across 12 different toxicity endpoints from the Tox21 dataset.

## Quick Start

### Single Molecule Prediction

```bash
python predict.py model_checkpoint.pt --smiles "CCO"
```

### Batch Prediction

```bash
python predict.py model_checkpoint.pt --batch-smiles "CCO" "c1ccccc1" "CC(=O)O"
```

### File-based Prediction

```bash
python predict.py model_checkpoint.pt --input-file molecules.txt --output-file results.csv
```

## Prediction Tasks

The model predicts probabilities for these 12 toxicity endpoints:

### Nuclear Receptor (NR) Assays:
- **NR-AR**: Androgen Receptor - hormone disruption
- **NR-AR-LBD**: Androgen Receptor Ligand Binding Domain
- **NR-AhR**: Aryl hydrocarbon Receptor - xenobiotic metabolism
- **NR-Aromatase**: Aromatase enzyme - steroid hormone synthesis
- **NR-ER**: Estrogen Receptor - hormone disruption
- **NR-ER-LBD**: Estrogen Receptor Ligand Binding Domain  
- **NR-PPAR-gamma**: Peroxisome Proliferator-Activated Receptor gamma

### Stress Response (SR) Assays:
- **SR-ARE**: Antioxidant Response Element - oxidative stress
- **SR-ATAD5**: DNA damage checkpoint
- **SR-HSE**: Heat Shock Element - cellular stress response
- **SR-MMP**: Mitochondrial Membrane Potential - cellular toxicity
- **SR-p53**: p53 tumor suppressor pathway - DNA damage

## Example Results

### Safe Molecule: Ethanol (CCO)
```
SMILES: CCO
  Predictions (probability):
    NR-AR          : 0.0402 (NEGATIVE) - Not androgenic
    NR-ER          : 0.2214 (NEGATIVE) - Not estrogenic  
    SR-p53         : 0.0864 (NEGATIVE) - No DNA damage
    ... (all predictions < 0.5, indicating low toxicity)
```

### Toxic Molecule: Estradiol (Natural Estrogen)
```
SMILES: C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O
  Predictions (probability):
    NR-ER          : 0.9212 (POSITIVE) - Strong estrogen receptor binding!
    NR-ER-LBD      : 0.8948 (POSITIVE) - Binds to ER ligand binding domain
    SR-MMP         : 0.5617 (POSITIVE) - May affect mitochondrial function
    ... (other assays mostly negative, showing specificity)
```

## Command Line Usage

### Basic Options

```bash
python predict.py CHECKPOINT [options]

Required:
  CHECKPOINT                    Path to trained model checkpoint

Input (choose one):
  --smiles SMILES              Single SMILES string
  --batch-smiles SMILES [...]  Multiple SMILES strings  
  --input-file FILE            File with SMILES (one per line)

Output:
  --output-file FILE           Save results to CSV file
  --format {table,json,csv}    Output format (default: table)
  --threshold FLOAT            Classification threshold (default: 0.5)

System:
  --device {auto,cpu,cuda}     Computing device (default: auto)
```

### Advanced Examples

**Test multiple molecules with JSON output:**
```bash
python predict.py model.pt \\
  --batch-smiles "CCO" "c1ccccc1" "C=O" \\
  --format json --threshold 0.3
```

**Process a file with custom threshold:**
```bash
python predict.py model.pt \\
  --input-file compounds.txt \\
  --output-file predictions.csv \\
  --threshold 0.3
```

**Force CPU usage:**
```bash
python predict.py model.pt --smiles "CCO" --device cpu
```

## Input File Format

For `--input-file`, create a text file with one SMILES string per line:

```
CCO
c1ccccc1
CC(=O)O
C1=CC=C(C=C1)O
C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O
```

## Output Formats

### Table Format (default)
```
SMILES: CCO
  Predictions (probability):
    NR-AR          : 0.0402 (NEGATIVE)
    NR-ER          : 0.2214 (NEGATIVE)
    ...
```

### JSON Format
```json
[
  {
    "smiles": "CCO",
    "predictions": {
      "NR-AR": 0.0402,
      "NR-ER": 0.2214,
      ...
    }
  }
]
```

### CSV Format
```csv
smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
CCO,0.0402,0.0314,0.0448,0.0140,0.2214,0.0524,0.0773,0.1992,0.0684,0.1718,0.1207,0.0864
```

## Programmatic Usage

You can also use the inference capabilities directly in Python:

```python
from molprop.inference import MolecularPredictor

# Load model
predictor = MolecularPredictor("path/to/model.pt")

# Single prediction
result = predictor.predict_smiles("CCO")
print(f"Estrogen receptor binding: {result['NR-ER']:.3f}")

# Batch prediction
molecules = ["CCO", "c1ccccc1", "C=O"]
results = predictor.predict_batch(molecules)

# File prediction
df = predictor.predict_file("input.txt", "output.csv")
```

## Model Information

Check what tasks your model can predict:

```python
predictor = MolecularPredictor("model.pt")
info = predictor.get_model_info()
print(f"Tasks: {info['task_names']}")
print(f"Device: {info['device']}")
print(f"Hidden dimensions: {info['hidden_dim']}")
```

## Interpretation Guidelines

### Probability Thresholds:
- **0.0-0.3**: Low probability of toxicity
- **0.3-0.7**: Moderate probability - requires further investigation
- **0.7-1.0**: High probability of toxicity

### Understanding Results:
- **POSITIVE**: Probability > threshold (default 0.5)
- **NEGATIVE**: Probability ≤ threshold
- **Multiple positives**: Molecule may have multiple toxicity mechanisms
- **All negatives**: Generally safer molecule (but not guaranteed safe)

### Biological Context:
- **Nuclear Receptor assays**: Hormone disruption, endocrine effects
- **Stress Response assays**: Cellular damage, oxidative stress, DNA damage
- **High specificity**: Few positive assays may indicate targeted effects
- **Broad toxicity**: Many positive assays suggest general cellular toxicity

## Error Handling

### Invalid SMILES:
```
SMILES: INVALID_SMILES
  ERROR: Invalid molecule
```

### Empty Results:
If all molecules are invalid, the script will still run but show error messages for each invalid SMILES.

## Performance Notes

- **CPU vs GPU**: GPU provides faster inference for large batches
- **Batch size**: Larger batches are more efficient than individual predictions
- **Memory**: Each molecule requires ~1-10KB depending on size
- **Speed**: ~10-100 molecules/second depending on hardware

## Troubleshooting

### Common Issues:

1. **Model loading errors**: Check checkpoint path and format
2. **SMILES parsing errors**: Verify SMILES strings are valid
3. **Memory issues**: Use smaller batches or switch to CPU
4. **Dependency errors**: Ensure all packages are installed

### Getting Help:

```bash
python predict.py --help
```

## Model Versions

Different trained models may have different performance characteristics:

- **Quick test models**: Fast training, lower accuracy
- **Default models**: Balanced training time and performance  
- **Big models**: Longer training, higher accuracy
- **Specialized models**: Trained on specific datasets or tasks

Always check the model's training configuration and validation performance before making critical decisions based on predictions.