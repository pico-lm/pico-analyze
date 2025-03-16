# Pico Analyze

Framework for analyzing learning dynamics of models trained using Pico. This package provides tools and metrics for understanding how models evolve during training.

Pico Analysis is designed to work seamlessly with models trained using the Pico framework. It provides:

- Component-based analysis of model internals
- Configurable metrics for learning dynamics
- Support for comparative analysis across training checkpoints
- Extensible architecture for custom metrics and components

## 🚀 Quick Start

```bash
# Initial Installation 

git clone https://github.com/your-org/pico-analysis.git
cd pico-analysis
source setup.sh

# Run Analysis
poetry run analyze --config_path configs/demo.yaml --repo_id pico-lm/demo --branch demo-1

```

## 🔑 Key Concepts

Components are the building blocks of analysis, representing specific parts of the model. There
are two main types of components: 

#### Simple Components
- Individual model elements:
  - Weight matrices
  - Activation values
  - Gradient tensors

#### Compound Components
- Combinations of simple components:
  - OV-Circuits (combining value and output projections)
  - Induction heads
  - Attention heads
  - Feed-forward blocks

### Metrics

Metrics are computations performed on components across different checkpoints:

#### Comparative Metrics
- Compare components across different training steps:
  - CKA (Centered Kernel Alignment)
  - Gradient Similarity

#### Single-checkpoint Metrics
- Analyze components at specific points:
  - Activation statistics
  - Weight distributions
 

### Configuration

Analysis is configured through YAML configuration files. For example:

```yaml
# configs/demo.yaml
metrics: 
  - metric_name: cka
    target_checkpoint: 100
    data_split: "val"
    components: 
      - component_name: ov_circuit
        layer_suffixes: 
          output_layer: "attention.o_proj"
          value_layer: "attention.v_proj"
        layers: [0,1,2,3,4,5,6,7,8,9,10,11]
      - component_name: simple
        data_type: "weights"
        layer_suffixes: "swiglu.w_2"
        layers: [0,1,2,3,4,5,6,7,8,9,10,11]
  - metric_name: per
    data_split: "train"
    components: 
      - component_name: simple
        data_type: "weights"
        layer_suffixes: "swiglu.w_2"
        layers: [0,1,2,3,4,5,6,7,8,9,10,11]
      
steps: 
  start: 0
  end: 100
  step: 50
```

## 🤝 Contributing

We welcome contributions in:
- New features, including new metrics and components
- Bug reports 
- Expanded Documentationk 


## 📝 License

Apache 2.0 License

## 📫 Contact

- GitHub: [rdiehlmartinez/pico](https://github.com/rdiehlmartinez/pico)
- Author: [Richard Diehl Martinez](https://richarddiehlmartinez.com)

## Citation

If you use Pico in your research, please cite:

```bibtex
@software{pico2024,
    author = {Diehl Martinez, Richard},
    title = {Pico: Framework for Training Tiny Language Models},
    year = {2024},
}
```
