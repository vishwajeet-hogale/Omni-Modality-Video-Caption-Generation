# Multi-Modal Diffusion-Based Text Generation

This implementation extends the existing SwinBart model with advanced multi-modal diffusion capabilities for video caption generation.

## 🚀 Features

### **Multi-Modal ViT Architecture**

- **Enhanced Multi-Modal ViT**: Processes images, audio, and text through a unified Vision Transformer
- **Cross-Modal Attention**: Enables interaction between different modalities
- **Audio Tokenization**: Converts audio to tokens for ViT processing
- **Text Tokenization**: Handles text input for multi-modal fusion

### **Diffusion-Based Text Generation**

- **Diffusion Text Generator**: Uses diffusion models for controllable text generation
- **Noise Scheduling**: Implements learnable noise schedules for better generation
- **Timestep Embeddings**: Incorporates temporal information in the diffusion process

### **Masked Language Modeling**

- **Masked Language Model**: Predicts masked tokens for better text understanding
- **Dynamic Masking**: Supports configurable masking probabilities
- **Cross-Modal Context**: Uses image and audio features to predict masked text

## 📁 File Structure

```
models/
├── multimodal_vit.py              # Multi-modal ViT components
├── diffusion_text_generator.py   # Diffusion-based text generation
├── masked_language_model.py      # Masked language modeling
└── model.py                      # Updated SwinBart with diffusion integration

configs/
└── default.yaml                  # Updated configuration with new sections

train_multimodal_diffusion.py     # Training script for diffusion model
inference_multimodal_diffusion.py # Inference script with multiple methods
test_multimodal_diffusion.py     # Comprehensive test suite
```

## 🔧 Configuration

### **Enable Multi-Modal Diffusion**

```yaml
# In configs/default.yaml
use_multimodal_diffusion: true # Set to true to enable
```

### **Multi-Modal ViT Configuration**

```yaml
multimodal:
  hidden_dim: 768
  num_heads: 8
  num_fusion_layers: 4
  output_dim: 768

  image:
    hidden_dim: 768
    num_heads: 8
    num_layers: 6

  audio:
    audio_dim: 128
    hidden_dim: 768
    patch_size: 16
    kernel_size: 3

  text:
    hidden_dim: 768
    max_length: 512
    model_name: "bert-base-uncased"
```

### **Diffusion Configuration**

```yaml
diffusion:
  vocab_size: 30522
  hidden_dim: 768
  num_timesteps: 1000
  max_length: 60
  model_name: "bert-base-uncased"
```

### **Masked Language Model Configuration**

```yaml
masked_lm:
  vocab_size: 30522
  hidden_dim: 768
  max_length: 60
  mask_prob: 0.15
  model_name: "bert-base-uncased"
```

## 🚀 Usage

### **1. Training**

```bash
# Train with multi-modal diffusion
python train_multimodal_diffusion.py

# Train with specific config
python train_multimodal_diffusion.py --config-name=multimodal_diffusion
```

### **2. Inference**

```bash
# Run inference with multiple methods
python inference_multimodal_diffusion.py

# Test specific method
python inference_multimodal_diffusion.py method=diffusion
```

### **3. Testing**

```bash
# Run comprehensive tests
python test_multimodal_diffusion.py

# Test individual components
python -c "from test_multimodal_diffusion import test_multimodal_vit; test_multimodal_vit()"
```

## 🧪 Testing

The test suite includes:

1. **Multi-Modal ViT Tests**

   - Audio tokenization
   - Text tokenization
   - Cross-modal attention
   - ViT encoder

2. **Diffusion Generator Tests**

   - Forward pass
   - Sampling
   - Noise scheduling

3. **Masked Language Model Tests**

   - Token masking
   - Loss computation
   - Prediction generation

4. **Integration Tests**
   - SwinBart model integration
   - End-to-end functionality

## 🔄 Model Methods

### **Standard Generation**

```python
# Standard SwinBart generation
captions = model.generate(images, audio, max_length=60)
```

### **Diffusion Generation**

```python
# Diffusion-based generation
captions = model.generate_with_diffusion(images, audio, max_length=60)
```

### **Masked Language Modeling**

```python
# Masked language modeling
predictions, mask_positions = model.generate_with_masking(
    images, audio, text_tokens, mask_prob=0.15
)
```

### **Multi-Modal Forward Pass**

```python
# Enhanced forward pass with diffusion
loss, generated_captions = model.forward_with_diffusion(
    images, audio, captions, is_new_video=True
)
```

## 📊 Performance Metrics

The system tracks multiple metrics:

- **Standard Loss**: Original SwinBart loss
- **Diffusion Loss**: Diffusion model loss
- **Masked LM Loss**: Masked language model loss
- **Total Loss**: Combined weighted loss
- **Narrative Coherence**: Percentage of connective words in generated text

## 🎯 Key Benefits

1. **Enhanced Multi-Modal Understanding**: Better fusion of visual, audio, and text information
2. **Controllable Generation**: Diffusion models provide more control over text generation
3. **Better Text Quality**: Masked language modeling improves text understanding
4. **Flexible Architecture**: Modular design allows easy extension and modification
5. **Comprehensive Testing**: Full test suite ensures reliability

## 🔧 Customization

### **Adding New Modalities**

```python
# Extend the multi-modal ViT
class CustomModalityEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your custom encoder

    def forward(self, x):
        # Your custom processing
        return processed_features
```

### **Custom Diffusion Schedules**

```python
# Implement custom noise schedules
def custom_beta_schedule(num_timesteps):
    return torch.linspace(0.0001, 0.02, num_timesteps)
```

### **Custom Attention Mechanisms**

```python
# Implement custom attention
class CustomAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # Your custom attention implementation
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Memory Issues**: Reduce batch size or model dimensions
2. **Device Mismatches**: Ensure all tensors are on the same device
3. **Import Errors**: Check that all dependencies are installed
4. **Config Errors**: Verify configuration parameters match model expectations

### **Debug Mode**

```python
# Enable debug mode for detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Future Enhancements

1. **Advanced Diffusion Schedules**: Implement more sophisticated noise schedules
2. **Multi-Scale Attention**: Add attention at multiple scales
3. **Temporal Diffusion**: Extend diffusion to temporal sequences
4. **Cross-Modal Diffusion**: Implement cross-modal diffusion processes
5. **Efficient Inference**: Add model compression and quantization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Swin Transformer for vision encoding
- BART for text generation
- Hugging Face Transformers for pre-trained models
- PyTorch Lightning for training framework
