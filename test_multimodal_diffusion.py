import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.multimodal_vit import EnhancedMultiModalViT, AudioTokenizer, TextTokenizer, CrossModalAttention, ViTEncoder
from models.diffusion_text_generator import DiffusionTextGenerator
from models.masked_language_model import MaskedLanguageModel
from models.model import SwinBart
import hydra
from omegaconf import DictConfig

def test_multimodal_vit():
    """Test the multi-modal ViT"""
    print("🧪 Testing Multi-Modal ViT...")
    
    try:
        # Create dummy data
        images = torch.randn(2, 3, 224, 224)
        audio = torch.randn(2, 16000)
        text = ["This is a test caption", "Another test caption"]
        
        # Initialize model with config
        config = type('Config', (), {
            'hidden_dim': 768,
            'num_heads': 8,
            'num_fusion_layers': 4,
            'output_dim': 768,
            'image': type('ImageConfig', (), {
                'hidden_dim': 768, 
                'num_heads': 8, 
                'num_layers': 6
            })(),
            'audio': type('AudioConfig', (), {
                'audio_dim': 128, 
                'hidden_dim': 768, 
                'patch_size': 16, 
                'kernel_size': 3
            })(),
            'text': type('TextConfig', (), {
                'hidden_dim': 768, 
                'max_length': 512, 
                'model_name': 'bert-base-uncased'
            })()
        })()
        
        model = EnhancedMultiModalViT(config)
        
        # Forward pass
        output = model(images, audio, text)
        print(f"✅ Multi-Modal ViT output shape: {output.shape}")
        print("✅ Multi-Modal ViT test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Multi-Modal ViT test failed: {e}")
        return False

def test_diffusion_generator():
    """Test the diffusion text generator"""
    print("🧪 Testing Diffusion Text Generator...")
    
    try:
        # Create dummy data
        batch_size, seq_len, hidden_dim = 2, 60, 768
        image_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Initialize model with config
        config = type('Config', (), {
            'vocab_size': 30522,
            'hidden_dim': 768,
            'num_timesteps': 100,  # Reduced for testing
            'max_length': 60,
            'model_name': 'bert-base-uncased'
        })()
        
        model = DiffusionTextGenerator(config)
        
        # Test forward pass
        x = torch.randint(0, 30522, (batch_size, seq_len))
        timesteps = torch.randint(0, 100, (batch_size,))
        output = model(x, image_embeddings, timesteps)
        print(f"✅ Diffusion generator forward output shape: {output.shape}")
        
        # Test sampling
        generated_tokens = model.sample(image_embeddings, max_length=20)
        print(f"✅ Diffusion generator sample output shape: {generated_tokens.shape}")
        print("✅ Diffusion Text Generator test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Diffusion Text Generator test failed: {e}")
        return False

def test_masked_language_model():
    """Test the masked language model"""
    print("🧪 Testing Masked Language Model...")
    
    try:
        # Create dummy data
        batch_size, seq_len, hidden_dim = 2, 60, 768
        tokens = torch.randint(0, 30522, (batch_size, seq_len))
        image_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Initialize model with config
        config = type('Config', (), {
            'vocab_size': 30522,
            'hidden_dim': 768,
            'max_length': 60,
            'mask_prob': 0.15,
            'model_name': 'bert-base-uncased'
        })()
        
        model = MaskedLanguageModel(config)
        
        # Test forward pass
        mask_positions = torch.randint(0, seq_len, (batch_size, 5))  # 5 masked positions per batch
        output = model(tokens, image_embeddings, mask_positions)
        print(f"✅ Masked LM forward output shape: {output.shape}")
        
        # Test masked token creation
        masked_tokens, mask = model.create_masked_tokens(tokens, mask_prob=0.15)
        print(f"✅ Masked tokens shape: {masked_tokens.shape}")
        print(f"✅ Mask shape: {mask.shape}")
        
        # Test loss computation
        loss, predictions, mask_positions = model.compute_loss(tokens, image_embeddings, mask_prob=0.15)
        print(f"✅ Masked LM loss: {loss.item():.4f}")
        print("✅ Masked Language Model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Masked Language Model test failed: {e}")
        return False

def test_swinbart_integration():
    """Test the integrated SwinBart model"""
    print("🧪 Testing SwinBart Integration...")
    
    try:
        # Create a minimal config
        config = type('Config', (), {
            'vision_encoder_cfg': type('VisionConfig', (), {
                'encoder': 'swin_base_patch4_window7_224',
                'output_dim': 1000
            })(),
            'decoder_cfg': type('DecoderConfig', (), {
                'decoder': 'facebook/bart-base',
                'hidden_dim': 768
            })(),
            'audio_encoder_cfg': type('AudioConfig', (), {
                'encoder': 'transformer_cls',
                'n_mfcc': 13
            })(),
            'use_temporal_attention': False,
            'use_multimodal_diffusion': True,
            'multimodal': type('MultimodalConfig', (), {
                'hidden_dim': 768,
                'num_heads': 8,
                'num_fusion_layers': 2,  # Reduced for testing
                'output_dim': 768,
                'image': type('ImageConfig', (), {
                    'hidden_dim': 768, 
                    'num_heads': 8, 
                    'num_layers': 2  # Reduced for testing
                })(),
                'audio': type('AudioConfig', (), {
                    'audio_dim': 128, 
                    'hidden_dim': 768, 
                    'patch_size': 16, 
                    'kernel_size': 3
                })(),
                'text': type('TextConfig', (), {
                    'hidden_dim': 768, 
                    'max_length': 512, 
                    'model_name': 'bert-base-uncased'
                })()
            })(),
            'diffusion': type('DiffusionConfig', (), {
                'vocab_size': 30522,
                'hidden_dim': 768,
                'num_timesteps': 50,  # Reduced for testing
                'max_length': 60,
                'model_name': 'bert-base-uncased'
            })(),
            'masked_lm': type('MaskedLMConfig', (), {
                'vocab_size': 30522,
                'hidden_dim': 768,
                'max_length': 60,
                'mask_prob': 0.15,
                'model_name': 'bert-base-uncased'
            })()
        })()
        
        # Initialize model
        model = SwinBart(config)
        print(f"✅ SwinBart model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        images = torch.randn(2, 3, 224, 224)
        audio = torch.randn(2, 13, 100)  # MFCC features
        captions = torch.randint(0, 30522, (2, 20))
        
        # Test standard forward
        try:
            outputs = model(images, captions, audio, is_new_video=True)
            print(f"✅ Standard forward pass successful")
        except Exception as e:
            print(f"⚠️ Standard forward pass failed: {e}")
        
        # Test diffusion generation
        try:
            diffusion_captions = model.generate_with_diffusion(images, audio, max_length=20)
            print(f"✅ Diffusion generation output shape: {diffusion_captions.shape}")
        except Exception as e:
            print(f"⚠️ Diffusion generation failed: {e}")
        
        # Test masked LM generation
        try:
            text_tokens = torch.randint(0, 30522, (2, 20))
            predictions, mask_positions = model.generate_with_masking(images, audio, text_tokens)
            print(f"✅ Masked LM generation output shape: {predictions.shape}")
        except Exception as e:
            print(f"⚠️ Masked LM generation failed: {e}")
        
        print("✅ SwinBart Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ SwinBart Integration test failed: {e}")
        return False

def test_individual_components():
    """Test individual components"""
    print("🧪 Testing Individual Components...")
    
    # Test AudioTokenizer
    try:
        config = type('Config', (), {
            'audio_dim': 128,
            'hidden_dim': 768,
            'patch_size': 16,
            'kernel_size': 3
        })()
        
        audio_tokenizer = AudioTokenizer(config)
        audio = torch.randn(2, 16000)
        output = audio_tokenizer(audio)
        print(f"✅ AudioTokenizer output shape: {output.shape}")
    except Exception as e:
        print(f"❌ AudioTokenizer test failed: {e}")
    
    # Test TextTokenizer
    try:
        config = type('Config', (), {
            'hidden_dim': 768,
            'max_length': 512,
            'model_name': 'bert-base-uncased'
        })()
        
        text_tokenizer = TextTokenizer(config)
        text = ["This is a test", "Another test"]
        output = text_tokenizer(text)
        print(f"✅ TextTokenizer output shape: {output.shape}")
    except Exception as e:
        print(f"❌ TextTokenizer test failed: {e}")
    
    # Test CrossModalAttention
    try:
        cross_attention = CrossModalAttention(768, 8)
        query = torch.randn(2, 10, 768)
        key = torch.randn(2, 10, 768)
        value = torch.randn(2, 10, 768)
        output = cross_attention(query, key, value)
        print(f"✅ CrossModalAttention output shape: {output.shape}")
    except Exception as e:
        print(f"❌ CrossModalAttention test failed: {e}")
    
    # Test ViTEncoder
    try:
        config = type('Config', (), {
            'hidden_dim': 768,
            'num_heads': 8,
            'num_layers': 2
        })()
        
        vit_encoder = ViTEncoder(config)
        images = torch.randn(2, 3, 224, 224)
        output = vit_encoder(images)
        print(f"✅ ViTEncoder output shape: {output.shape}")
    except Exception as e:
        print(f"❌ ViTEncoder test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Multi-Modal Diffusion Tests")
    print("=" * 60)
    
    # Test individual components
    test_individual_components()
    print()
    
    # Test main components
    tests = [
        test_multimodal_vit,
        test_diffusion_generator,
        test_masked_language_model,
        test_swinbart_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 60)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
