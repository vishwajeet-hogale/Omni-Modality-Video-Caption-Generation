from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
import torch.nn as nn 
import torch

class TextDecoder(nn.Module):
    def __init__(self, decoder_name: str):
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        self.bart_model = BartForConditionalGeneration.from_pretrained(decoder_name).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(decoder_name)
        self.projection_images_audio = nn.Linear(1536, 768)
    def forward(self, image_audio_embeddings, captions):
        # Tokenize captions
        tokenized = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=60)

        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        
        # Proper teacher forcing: decoder_input_ids excludes last token, labels exclude first token
        decoder_input_ids = input_ids[:, :-1].contiguous()  # Remove last token for decoder input
        decoder_attention_mask = attention_mask[:, :-1].contiguous()  # Corresponding attention mask
        labels = input_ids[:, 1:].contiguous()  # Remove first token (BOS) for labels
        
        seq_len = decoder_input_ids.shape[1]
        # print(f"Image/Audio embeddings shape: {image_audio_embeddings.shape}")

        # Project image/audio embeddings to match BART's hidden size
        projected_embeddings = self.projection_images_audio(image_audio_embeddings)
        # print(f"Projected embeddings shape: {projected_embeddings.shape}")
        
        # Create proper encoder outputs for BART - single token sequence representing the multimodal context
        encoder_hidden_states = projected_embeddings.unsqueeze(1)  # [B, 1, 768]
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # Forward through BART with proper teacher forcing
        outputs = self.bart_model(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
        )
        return outputs
    def generate(self, image_audio_embeddings, max_length=60, num_beams=4):
        # Ensure we're in eval mode for generation
        self.bart_model.eval()
        
        # Create decoder_input_ids with just the BOS token (for BART generation)
        batch_size = image_audio_embeddings.shape[0]
        decoder_input_ids = torch.full((batch_size, 1), self.tokenizer.bos_token_id, 
                                     dtype=torch.long, device=self.device)
        
        # Project and prepare encoder outputs
        projected_embeddings = self.projection_images_audio(image_audio_embeddings)
        encoder_hidden_states = projected_embeddings.unsqueeze(1)  # [B, 1, 768]
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        # Generate using BART's generate method
        with torch.no_grad():
            generated_ids = self.bart_model.generate(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        return generated_ids
