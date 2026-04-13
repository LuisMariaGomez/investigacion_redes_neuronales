from transformers import AutoTokenizer, AutoModel
import torch


class HFEmbedder:
    def __init__(self, model_name="microsoft/Multilingual-MiniLM-L12-H384"):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, texts, batch_size=32):
        import torch
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            inputs = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings).numpy()