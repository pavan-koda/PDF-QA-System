"""
Interactive Model Selector and Downloader
Allows users to select and download AI models on first run
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "vision_models": {
        "llama3.2-vision": {
            "name": "Llama 3.2-Vision 11B (Recommended for PDFs)",
            "size": "~7GB via Ollama",
            "speed": "Medium",
            "quality": "Exceptional",
            "description": "Best for PDFs with diagrams, charts, images - understands visual layout",
            "default": True,
            "requires_ollama": True,
            "ollama_command": "ollama pull llama3.2-vision:11b"
        },
        "llama3.2-vision-90b": {
            "name": "Llama 3.2-Vision 90B",
            "size": "~55GB via Ollama",
            "speed": "Slow",
            "quality": "Best-in-class",
            "description": "Ultimate quality for complex documents (requires 64GB+ RAM)",
            "requires_ollama": True,
            "ollama_command": "ollama pull llama3.2-vision:90b"
        },
        "colpali-only": {
            "name": "ColPali Visual Retriever Only",
            "size": "~1GB",
            "speed": "Very Fast",
            "quality": "Good for retrieval",
            "description": "Fast visual retrieval without LLM - good for search only"
        }
    },
    "embeddings": {
        "all-MiniLM-L6-v2": {
            "name": "MiniLM (Recommended)",
            "size": "~80MB",
            "speed": "Fast",
            "quality": "Good",
            "description": "Lightweight and fast, perfect for most use cases",
            "default": True
        },
        "all-mpnet-base-v2": {
            "name": "MPNet",
            "size": "~420MB",
            "speed": "Medium",
            "quality": "Excellent",
            "description": "Higher quality embeddings, better semantic understanding"
        },
        "all-distilroberta-v1": {
            "name": "DistilRoBERTa",
            "size": "~290MB",
            "speed": "Fast",
            "quality": "Very Good",
            "description": "Balanced speed and quality, distilled RoBERTa"
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "name": "Multilingual MiniLM",
            "size": "~420MB",
            "speed": "Medium",
            "quality": "Good",
            "description": "Supports 50+ languages for multilingual documents"
        },
        "sentence-transformers/multi-qa-mpnet-base-dot-v1": {
            "name": "Multi-QA MPNet",
            "size": "~420MB",
            "speed": "Medium",
            "quality": "Excellent",
            "description": "Optimized for question-answering tasks"
        },
        "BAAI/bge-small-en-v1.5": {
            "name": "BGE Small",
            "size": "~130MB",
            "speed": "Fast",
            "quality": "Very Good",
            "description": "High-performance Chinese model, also good for English"
        },
        "BAAI/bge-base-en-v1.5": {
            "name": "BGE Base",
            "size": "~420MB",
            "speed": "Medium",
            "quality": "Excellent",
            "description": "State-of-the-art embedding model for retrieval"
        }
    },
    "qa_model": {
        "extractive": {
            "name": "Smart Extractive (Recommended)",
            "size": "0MB (no download)",
            "speed": "Very Fast",
            "quality": "Excellent for structured docs",
            "description": "Uses regex patterns to extract info - best for invoices/bills",
            "default": True,
            "offline": True
        },
        "distilbert-base-cased-distilled-squad": {
            "name": "DistilBERT",
            "size": "~250MB",
            "speed": "Medium",
            "quality": "Very Good",
            "description": "AI-powered QA, good for complex questions"
        },
        "deepset/roberta-base-squad2": {
            "name": "RoBERTa",
            "size": "~500MB",
            "speed": "Slower",
            "quality": "Excellent",
            "description": "Most accurate, best for research papers"
        },
        "bert-large-uncased-whole-word-masking-finetuned-squad": {
            "name": "BERT Large",
            "size": "~1.3GB",
            "speed": "Slow",
            "quality": "Excellent",
            "description": "Large BERT model for high accuracy QA"
        },
        "deepset/electra-base-squad2": {
            "name": "ELECTRA Base",
            "size": "~400MB",
            "speed": "Medium",
            "quality": "Very Good",
            "description": "ELECTRA model optimized for question answering"
        }
    },
    "generator": {
        "none": {
            "name": "No Generator (Recommended)",
            "size": "0MB",
            "speed": "N/A",
            "quality": "N/A",
            "description": "Use extractive answers only - more accurate",
            "default": True,
            "offline": True
        },
        "gpt2": {
            "name": "GPT-2 Small",
            "size": "~500MB",
            "speed": "Fast",
            "quality": "Basic",
            "description": "Lightweight text generation (117M parameters)"
        },
        "gpt2-medium": {
            "name": "GPT-2 Medium",
            "size": "~1.5GB",
            "speed": "Medium",
            "quality": "Good",
            "description": "Better text generation (345M parameters)"
        },
        "gpt2-large": {
            "name": "GPT-2 Large",
            "size": "~3GB",
            "speed": "Slower",
            "quality": "Very Good",
            "description": "High-quality text generation (774M parameters)"
        },
        "gpt2-xl": {
            "name": "GPT-2 XL",
            "size": "~6GB",
            "speed": "Very Slow",
            "quality": "Excellent",
            "description": "Best GPT-2 model (1.5B parameters, requires GPU)"
        },
        "facebook/opt-350m": {
            "name": "OPT-350M",
            "size": "~700MB",
            "speed": "Fast",
            "quality": "Good",
            "description": "Meta's OPT model - efficient alternative to GPT-2"
        },
        "facebook/opt-1.3b": {
            "name": "OPT-1.3B",
            "size": "~2.5GB",
            "speed": "Medium",
            "quality": "Very Good",
            "description": "Larger OPT model for better generation"
        },
        "google/flan-t5-small": {
            "name": "FLAN-T5 Small",
            "size": "~300MB",
            "speed": "Fast",
            "quality": "Good",
            "description": "Google's instruction-tuned model - good for QA"
        },
        "google/flan-t5-base": {
            "name": "FLAN-T5 Base",
            "size": "~900MB",
            "speed": "Medium",
            "quality": "Very Good",
            "description": "Better FLAN-T5 model - excellent for answering"
        },
        "google/flan-t5-large": {
            "name": "FLAN-T5 Large",
            "size": "~3GB",
            "speed": "Slower",
            "quality": "Excellent",
            "description": "Large instruction-tuned model - best for QA tasks"
        },
        "microsoft/phi-2": {
            "name": "Phi-2",
            "size": "~5.5GB",
            "speed": "Medium",
            "quality": "Excellent",
            "description": "Microsoft's 2.7B model - high quality reasoning"
        },
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "name": "TinyLlama Chat",
            "size": "~2.2GB",
            "speed": "Fast",
            "quality": "Good",
            "description": "Compact Llama-based chat model (1.1B parameters)"
        },
        "stabilityai/stablelm-2-zephyr-1_6b": {
            "name": "StableLM Zephyr 1.6B",
            "size": "~3GB",
            "speed": "Medium",
            "quality": "Very Good",
            "description": "Stability AI's instruction-tuned model"
        },
        "google/gemma-2b": {
            "name": "Gemma 2B",
            "size": "~5GB",
            "speed": "Fast",
            "quality": "Very Good",
            "description": "Google's lightweight Gemma model (2B parameters)"
        },
        "google/gemma-7b": {
            "name": "Gemma 7B",
            "size": "~14GB",
            "speed": "Slow",
            "quality": "Excellent",
            "description": "Google's full Gemma model (7B parameters, requires GPU)"
        },
        "google/gemma-2b-it": {
            "name": "Gemma 2B Instruct",
            "size": "~5GB",
            "speed": "Fast",
            "quality": "Excellent",
            "description": "Instruction-tuned Gemma 2B, best for QA tasks"
        },
        "google/gemma-7b-it": {
            "name": "Gemma 7B Instruct",
            "size": "~14GB",
            "speed": "Slow",
            "quality": "Excellent",
            "description": "Instruction-tuned Gemma 7B, premium quality (requires 16GB+ VRAM)"
        },
        "google/gemma-2-2b": {
            "name": "Gemma 2 2B",
            "size": "~5GB",
            "speed": "Fast",
            "quality": "Very Good",
            "description": "Gemma 2 generation, improved performance (2B parameters)"
        },
        "google/gemma-2-9b": {
            "name": "Gemma 2 9B",
            "size": "~18GB",
            "speed": "Slow",
            "quality": "Excellent",
            "description": "Gemma 2 generation, state-of-the-art (requires 20GB+ VRAM)"
        },
        "google/gemma-2-2b-it": {
            "name": "Gemma 2 2B Instruct",
            "size": "~5GB",
            "speed": "Fast",
            "quality": "Excellent",
            "description": "Gemma 2 instruction-tuned 2B, excellent for QA tasks"
        },
        "google/gemma-2-9b-it": {
            "name": "Gemma 2 9B Instruct",
            "size": "~18GB",
            "speed": "Slow",
            "quality": "Excellent",
            "description": "Gemma 2 instruction-tuned 9B, best quality (requires 20GB+ VRAM)"
        },
        "google/gemma-2-27b-it": {
            "name": "Gemma 2 27B Instruct",
            "size": "~54GB",
            "speed": "Very Slow",
            "quality": "Exceptional",
            "description": "Gemma 2 largest model, top-tier quality (requires 48GB+ VRAM)"
        },
        "google/gemma-3-4b": {
            "name": "Gemma 3 4B",
            "size": "~8GB",
            "speed": "Fast",
            "quality": "Excellent",
            "description": "Gemma 3 compact model, latest generation (4B parameters)"
        },
        "google/gemma-3-12b": {
            "name": "Gemma 3 12B",
            "size": "~24GB",
            "speed": "Medium",
            "quality": "Exceptional",
            "description": "Gemma 3 mid-size, cutting-edge quality (requires 24GB+ VRAM)"
        },
        "google/gemma-3-4b-it": {
            "name": "Gemma 3 4B Instruct",
            "size": "~8GB",
            "speed": "Fast",
            "quality": "Exceptional",
            "description": "Gemma 3 instruction-tuned 4B, best-in-class for QA (recommended)"
        },
        "google/gemma-3-12b-it": {
            "name": "Gemma 3 12B Instruct",
            "size": "~24GB",
            "speed": "Medium",
            "quality": "Exceptional",
            "description": "Gemma 3 instruction-tuned 12B, premium quality (requires 24GB+ VRAM)"
        },
        "meta-llama/Llama-2-7b-chat-hf": {
            "name": "Llama 2 7B Chat",
            "size": "~13GB",
            "speed": "Slow",
            "quality": "Excellent",
            "description": "Meta's Llama 2 chat model (requires HuggingFace auth & 16GB VRAM)"
        },
        "meta-llama/Llama-2-13b-chat-hf": {
            "name": "Llama 2 13B Chat",
            "size": "~26GB",
            "speed": "Very Slow",
            "quality": "Excellent",
            "description": "Larger Llama 2 chat (requires HF auth & 24GB+ VRAM)"
        },
        "openai/gpt-oss-20b": {
            "name": "GPT-OSS 20B",
            "size": "~40GB",
            "speed": "Medium",
            "quality": "Exceptional",
            "description": "OpenAI's open-weight model with reasoning capabilities (21B params)"
        },
        "openai/gpt-oss-120b": {
            "name": "GPT-OSS 120B",
            "size": "~240GB",
            "speed": "Very Slow",
            "quality": "Exceptional",
            "description": "OpenAI's largest open model with advanced reasoning (117B params, requires 80GB GPU)"
        },
        "mistralai/Mistral-7B-Instruct-v0.3": {
            "name": "Mistral 7B Instruct",
            "size": "~14GB",
            "speed": "Medium",
            "quality": "Excellent",
            "description": "Mistral AI's instruction-tuned model, excellent for QA (7B params)"
        },
        "mistralai/Mistral-12B-Instruct-v0.3": {
            "name": "Mistral 12B Instruct",
            "size": "~24GB",
            "speed": "Slower",
            "quality": "Exceptional",
            "description": "Mistral AI's larger instruction model, top-tier reasoning (12B params)"
        },
        "microsoft/Phi-3-mini-4k-instruct": {
            "name": "Phi-3 Mini 4K",
            "size": "~7.5GB",
            "speed": "Fast",
            "quality": "Excellent",
            "description": "Microsoft's compact model, strong reasoning (3.8B params, 4K context)"
        },
        "microsoft/Phi-3.5-mini-instruct": {
            "name": "Phi-3.5 Mini",
            "size": "~7.5GB",
            "speed": "Fast",
            "quality": "Excellent",
            "description": "Phi-3.5 Mini with improved reasoning (3.8B params, 128K context)"
        },
        "Qwen/Qwen2.5-7B-Instruct": {
            "name": "Qwen 2.5 7B Instruct",
            "size": "~14GB",
            "speed": "Medium",
            "quality": "Excellent",
            "description": "Alibaba's Qwen model, strong multilingual QA (7B params)"
        },
        "Qwen/Qwen2.5-14B-Instruct": {
            "name": "Qwen 2.5 14B Instruct",
            "size": "~28GB",
            "speed": "Slower",
            "quality": "Exceptional",
            "description": "Qwen 2.5 larger model, excellent reasoning (14B params)"
        }
    }
}


class ModelSelector:
    def __init__(self):
        self.config_file = Path("model_config.json")
        self.cache_dir = Path.home() / ".cache" / "huggingface"

    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is already downloaded."""
        if model_id in ["extractive", "none"]:
            return True

        # Check in local directory first
        if Path(model_id).exists():
            return True

        # Check in HuggingFace cache
        model_name = model_id.replace("/", "--")
        cache_paths = [
            self.cache_dir / "hub" / f"models--{model_name}",
            Path(model_id)
        ]

        return any(p.exists() for p in cache_paths)

    def load_saved_config(self) -> Dict:
        """Load previously saved configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_config(self, config: Dict):
        """Save model configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def print_header(self):
        """Print welcome header."""
        print("\n" + "="*70)
        print("  PDF Q&A SYSTEM - MODEL CONFIGURATION".center(70))
        print("="*70 + "\n")

    def print_model_info(self, category: str, model_id: str, info: Dict, index: int):
        """Print information about a model."""
        downloaded = self.is_model_downloaded(model_id)
        status = "✓ DOWNLOADED" if downloaded else "⬇ NEEDS DOWNLOAD"
        default = " (DEFAULT)" if info.get("default") else ""

        print(f"\n[{index}] {info['name']}{default}")
        print(f"    Status: {status}")
        print(f"    Size: {info['size']} | Speed: {info['speed']} | Quality: {info['quality']}")
        print(f"    {info['description']}")

    def select_model(self, category: str, category_name: str) -> str:
        """Interactive model selection for a category."""
        models = MODELS[category]

        print(f"\n{'─'*70}")
        print(f"SELECT {category_name.upper()}:")
        print(f"{'─'*70}")

        model_list = list(models.items())

        # Print all options
        for i, (model_id, info) in enumerate(model_list, 1):
            self.print_model_info(category, model_id, info, i)

        # Find default
        default_idx = next((i for i, (_, info) in enumerate(model_list, 1)
                          if info.get("default")), 1)

        print(f"\n{'─'*70}")

        while True:
            choice = input(f"Enter choice [1-{len(model_list)}] (default: {default_idx}): ").strip()

            if not choice:
                choice = str(default_idx)

            if choice.isdigit() and 1 <= int(choice) <= len(model_list):
                selected_id = model_list[int(choice) - 1][0]
                selected_info = model_list[int(choice) - 1][1]

                print(f"\n✓ Selected: {selected_info['name']}")
                return selected_id
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(model_list)}")

    def delete_old_generator(self, old_model_id: str):
        """Delete old generator model to save space."""
        if old_model_id in ["none", "extractive"]:
            return  # Nothing to delete

        # Delete local directory
        local_dir = Path(f"./{old_model_id.replace('/', '--')}")
        if local_dir.exists():
            import shutil
            try:
                shutil.rmtree(local_dir)
                print(f"✓ Deleted old generator model: {local_dir}")
            except Exception as e:
                print(f"⚠ Could not delete {local_dir}: {str(e)}")

    def download_model(self, model_id: str, model_type: str):
        """Download a model if needed."""
        if model_id in ["extractive", "none"]:
            return True

        if self.is_model_downloaded(model_id):
            print(f"✓ Model already downloaded: {model_id}")
            return True

        print(f"\n{'='*70}")
        print(f"DOWNLOADING: {model_id}")
        print(f"{'='*70}\n")

        try:
            if model_type == "embeddings":
                from sentence_transformers import SentenceTransformer
                print("Downloading embedding model...")
                model = SentenceTransformer(model_id)
                print(f"✓ Successfully downloaded: {model_id}")
                return True

            elif model_type == "qa_model" and model_id != "extractive":
                from transformers import pipeline
                print("Downloading QA model...")
                qa_pipeline = pipeline("question-answering", model=model_id)
                print(f"✓ Successfully downloaded: {model_id}")
                return True

            elif model_type == "generator" and model_id != "none":
                from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
                import shutil
                import torch

                print("Downloading generator model...")
                print(f"Note: This may take several minutes for large models...")

                # Define save directory (but don't create yet)
                save_dir = Path(f"./{model_id.replace('/', '--')}")

                try:
                    # Special handling for Gemma 3 models (use pipeline method)
                    if 'gemma-3' in model_id.lower():
                        print("Detected Gemma 3 model - using optimized pipeline download...")

                        # Determine device and dtype
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

                        print(f"Using device: {device}")
                        print("Downloading via pipeline (this may take several minutes)...")

                        # Download using pipeline method (downloads to cache automatically)
                        pipe = pipeline(
                            "text-generation",  # Use text-generation for text-only QA
                            model=model_id,
                            device=device,
                            torch_dtype=torch_dtype,
                            trust_remote_code=True
                        )

                        # Save to local directory
                        save_dir.mkdir(exist_ok=True)
                        print("Saving model locally...")
                        pipe.tokenizer.save_pretrained(save_dir)
                        pipe.model.save_pretrained(save_dir)

                        print(f"✓ Successfully downloaded Gemma 3 to: {save_dir}")
                        return True

                    # Check if it's a T5/FLAN-T5 model (seq2seq)
                    elif 't5' in model_id.lower() or 'flan' in model_id.lower():
                        print("Loading tokenizer...")
                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        print("Loading model (this may take a while)...")
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

                        # Only create directory after successful download
                        save_dir.mkdir(exist_ok=True)
                        print("Saving model locally...")
                        tokenizer.save_pretrained(save_dir)
                        model.save_pretrained(save_dir)

                        print(f"✓ Successfully downloaded to: {save_dir}")
                        return True

                    else:
                        # For GPT-2, OPT, Llama, Phi, StableLM, Gemma 1/2 (causal LM)
                        print("Loading tokenizer...")
                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        print("Loading model (this may take a while)...")
                        model = AutoModelForCausalLM.from_pretrained(model_id)

                        # Only create directory after successful download
                        save_dir.mkdir(exist_ok=True)
                        print("Saving model locally...")
                        tokenizer.save_pretrained(save_dir)
                        model.save_pretrained(save_dir)

                        print(f"✓ Successfully downloaded to: {save_dir}")
                        return True

                except Exception as download_error:
                    # Clean up empty directory if it was created
                    if save_dir.exists():
                        try:
                            shutil.rmtree(save_dir)
                            print(f"Cleaned up incomplete download directory: {save_dir}")
                        except:
                            pass
                    raise download_error  # Re-raise to be caught by outer except

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error downloading {model_id}:")
            print(f"   {error_msg}")

            # Provide helpful error messages
            if "401" in error_msg or "authentication" in error_msg.lower():
                print("\n💡 This model requires HuggingFace authentication.")
                print("   Please follow these steps:")
                print("   1. Create account at https://huggingface.co")
                print("   2. Get your token from https://huggingface.co/settings/tokens")
                print("   3. Run: huggingface-cli login")
                print("   4. Or set: HF_TOKEN environment variable")
            elif "gated" in error_msg.lower() or "access" in error_msg.lower():
                print(f"\n💡 This model requires access approval.")
                print(f"   Visit: https://huggingface.co/{model_id}")
                print("   Click 'Request Access' and wait for approval.")
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                print("\n💡 Network issue detected. Please check your internet connection.")
            else:
                print("\n💡 You can try:")
                print("   - Choose a different model")
                print("   - Check your internet connection")
                print("   - Try again later")

            return False

    def run_interactive_setup(self) -> Dict:
        """Run interactive model selection."""
        self.print_header()

        # Check for existing config
        saved_config = self.load_saved_config()

        if saved_config:
            print("Found existing configuration:")
            print(f"  Embedding: {saved_config.get('embedding_model', 'N/A')}")
            print(f"  QA Model: {saved_config.get('qa_model', 'N/A')}")
            print(f"  Generator: {saved_config.get('generator_model', 'N/A')}")

            use_saved = input("\nUse this configuration? [Y/n]: ").strip().lower()
            if use_saved != 'n':
                print("\n✓ Using saved configuration\n")
                return saved_config

        print("\nLet's configure your models...\n")

        # Select models
        config = {
            'embedding_model': self.select_model('embeddings', 'Embedding Model'),
            'qa_model': self.select_model('qa_model', 'Question Answering Mode'),
            'generator_model': self.select_model('generator', 'Text Generator (Optional)')
        }

        # Delete old generator model if changed (to save storage)
        if saved_config and saved_config.get('generator_model') != config['generator_model']:
            old_generator = saved_config.get('generator_model')
            if old_generator and old_generator not in ['none', 'extractive']:
                print(f"\n{'='*70}")
                print(f"STORAGE MANAGEMENT")
                print(f"{'='*70}\n")
                print(f"Old generator: {old_generator}")
                print(f"New generator: {config['generator_model']}")
                print("\nDeleting old generator model to save storage...")
                self.delete_old_generator(old_generator)

        # Download models
        print(f"\n\n{'='*70}")
        print("DOWNLOADING REQUIRED MODELS")
        print(f"{'='*70}\n")

        models_to_download = [
            (config['embedding_model'], 'embeddings'),
            (config['qa_model'], 'qa_model'),
            (config['generator_model'], 'generator')
        ]

        for model_id, model_type in models_to_download:
            if not self.is_model_downloaded(model_id):
                print(f"\nDownloading {model_type}: {model_id}")
                if not self.download_model(model_id, model_type):
                    print(f"⚠ Warning: Failed to download {model_id}")

        # Save configuration
        self.save_config(config)

        print(f"\n\n{'='*70}")
        print("✓ CONFIGURATION COMPLETE!")
        print(f"{'='*70}\n")
        print("Your app is now ready to work OFFLINE!")
        print("All models are cached locally.\n")

        # Show summary
        print("Configuration Summary:")
        print(f"  Embedding Model: {MODELS['embeddings'][config['embedding_model']]['name']}")
        print(f"  QA Mode: {MODELS['qa_model'][config['qa_model']]['name']}")
        print(f"  Generator: {MODELS['generator'][config['generator_model']]['name']}")

        print(f"\n{'='*70}\n")

        return config

    def update_config_file(self, model_config: Dict):
        """Update config.py with selected models."""
        # Convert boolean values to Python's True/False (capitalized)
        use_advanced_qa = 'True' if model_config['qa_model'] != 'extractive' else 'False'
        use_generator = 'True' if model_config['generator_model'] != 'none' else 'False'

        config_content = f'''"""
Configuration file for the PDF Q&A System
Auto-generated by model_selector.py
"""

# QA Engine Configuration
QA_CONFIG = {{
    # Mode: 'extractive' (fast, accurate, uses regex) or 'generative' (slower, uses AI)
    'mode': 'extractive' if '{model_config['qa_model']}' == 'extractive' else 'advanced',

    # Use advanced QA model (requires more memory but better accuracy)
    'use_advanced_qa': {use_advanced_qa},

    # Advanced QA model to use if use_advanced_qa=True
    'advanced_qa_model': '{model_config['qa_model']}',

    # Number of top chunks to retrieve
    'top_k_chunks': 5,

    # Maximum answer length
    'max_answer_length': 800,
}}

# Embedding Model Configuration
EMBEDDING_CONFIG = {{
    'model_name': '{model_config['embedding_model']}',
}}

# Generator Model Configuration
GENERATOR_CONFIG = {{
    'model_name': '{model_config['generator_model']}',
    'use_generator': {use_generator},
}}

# PDF Processing Configuration
PDF_CONFIG = {{
    'chunk_size': 400,  # words per chunk
    'chunk_overlap': 50,  # overlapping words between chunks
}}

# Flask Configuration
FLASK_CONFIG = {{
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'max_content_length': 16 * 1024 * 1024,  # 16MB
}}
'''

        with open('config.py', 'w') as f:
            f.write(config_content)

        print("✓ Updated config.py with your selections")


def main():
    """Main entry point."""
    selector = ModelSelector()

    try:
        config = selector.run_interactive_setup()
        selector.update_config_file(config)

        print("\n✅ Setup complete! Starting the application...\n")
        return True

    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        return False
    except Exception as e:
        print(f"\n\n❌ Error during setup: {str(e)}")
        return False


if __name__ == "__main__":
    main()
