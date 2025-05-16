from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model configurations
    model_name: str = "google/flan-t5-small"
    num_labels: int = 3  # positive, negative, neutral

    # Training configurations
    learning_rate: float = 1e-4
    batch_size: int = 4  # Small batch size for Colab
    epochs: int = 5

    # LoRA configurations
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Paths
    train_data_path: str = "data/new_train.csv"
    test_data_path: str = "data/new_test.csv"

    # Output
    output_dir: str = "sentiment_model"
