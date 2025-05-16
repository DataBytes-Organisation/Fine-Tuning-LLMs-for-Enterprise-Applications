from dataclasses import dataclass

@dataclass
class ModelConfig:
    # model configurations
    model_name: str = "google/flan-t5-small"
    num_labels: int = 3  # positive, negative, neutral

    # training configurations
    learning_rate: float = 5e-5
    batch_size: int = 4  # small batch size for Colab
    epochs: int = 8

    # LoRA configurations
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    # paths
    train_data_path: str = "data/new_train.csv"
    test_data_path: str = "data/new_test.csv"

    # Output
    output_dir: str = "sentiment_model"
