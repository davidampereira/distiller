"""Main entry point for Finer - LLM knowledge distillation."""

from src import distiller, jsonReader, messageModels
from src.config import get
from src.logger import setup_logger


def main():
    """Run the full distillation pipeline."""
    # Initialize logging from config
    logger = setup_logger(
        level=get("logging.level", "INFO"),
        log_file=get("logging.file"),
    )


    
    model_save_dir = get("training.output_dir", "distilledModel")
    model = get("training.model", "Qwen/Qwen3-4B")
    messageModels.get_messages()

    logger.info("Formatting conversation data...")
    formatted = jsonReader.format_start()

    logger.info(f"Starting distillation with model: {model}")
    distiller.distill(model, formatted, model_save_dir)

    logger.info("Distillation complete!")


if __name__ == "__main__":
    main()
