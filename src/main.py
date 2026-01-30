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

    # Load configuration
    messages = [
        "What are good ways to greet someone formally?",
        "How can I make someone feel welcome?",
        "What are friendly ways to salute someone?",
    ]
    directory = get("data_collection.conversations_dir", "conversations")
    iterations = get("data_collection.iterations_per_model", 10)
    model_save_dir = get("training.output_dir", "distilledModel")
    model = get("training.model", "Qwen/Qwen3-4B")

    total_messages = len(messages)

    for count, message in enumerate(messages, 1):
        logger.info(f"Processing message {count}/{total_messages}: {message[:50]}...")
        messageModels.getMessages(message, f"nr_{count - 1}", directory, iterations)

    logger.info("Formatting conversation data...")
    formatted = jsonReader.reader(directory)

    logger.info(f"Starting distillation with model: {model}")
    distiller.distill(model, formatted, model_save_dir)

    logger.info("Distillation complete!")


if __name__ == "__main__":
    main()
