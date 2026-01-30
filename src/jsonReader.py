"""JSON conversation data reader and formatter."""

import json
import os

from src.logger import get_logger

logger = get_logger("finer.jsonReader")


def formatter(format_data: tuple) -> dict:
    """Format raw data into training example dictionary."""
    if len(format_data) == 3:
        return {
            "user_message": format_data[0],
            "chatbot_reasoning": format_data[1],
            "chatbot_response": format_data[2],
        }
    return {
        "user_message": format_data[0],
        "chatbot_reasoning": "",
        "chatbot_response": format_data[1],
    }


def reader(response_dir: str = "modelResponses") -> list[dict]:
    """Read and parse JSON conversation files from a directory.

    Args:
        response_dir: Directory containing JSON response files.

    Returns:
        List of formatted conversation dictionaries.

    Raises:
        FileNotFoundError: If directory doesn't exist.
        ValueError: If path is a file, not directory.
    """
    if not os.path.exists(response_dir):
        raise FileNotFoundError(f"{response_dir} could not be accessed")

    if os.path.isfile(response_dir):
        raise ValueError("Path given is not a directory")

    messages = []
    file_count = 0

    for entry in os.scandir(response_dir):
        if entry.is_file() and entry.name.endswith(".json"):
            file_count += 1
            try:
                with open(entry.path) as f:
                    data = json.load(f)

                user_input = data["input"][0]["content"][0]["text"]
                response = data["choices"][0]["message"]["content"]
                reasoning = data["choices"][0]["message"].get("reasoning")

                if reasoning:
                    messages.append(formatter((user_input, reasoning, response)))
                else:
                    messages.append(formatter((user_input, response)))

            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to parse {entry.name}: {e}")
                continue

    logger.info(f"Loaded {len(messages)} conversations from {file_count} files")
    return messages
