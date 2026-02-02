"""OpenRouter API integration for collecting LLM responses."""

import base64
import json
import os

import requests
from dotenv import load_dotenv

from src.config import get
from src.logger import get_logger

load_dotenv()

logger = get_logger("finer.messageModels")


def encode_file_to_base64(file: str) -> str:
    """Encode a file to base64 string."""
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_files(files: list[str]) -> dict:
    """Parse files for API request (currently unused)."""
    message = {}
    for file in files:
        if not file.startswith(("http://", "https://")):
            base64_file = encode_file_to_base64(file)
            data_url = f"data:application/pdf;base64,{base64_file}"
        else:
            data_url = file
        name = file.split("/")[-1]
        message.update(
            {
                "type": "file",
                "file": {"filename": name, "file_data": data_url},
            }
        )
    return message


def get_messages():
    messagesloc = get("data_collection.questionsloc", "messages/questions.txt")
    iterations = get("data_collection.iterations_per_model", 10)
    count = 0

    with open(messagesloc, "r") as f:
        while True:
            message = f.readline()
            if(message == ""):
                break
            message = message.strip()
            get_response(message, f"{count}", iterations)
            count += 1


def get_response(
    message: str,
    file_write: str,
    iterations: int = 1,
):
    """Collect responses from multiple LLMs for a given message.

    Args:
        message: User message to send to models.
        file_write: Base filename for saving responses.
        response_dir: Directory to save response files.
        iterations: Number of responses per model.
        files: Optional files to attach (not currently supported).
    """
    response_dir = get("data_collection.conversation_dir", "conversations")
    if not os.path.exists(response_dir):
        try:
            os.mkdir(response_dir)
            logger.info(f"Created response directory: {response_dir}")
        except Exception as e:
            logger.error(f"Couldn't create directory: {e}")
            raise

    if os.path.isfile(response_dir):
        raise ValueError("Path given is not a directory")


    # Get models from config
    models = get(
        "data_collection.models",
        [
            {"id": "tngtech/deepseek-r1t2-chimera:free", "name": "deepseek"},
            {"id": "z-ai/glm-4.5-air:free", "name": "glm"},
        ],
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": message}],
        }
    ]

    total = len(models) * iterations
    count = 0

    for model in models:
        for i in range(iterations):
            count += 1
            logger.info(f"Request {count}/{total} - {model['name']} iteration {i + 1}")

            model_response = openrouter_request(model["id"], messages)
            model_response["input"] = messages

            output_file = f"{response_dir}/{file_write}_{model['name']}_{i}.json"
            with open(output_file, "w") as f:
                json.dump(model_response, f, indent=4)

            logger.debug(f"Saved response to {output_file}")


def openrouter_request(model: str, message: list) -> dict:
    """Make a request to the OpenRouter API.

    Args:
        model: Model identifier.
        message: Message payload.

    Returns:
        API response as dictionary.

    Raises:
        requests.HTTPError: If API request fails.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        logger.error("OPENROUTER_API_KEY not set in environment")
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": message}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise


# Backwards compatibility aliases
openRouterRequest = openrouter_request
parseFiles = parse_files
