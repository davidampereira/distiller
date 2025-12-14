import json
import os
from datasets import Dataset

def formatter(format):
    return {"user_message": format[0], "chatbot_reasoning": format[1], "chatbot_response": format[2]}

def reader(responseDir = "modelResponses"):

    if os.path.exists(responseDir) == False:
        raise Exception(f"{dir} could not be accessed")
    elif os.path.isfile(responseDir):
        raise Exception("Path given is not directory")

    messages = []

    for entry in os.scandir(responseDir):  
        if entry.is_file():  # check if it's a file
            with open(entry.path, "r") as f:
                data = json.load(f)
            userInput = data["input"][0]["content"][0]["text"]
            reasoning = data["choices"][0]["message"]["reasoning"]
            response = data["choices"][0]["message"]["content"]
            messages.append(formatter([userInput, reasoning, response]))
    return messages
