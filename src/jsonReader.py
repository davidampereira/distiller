import json
import os

def formatter(format):
    return  f"""
    <|im_start|>user
    {format[0]}<|im_end|>
    <|im_start|>assistant
    <think>{format[1]}</think>
    {format[2]}<|im_end|>
    """

def reader(responseDir = "modelResponses"):

    if os.path.exists(responseDir) == False:
        raise Exception(f"{dir} could not be accessed")
    elif os.path.isfile(responseDir):
        raise Exception("Path given is not directory")

    messages = []

    for entry in os.scandir(responseDir):  
        if entry.is_file():  # check if it's a file
            print("dsa")
            with open(entry.path, "r") as f:
                data = json.load(f)
            userInput = data["input"][0]["content"][0]["text"]
            reasoning = data["choices"][0]["message"]["reasoning"]
            response = data["choices"][0]["message"]["content"]
            messages.append(formatter([userInput, reasoning, response]))
    return messages
