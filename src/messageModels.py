from dotenv import load_dotenv
load_dotenv()
import requests
import json
import base64
import os

def encode_file_to_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def parseFiles(files):
    message = dict()
    for file in files:
        if(file[:7] != "http://" and file[:8] != "https://"):
            base64_file = encode_file_to_base64(file)
            data_url = f"data:application/pdf;base64,{base64_file}"
        else: data_url = file
        name = file.split("/")[-1]
        message.update({
                "type": "file",
                "file": {
                    "filename": name,
                    "file_data": data_url
                },
        })
    return message


def getMessages(message, fileWrite, responseDir, it=1, files=None):
    if os.path.exists(responseDir) == False:
        try:
            os.mkdir(responseDir)
        except Exception as e:
            print("Couldn't create given directory")
            print(e)
            exit()
    elif os.path.isfile(responseDir):
        raise Exception("Path given is not directory")
    models = [("tngtech/deepseek-r1t2-chimera:free", "deepseek"), ("z-ai/glm-4.5-air:free", "glm")]
    messages = [
            {
                "role": "user",
                "content": [
            {
                "type": "text",
                "text": message,
            },
            ]
        }
    ]
    if files is not None:
        raise Exception("Attaching files not supported")
        # messages[0]["content"].append(parseFiles(files))
    count = 0
    total = len(models) * it
    for model in models:
        for i in range(it):
            count += 1
            print(f"{count} / {total}")
            modelResponse = openRouterRequest(model[0], messages)
            modelResponse["input"] = messages
            with open(f"{responseDir}/{fileWrite}_{model[1]}_{i}.json", "w") as f:
                json.dump(modelResponse, f, indent=4)

def openRouterRequest(model, message):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
    "model": model,
    "messages": message
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()
    except Exception as e:
        raise(f"{e}")

