from os import wait
import src.distiller as distiller
import src.jsonReader as jsonReader

def main():
    model_name = "Qwen/Qwen3-4B"
    responseDir = "testResponses"
    saveModelTo = "distilledModels"
    dataset = jsonReader.reader(responseDir)
    

if __name__ == "__main__":
    main()
