import distiller
import jsonReader
import messageModels

def main():
    messages = ['What are good ways to greet someone formally?', 'How can I make someone feel welcome?', 'What are friendly ways to salute someone?']
    directory = "conversations"
    amm = 10
    modelSaveDir = "distilledModel"
    model = "Qwen/Qwen3-4B"
    count = 0
    totalMessages = len(messages)
    for i in range(len(messages)):
        count += 1
        print(f"Messages: {count} / {totalMessages}")
        messageModels.getMessages(messages[i], f"nr_{i}", directory, amm)
    formatted = jsonReader.reader(directory)
    distiller.distill(model, formatted, modelSaveDir)

if __name__ == "__main__":
    main()
