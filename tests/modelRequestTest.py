import src.messageModels as messageModels

def main():
    message = "Hello"
    files = ["https://bitcoin.org/bitcoin.pdf"]
    fileToWrite = "testResponse"
    directory = "testResponses"
    amm = 1
    messageModels.getMessages(message, fileToWrite, directory, amm)

if __name__ == "__main__":
    main()
