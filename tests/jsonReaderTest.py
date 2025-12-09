import src.jsonReader as jsonReader

def main():
    responseDir = "testResponses"
    print(jsonReader.reader(responseDir))

if __name__ == "__main__":
    main()
