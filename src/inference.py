from setfit import SetFitModel

def main(text):
    model = SetFitModel.from_pretrained("carlesoctav/SentimentClassifierBarbieDune-8shot")
    preds = model(text)
    print(preds)

if __name__ =="__main__":
    text = input()
    main(text)


