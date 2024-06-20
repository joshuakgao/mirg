import json

class MuMuQA():
    def __init__(self):
        self.train = json.loads(open("./data/muMuQA/train/train.json", 'r').read())
        self.dev = json.loads(open("./data/muMuQA/eval/dev.json", 'r').read())
        self.test = json.loads(open("./data/muMuQA/eval/test.json", 'r').read())


    def get_image_urls(self):
        image_urls = [example['image'] for example in self.train]
        print(f"{len(image_urls)} examples in database")
        return image_urls


    def get_examples_by_indexes(self, indexes):
        return [self.train[i] for i in indexes]            





if __name__ == "__main__":
    muMuQA = MuMuQA()
    # muMuQA.get_muMuQA_image_urls()
    examples = muMuQA.get_examples_by_indexes([0,2])