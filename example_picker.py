from datasets import load_from_disk

dataset = load_from_disk("converted")

train_dataset = dataset['train']

accepted = 0
rejected = 0

for i, example in enumerate(train_dataset):
    if i > 100:
        break
    print(f"topic: {example['topic']}")
    print("comment: %s" % (example['comment'].replace('\n', ' ')))
    prompt = input("accept?")
    if prompt == "y":
        print("accepted")
        accepted += 1
    else:
        rejected += 1