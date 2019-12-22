from os import listdir, remove

path = "./data/weights/"

weights = [path + f for f in sorted(listdir(path))]

for weight in weights:
	score = int(weight.split(".")[-2])
	if score < 99500:
		remove(weight)
