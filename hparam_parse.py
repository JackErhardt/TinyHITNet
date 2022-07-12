import re

def parseFile(filepath):
	with open(filepath, "r") as file:
		for line in file:
			if re.search("'epe'", line):
				epe = float(re.findall("\d+\.\d+", line)[0])
			if re.search("'rate_1'", line):
				rate1 = float(re.findall("\d+\.\d+", line)[0])
			if re.search("'rate_1e-1", line):
				rate0_1 = float(re.findall("\d+\.\d+", line)[0])
			if re.search("'rate_3", line):
				rate3 = float(re.findall("\d+\.\d+", line)[0])
	return [epe, rate0_1, rate1, rate3]

if __name__ == "__main__":
	modeList = ["same", "w", "h"]
	listList = ["square200", "square100"]
	padList  = [-1, 0, 32, 64, 128]

	for mode in modeList:
		print(f"Mode: {mode}")
		for listt in listList:
			print(f"\tList:{listt}")
			data = [parseFile(f"./sweep/{listt}{mode}{pad}.txt") for pad in padList]
			print(f"\t\tEPE:\n\t\t{[line[0] for line in data]}")
			print(f"\t\tRate 0.1%:\n\t\t{[line[1] for line in data]}")
			print(f"\t\tRate 1%:\n\t\t{[line[2] for line in data]}")
			print(f"\t\tRate 3%:\n\t\t{[line[3] for line in data]}")
