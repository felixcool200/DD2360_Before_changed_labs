import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

n_bins = 4096

histData = []
with open('histOutput.txt') as f:
    lines = f.readlines()
    for line in lines:
        histData.append(int(line))


fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
axs[0].hist(histData, bins=n_bins)

plt.show()