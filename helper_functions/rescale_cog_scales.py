import h5py
import numpy as np



pathGTTrain = "CLSA_CFA_GRAHAM_GoodAndUsable.hdf5"
with h5py.File(pathGTTrain, "r") as f:  # "with" close the file after its nested commands
    everything = f["gt"][:][:]
    cog = f["gt"][:][:, 28]
    mem = f["gt"][:][:, 29]
    spd = f["gt"][:][:, 30]
    exe = f["gt"][:][:, 31]
    stp = f["gt"][:][:, 24]

max = 100
min = -230
cog_new = (100-0)/(max-min)*(cog-max)+100
#
#
max = 0
min = 10
mem_new = (100-0)/(max-min)*(mem-max)+100
#
max = 0
min = 100
spd_new = (100-0)/(max-min)*(spd-max)+100
#
max = 0
min = 32
exe_new = (100-0)/(max-min)*(exe-max)+100
#
max = 0.5
min = 2.5
stp_new = (100-0)/(max-min)*(stp-max)+100


everything[:, 28] = cog_new
everything[:, 29] = mem_new
everything[:, 30] = spd_new
everything[:, 31] = exe_new
everything[:, 24] = stp_new

print('train')
data = everything[0:18000]
print("age mean" + str(np.mean(data[:, 1])) + '||' "std" + str(np.std(data[:, 1])))
print("bmi mean" + str(np.mean(data[:, 35])) + '||' "std" + str(np.std(data[:, 35])))
print("sbp mean" + str(np.mean(data[:, 32])) + '||' "std" + str(np.std(data[:, 32])))
print("dpb mean" + str(np.mean(data[:, 33])) + '||' "std" + str(np.std(data[:, 33])))
print("exec mean" + str(np.mean(data[:, 31])) + '||' "std" + str(np.std(data[:, 31])))
print("speed mean" + str(np.mean(data[:, 30])) + '||' "std" + str(np.std(data[:, 30])))
print("mem mean" + str(np.mean(data[:, 29])) + '||' "std" + str(np.std(data[:, 29])))
print("inhib mean" + str(np.mean(data[:, 24])) + '||' "std" + str(np.std(data[:, 24])))
print("cog mean" + str(np.mean(data[:, 28])) + '||' "std" + str(np.std(data[:, 28])))

print('val')
data = everything[18000:21860]
print("age mean" + str(np.mean(data[:, 1])) + '||' "std" + str(np.std(data[:, 1])))
print("bmi mean" + str(np.mean(data[:, 35])) + '||' "std" + str(np.std(data[:, 35])))
print("sbp mean" + str(np.mean(data[:, 32])) + '||' "std" + str(np.std(data[:, 32])))
print("dpb mean" + str(np.mean(data[:, 33])) + '||' "std" + str(np.std(data[:, 33])))
print("exec mean" + str(np.mean(data[:, 31])) + '||' "std" + str(np.std(data[:, 31])))
print("speed mean" + str(np.mean(data[:, 30])) + '||' "std" + str(np.std(data[:, 30])))
print("mem mean" + str(np.mean(data[:, 29])) + '||' "std" + str(np.std(data[:, 29])))
print("inhib mean" + str(np.mean(data[:, 24])) + '||' "std" + str(np.std(data[:, 24])))
print("cog mean" + str(np.mean(data[:, 28])) + '||' "std" + str(np.std(data[:, 28])))

print('test')
data = everything[21860:]
print("age mean" + str(np.mean(data[:, 1])) + '||' "std" + str(np.std(data[:, 1])))
print("bmi mean" + str(np.mean(data[:, 35])) + '||' "std" + str(np.std(data[:, 35])))
print("sbp mean" + str(np.mean(data[:, 32])) + '||' "std" + str(np.std(data[:, 32])))
print("dpb mean" + str(np.mean(data[:, 33])) + '||' "std" + str(np.std(data[:, 33])))
print("exec mean" + str(np.mean(data[:, 31])) + '||' "std" + str(np.std(data[:, 31])))
print("speed mean" + str(np.mean(data[:, 30])) + '||' "std" + str(np.std(data[:, 30])))
print("mem mean" + str(np.mean(data[:, 29])) + '||' "std" + str(np.std(data[:, 29])))
print("inhib mean" + str(np.mean(data[:, 24])) + '||' "std" + str(np.std(data[:, 24])))
print("cog mean" + str(np.mean(data[:, 28])) + '||' "std" + str(np.std(data[:, 28])))

import matplotlib.pyplot as plt
n, bins, patches = plt.hist(x=cog, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

print(1)
with h5py.File(pathGTTrain, "r+") as f:
    f["gt"][...] = everything
