#%% Manuscript only
import matplotlib.pyplot as plt

xT = [115.83, 58.23, 5.64]
yT = [73.80, 57.93, 52.52]
labelsT = ['Supervised', 'SimCLR', 'SimCLR-UAL']

xF = [116.29, 59.71, 6.20]
yF = [74.43, 54.88, 52.74]
labelsF = ['Supervised', 'SimCLR', 'SimCLR-UAL']

plt.figure(figsize=(10, 10))

plt.scatter(xT, yT, color='red', label='Three-layer DCNN')
plt.scatter(xF, yF, color='blue', label='Three-layer DCNN-SVM')

for i, label in enumerate(labelsT):
    plt.text(xT[i], yT[i], label, ha='center')

for j, lab in enumerate(labelsF):
    plt.text(xF[j], yF[j], lab, ha='center')

plt.title('Training time vs. Testing accuracy with different methods (Full CIFAR-10 Dataset)')
plt.ylabel('Testing accuracy (%)')
plt.xlabel('Training time (seconds)')

plt.legend()
plt.tight_layout()

# %% Added diagram for PPT
plt.title('Number of samples vs. Testing accuracy with different methods (Full CIFAR-10 Dataset)')
plt.ylabel('Testing accuracy (%)')
plt.xlabel('Number of samples (per class)')

selected = [1000, 2000, 3000, 4000, 5000]

threeAcc = [74.07, 74.67, 76.26, 75.09, 73.80]
fourAcc = [75.34, 74.75, 75.36, 75.96, 74.43]

plt.plot(selected, threeAcc, color='r', marker='o', label='Three-layer DCNN')
plt.plot(selected, fourAcc, color='b', marker='o', label='Three-layer DCNN-SVM')

plt.legend()
plt.tight_layout()

# %%
