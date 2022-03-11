import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# read in the data file
df = pd.read_csv(r'D:\Jessica\Documents\School\Project Winter\Jessica_Feb25\CompiledData.csv', skiprows=3)

# import the data for a given frequency
f = 2450
findex = df.loc[df["fMHz"] == f]

# import the milk and cream data for the frequency given above
skimdc = findex["skim_dc"].values
skimdl = findex["skim_dl"].values

milk1dc = findex["1_dc"].values
milk1dl = findex["1_dl"].values

milk2dc = findex["2_dc"].values
milk2dl = findex["2_dl"].values

homodc = findex["homo_dc"].values
homodl = findex["homo_dl"].values

# import the cream data for the frequency given above
cream5dc = findex["5_dc"].values
cream5dl = findex["5_dl"].values

cream10dc = findex["10_dc"].values
cream10dl = findex["10_dl"].values

cream18dc = findex["18_dc"].values
cream18dl = findex["18_dl"].values

cream35dc = findex["35_dc"].values
cream35dl = findex["35_dl"].values

# prepare data sets to plot
fat = np.array([0.10, 1.00, 2.00, 3.25, 5.00, 10.00, 18.00, 35.00]).reshape(-1, 1)
dc = np.concatenate([skimdc, milk1dc, milk2dc, homodc, cream5dc, cream10dc, cream18dc, cream35dc])
dl = np.concatenate([skimdl, milk1dl, milk2dl, homodl, cream5dl, cream10dl, cream18dl, cream35dl])

# split the data randomly for training and testing
sample_size = 0.2
random = 0
fat_train, fat_test, dc_train, dc_test = train_test_split(fat, dc, test_size=sample_size, random_state=random)
fat_train2, fat_test2, dl_train, dl_test = train_test_split(fat, dl, test_size=sample_size, random_state=random)

dcmodel = LinearRegression().fit(fat_train, dc_train)
print("Dielectric Constant Equation: ", dcmodel.coef_ , "x + ", dcmodel.intercept_)
print("R2 (Training Set): ", dcmodel.score(fat_train, dc_train))
print("R2 (Test Set): ", dcmodel.score(fat_test, dc_test))

dlmodel = LinearRegression().fit(fat_train2, dl_train)
print("Dielectric Loss Equation: ", dlmodel.coef_ , "x + ", dlmodel.intercept_)
print("R2 (Training Set): ", dlmodel.score(fat_train2, dl_train))
print("R2 (Test Set): ", dlmodel.score(fat_test2, dl_test))

# perform fat content prediction based on dc_test data and compare with fat_test
print("Fat Content Prediction using Test Data")
fatpredict = []
fatpredict2 = []
for i in range(0, len(dc_test)):
    fatpredict.append((dc_test[i] - dcmodel.intercept_)/dcmodel.coef_)
    fatpredict2.append((dl_test[i] - dlmodel.intercept_) / dlmodel.coef_)
    print("Measured Fat: ", fat_test[i], "Predicted Fat from DC: ", fatpredict[i], "Predicted Fat from DL: ", fatpredict2[i])
    i += 1

# plot the models, training sets, and test sets
fat_ = np.linspace(0, 35, 100)

plot1 = plt.figure(1)
plt.scatter(fat_train, dc_train, label="Training Set")
plt.scatter(fat_test, dc_test, label="Test Set")
plt.plot(fat_, dcmodel.coef_*fat_ + dcmodel.intercept_, label="Model")

handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]
plt.legend([handles[i] for i in order], [labels[i] for i in order])

plt.xlabel("Butterfat Content (%)")
plt.ylabel("Dielectric Constant ($\epsilon^{'}$)")

plot2 = plt.figure(2)
plt.scatter(fat_train2, dl_train, label="Training Set")
plt.scatter(fat_test2, dl_test, label="Test Set")
plt.plot(fat_, dlmodel.coef_*fat_ + dlmodel.intercept_, label="Model")

handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]
plt.legend([handles[i] for i in order], [labels[i] for i in order])

plt.xlabel("Butterfat Content (%)")
plt.ylabel("Dielectric Loss ($\epsilon^{''}$)")

plt.show()