import pandas as pd


filename = 'cardio_train.csv'
# my_data = np.genfromtxt(filename, delimiter=';')
data = pd.read_csv(filename, sep=';')

data = data.abs()
ap_hi = data.columns.get_loc('ap_hi')
# index = 0
for i in range(len(data['ap_hi'])):
    if data.iloc[i, ap_hi] > 9999:
        data.iloc[i, ap_hi] /= 100
    elif data.iloc[i, ap_hi] > 999:
        data.iloc[i, ap_hi] /= 10
    elif data.iloc[i, ap_hi] > 350:
        data.iloc[i, ap_hi] /= 3
    elif data.iloc[i, ap_hi] < 3:
        data.iloc[i, ap_hi] *= 100
    elif data.iloc[i, ap_hi] < 50:
        data.iloc[i, ap_hi] *= 15
    elif data.iloc[i, ap_hi] < 90:
        data.iloc[i, ap_hi] *= 3

print(data['ap_hi'].max(), data['ap_hi'].min())