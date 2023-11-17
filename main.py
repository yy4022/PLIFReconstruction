import numpy as np

from preprocess_dataset import preprocess_data, preprocess_old_data, show_image

# PART 1: define the parameters
# 1. provide all filenames of PIV, PLIF data
file1_PIV = str('data/Attached state/D1F1_air240_PIV_1001to2000.mat') # attached-1000
file1_PLIF = str('data/Attached state/D1F1_air240_PLIF_1001to2000.mat')
file2_PIV = str('data/Attached state/D1F1_air240_PIV_2001to3000.mat') # attached-1000
file2_PLIF = str('data/Attached state/D1F1_air240_PLIF_2001to3000.mat')

# file3_PIV = str('data/Detachment process/D1F1_air240_PIV_4201to4500.mat') # detachment-300
# file3_PLIF = str('data/Detachment process/D1F1_air240_PLIF_4201to4500.mat')
file4_PIV = str('data/Detachment process/D1F1_air240_PIV_13401to13600.mat') # detachment-200
file4_PLIF = str('data/Detachment process/D1F1_air240_PLIF_13401to13600.mat')

file5_PIV = str('data/Lifted state/D1F1_air240_PIV_13601to14000.mat') # lift-400
file5_PLIF = str('data/Lifted state/D1F1_air240_PLIF_13601to14000.mat')
file6_PIV = str('data/Lifted state/D1F1_air240_PIV_14001to14999.mat') # lift-999
file6_PLIF = str('data/Lifted state/D1F1_air240_PLIF_14001to14999.mat')

file7_PIV = str('data/Reattachment process/D1F1_air240_PIV_451to650.mat') # reattachment-200
file7_PLIF = str('data/Reattachment process/D1F1_air240_PLIF_451to650.mat')
file8_PIV = str('data/Reattachment process/D1F1_air240_PIV_6201to6700.mat') # reattachment-500
file8_PLIF = str('data/Reattachment process/D1F1_air240_PLIF_6201to6700.mat')

# PART 2: preprocess the datasets
# 1. preprocess the datasets, then return the cropped datasets
PIV_data1, PLIF_data1 = preprocess_old_data(file1_PIV, file1_PLIF)[:2]
PIV_data2, PLIF_data2 = preprocess_old_data(file2_PIV, file2_PLIF)[:2]
# PIV_data3, PLIF_data3 = preprocess_old_data(file3_PIV, file3_PLIF)[:2]
PIV_data4, PLIF_data4 = preprocess_old_data(file4_PIV, file4_PLIF)[:2]
PIV_data5, PLIF_data5 = preprocess_old_data(file5_PIV, file5_PLIF)[:2]
PIV_data6, PLIF_data6 = preprocess_old_data(file6_PIV, file6_PLIF)[:2]
PIV_data7, PLIF_data7 = preprocess_old_data(file7_PIV, file7_PLIF)[:2]
PIV_data8, PLIF_data8 = preprocess_old_data(file8_PIV, file8_PLIF)[:2]

# 2. concatenate the datasets as required
PIV_attached_data = np.concatenate((PIV_data1, PIV_data2), axis=1)
PLIF_attached_data = np.concatenate((PLIF_data1, PLIF_data2), axis=0)

PIV_detachment_data = PIV_data4
PLIF_detachment_data = PLIF_data4

PIV_lifted_data = np.concatenate((PIV_data5, PIV_data6), axis=1)
PLIF_lifted_data = np.concatenate((PLIF_data5, PLIF_data6), axis=0)

PIV_reattachment_data = np.concatenate((PIV_data7, PIV_data8), axis=1)
PLIF_reattachment_data = np.concatenate((PLIF_data7, PLIF_data8), axis=0)

# 3. split the datasets for training, evaluation and testing


