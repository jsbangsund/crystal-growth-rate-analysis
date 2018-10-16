import os
import glob
dirs = ['170C_4x_timeseries','170C_10x_timeseries','170C_10x_timeseries_2','170C_10x_timeseries_3','170C_10x_timeseries_4']
base_dir = 'C:\\Users\\HolmesGroup\\Desktop\\User Data\\JSB\\181004_SubstrateVariation\\Glass-ITO\\'
str1='160C'
str2='170C'
for dir in dirs:
    base = os.path.join(base_dir,dir)
    files=glob.glob(os.path.join(base,'*.png'))
    for f in files:
        os.rename(f,f.replace(str1,str2))