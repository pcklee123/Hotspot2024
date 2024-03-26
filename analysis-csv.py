import pandas as pd
import matplotlib.pyplot as plt

def scale_to_initial(df, attr):
    df[attr + '_scl'] = df[attr] / df[attr][0]
num_files = 1
fig, plots = plt.subplots(num_files, 1)
if num_files <= 1: plots = [plots]
plt.subplots_adjust(hspace=0.4)
filenames = ['log.csv']
for i, ax in enumerate(plots):
    df = pd.read_csv(filenames[i])
    #print(df.columns)
    df['KEavg_ele'] = df['KEt_e'] / -df['ne']
    df['KEavg_deut'] = df['KEt_d'] / df['ni']
    df.plot(ax=ax, y=['KEt_e', 'KEt_d', 'Ele_pot', 'Mag_pot', 'E_tot'], secondary_y = 'nc_ele')
    scale_to_initial(df, 'ne')
    scale_to_initial(df, 'E_tot')
    df.plot(y=['ne_scl', 'E_tot_scl'])
    #ax2 = ax.twinx()
    #df.plot(ax=ax2, y=['ncalc_ele'])
    #j.set_title() # some attributes
plt.show()
