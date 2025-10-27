import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#CSV Plot
#csv_path = 'radar-ssh-gui/src/SAR_DRONE_MOTION_DATA - SAR_DRONE_MOTION_DATA.csv'
#csv_path = 'radar-ssh-gui/src/team4_test_flight_one.csv'
#csv_path = 'radar-ssh-gui/src/team4_test_radar_slide.csv'
#csv_path = 'radar-ssh-gui/src/team4_july22_slide1.csv'
csv_path = 'DATAS/July31Flight4/pos.csv'

df = pd.read_csv(csv_path, skiprows=5, low_memory=False)

#Make sure to have the right columns
# print(df.columns[:20])

#convert the columns to numeric values
x = pd.to_numeric(df.iloc[:, 6], errors='coerce')  # X position
y = pd.to_numeric(df.iloc[:, 8], errors='coerce')  # Y position
z = pd.to_numeric(df.iloc[:, 7], errors='coerce')  # Z position

#drop any NaN vals
valid = (~x.isna()) & (~y.isna()) & (~z.isna())
x, y, z = x[valid], y[valid], z[valid]

#take every 300th point to reduce the number of points plotted
#x, y, z = x[::300], y[::300], z[::300]


#plot to data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot start and end points
ax.plot(x.iloc[0], y.iloc[0], z.iloc[0], marker='o', color='green', markersize=10, label='Start Point') #plot start point
ax.plot(x.iloc[-1], y.iloc[-1], z.iloc[-1], marker='o', color='red', markersize=10, label='End Point') #plot end point
ax.plot(x, y, z, marker='.', label='Drone Path', linewidth=1) #plot the drone path


#ensure all axes have the same scale
max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2

mid_x = (x.max() + x.min()) / 2
mid_y = (y.max() + y.min()) / 2
mid_z = (z.max() + z.min()) / 2

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

#label and axes
ax.set_xlabel('X (m)')
ax.set_zlabel('Y (m)')
ax.set_ylabel('Z (m)')
ax.set_title('3D Drone Trajectory')
ax.legend()
plt.tight_layout()
plt.show()
