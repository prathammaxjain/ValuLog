# ======================================
# ML-based Spatial AQI Map Visualization
# ======================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------
# 1. AQI data (fake data)
# ----------------------------
np.random.seed(42)
num_cities = 40

lats = np.random.uniform(8, 37, num_cities)        # India lat range
lons = np.random.uniform(68, 97, num_cities)       # India lon range
aqi = (
    -100 + 4*lats + 2*lons + np.random.normal(0, 15, num_cities)
)

df = pd.DataFrame({'Latitude': lats, 'Longitude': lons, 'AQI': aqi})
#print("Sample data:\n", df.head())

# ----------------------------
# 2. Train ML regression model
# ----------------------------
X = df[['Latitude', 'Longitude']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ----------------------------
# 3. Create spatial grid for prediction
# ----------------------------
grid_lat = np.linspace(df['Latitude'].min(), df['Latitude'].max(), 200)
grid_lon = np.linspace(df['Longitude'].min(), df['Longitude'].max(), 200)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
grid_points = np.c_[lat_grid.ravel(), lon_grid.ravel()]

# Predict AQI for the grid
aqi_pred = model.predict(grid_points)
aqi_grid = aqi_pred.reshape(lat_grid.shape)

# ----------------------------
# 4. Plot predicted AQI map
# ----------------------------
plt.figure(figsize=(10, 8))
contour = plt.contourf(lon_grid, lat_grid, aqi_grid, levels=20, cmap='RdYlGn_r')
plt.scatter(df['Longitude'], df['Latitude'], c=df['AQI'], edgecolor='k', cmap='RdYlGn_r', s=80)
plt.colorbar(contour, label='Predicted AQI')
plt.title('Spatial AQI Map (Random Forest Regression)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
