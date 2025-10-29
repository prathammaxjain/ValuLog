# ----------------------------
# #Testing user datapoint for validity
# ----------------------------
lat,lon,usr_aqi = map(float, input().split()) #To take from google sheet (probably) 

pred_aqi = model.predict([[lat, lon]])[0]
if abs(usr_aqi - pred_aqi) < delta:
    print('OKAY')
    #Add datapoint to dataset with weight 
    #user --> weight 0.7
    #machine --> weight 1
else:
    print('NOKAY')

print(f"Reported AQI at ({lat}, {lon}): {usr_aqi:.2f}")
print(f"Predicted AQI at ({lat}, {lon}): {pred_aqi:.2f}")
