delta = 0
for i in range(len(lats)):
    delta = max(delta, float(aqi[i] - model.predict([[lats[i], lons[i]]])[0]))
print(delta)
