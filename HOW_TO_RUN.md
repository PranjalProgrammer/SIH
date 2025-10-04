# 🚀 How to Run NASA TEMPO Air Quality Monitor

## ✅ **QUICK START (Recommended)**

The application is now ready to run! Here's the simplest way to get started:

### **Option 1: Use the Startup Script (Easiest)**
```bash
./start_app.sh
```
This will:
- Start the Flask backend automatically
- Serve the frontend on http://localhost:8000
- Show you all the available endpoints

### **Option 2: Manual Start**
```bash
# Terminal 1: Start the backend
python3 flask-backend.py

# Terminal 2: Serve the frontend (optional)
python3 -m http.server 8000
```

### **Option 3: Just Open the HTML File**
```bash
# Start the backend
python3 flask-backend.py &

# Open index.html directly in your browser
# The app will work with mock data for demonstration
```

## 🌐 **Access the Application**

- **Frontend**: http://localhost:8000 (or open index.html directly)
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

## 📊 **What You'll See**

The application includes:
- ✅ **Interactive Map**: Leaflet-based map with air quality data
- ✅ **5 Pollutants**: NO2, O3, HCHO, SO2, CO
- ✅ **Real-time Data**: Mock data that updates automatically
- ✅ **Location Saving**: Click on the map to save locations
- ✅ **Responsive Design**: Works on desktop and mobile

## 🔧 **API Endpoints**

Test these endpoints to verify everything works:

```bash
# Health check
curl http://localhost:5000/health

# Get all air quality data
curl http://localhost:5000/api/air-quality

# Get specific pollutant (NO2)
curl http://localhost:5000/api/air-quality/NO2

# Get application status
curl http://localhost:5000/api/status

# Save a location
curl -X POST http://localhost:5000/api/locations \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Location", "latitude": 40.7128, "longitude": -74.0060}'
```

## 🛠️ **Troubleshooting**

### **Backend won't start?**
```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill any existing processes
pkill -f flask-backend.py

# Try a different port
PORT=5001 python3 flask-backend.py
```

### **No data showing on map?**
- The app uses mock data for demonstration
- Check browser console for JavaScript errors
- Ensure backend is running on http://localhost:5000

### **CORS errors?**
- Make sure Flask-CORS is installed: `pip install Flask-CORS`
- Check that backend is running before opening frontend

### **Map not loading?**
- Check internet connection (needed for Leaflet map tiles)
- Try refreshing the page
- Check browser console for errors

## 📁 **Project Structure**

```
├── flask-backend.py          # Flask API server
├── index.html               # Frontend web page
├── app-styles.css           # CSS styles
├── app-script.js            # JavaScript functionality
├── start_app.sh            # Startup script
├── requirements.txt          # Python dependencies
├── config.json              # Configuration
└── README.md               # Full documentation
```

## 🎯 **Features Working**

- ✅ **Real-time air quality visualization**
- ✅ **Interactive Leaflet maps**
- ✅ **Multiple pollutant support**
- ✅ **Location saving functionality**
- ✅ **Responsive web interface**
- ✅ **RESTful API endpoints**
- ✅ **Mock data for demonstration**

## 🔄 **Next Steps**

1. **Explore the web interface** - Click on the map to see air quality data
2. **Try different pollutants** - Use the dropdown to switch between NO2, O3, etc.
3. **Save locations** - Click on the map to save interesting locations
4. **Check the API** - Use curl or Postman to explore the REST API
5. **Add real data** - Set AWS credentials to use real NASA TEMPO data

## 📞 **Need Help?**

If you're still having issues:

1. **Check the logs**: Look at the terminal output for error messages
2. **Verify Python version**: `python3 --version` (should be 3.9+)
3. **Test individual components**: Run `python3 test_complete_functionality.py`
4. **Check dependencies**: `pip list | grep -E "(flask|boto3|numpy)"`

---

**🎉 The application is ready to use! Just run `./start_app.sh` and open your browser.**