# NASA TEMPO Air Quality Data Integration Project

A comprehensive web application that fetches, processes, and visualizes NASA TEMPO air quality data in real-time. The project consists of a Flask backend that processes NetCDF files from AWS S3 and a React frontend with interactive Leaflet maps.

## Features

- **Real-time Data Processing**: Fetches NASA TEMPO air quality data from AWS S3 bucket
- **NetCDF Processing**: Converts NetCDF files to GeoJSON format for web visualization
- **Interactive Maps**: Leaflet-based maps with dynamic air quality overlays
- **Multiple Pollutants**: Support for NO2, O3, HCHO, SO2, and CO monitoring
- **User Location Pinning**: Click-to-pin functionality for saving user locations
- **Automated Updates**: Hourly data refresh with background processing
- **Responsive Design**: Mobile-friendly interface with modern UI/UX

## Project Structure

```
├── flask-backend.py          # Flask backend with NASA TEMPO data processing
├── react-frontend.js         # React frontend component (standalone)
├── index.html               # Complete HTML application
├── app-styles.css           # CSS styles for the application
├── requirements.txt         # Python dependencies
├── config-files.md          # Configuration and deployment files
└── readme.md               # This file
```

## Quick Start

### Prerequisites

- Python 3.9+
- AWS Account with S3 access
- Modern web browser

### Local Development

1. **Clone and setup the project:**
   ```bash
   git clone <your-repo>
   cd nasa-tempo-air-quality
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials:**
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```

4. **Start the backend:**
   ```bash
   python flask-backend.py
   ```

5. **Open the frontend:**
   ```bash
   # Simply open index.html in your browser
   open index.html
   # Or serve it with a simple HTTP server:
   python -m http.server 8000
   ```

6. **Access the application:**
   - Frontend: http://localhost:8000
   - Backend API: http://localhost:5000

## API Endpoints

### Backend API (`http://localhost:5000/api`)

- `GET /health` - Health check endpoint
- `GET /air-quality` - Get all air quality data
- `GET /air-quality/{pollutant}` - Get data for specific pollutant
- `GET /air-quality/bounds` - Get data within geographical bounds
- `POST /locations` - Save user location
- `GET /locations` - Get saved user locations

### Example API Usage

```bash
# Get all air quality data
curl http://localhost:5000/api/air-quality

# Get NO2 data only
curl http://localhost:5000/api/air-quality/NO2

# Get data within bounds
curl "http://localhost:5000/api/air-quality/bounds?lat_min=40&lat_max=41&lon_min=-74&lon_max=-73"

# Save a location
curl -X POST http://localhost:5000/api/locations \
  -H "Content-Type: application/json" \
  -d '{"name": "My Location", "latitude": 40.7128, "longitude": -74.0060}'
```

## Deployment Options

### Option 1: Heroku (Free Tier)

1. **Install Heroku CLI:**
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Ubuntu/Debian
   curl https://cli-assets.heroku.com/install-ubuntu.sh | sh
   ```

2. **Deploy backend:**
   ```bash
   # Login to Heroku
   heroku login
   
   # Create app
   heroku create nasa-tempo-air-quality-backend
   
   # Set environment variables
   heroku config:set AWS_ACCESS_KEY_ID=your_key
   heroku config:set AWS_SECRET_ACCESS_KEY=your_secret
   heroku config:set AWS_DEFAULT_REGION=us-east-1
   
   # Deploy
   git add .
   git commit -m "Deploy NASA TEMPO backend"
   git push heroku main
   ```

3. **Deploy frontend:**
   - Use Netlify, Vercel, or GitHub Pages for static hosting
   - Update `REACT_APP_API_URL` to point to your Heroku backend

### Option 2: Railway (Free Tier)

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Deploy:**
   ```bash
   railway login
   railway init
   railway up
   ```

### Option 3: Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Individual containers:**
   ```bash
   # Backend
   docker build -t tempo-backend .
   docker run -p 5000:5000 -e AWS_ACCESS_KEY_ID=your_key tempo-backend
   
   # Frontend
   docker build -t tempo-frontend ./frontend
   docker run -p 80:80 tempo-frontend
   ```

## Configuration

### Environment Variables

**Backend:**
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_DEFAULT_REGION` - AWS region (default: us-east-1)
- `FLASK_ENV` - Flask environment (development/production)

**Frontend:**
- `REACT_APP_API_URL` - Backend API URL

### Data Sources

The application fetches data from NASA TEMPO S3 bucket:
- **Bucket**: `nasa-tempo-air-quality`
- **Format**: NetCDF files
- **Update Frequency**: Hourly
- **Coverage**: North America
- **Resolution**: 0.1° x 0.1°

## Features in Detail

### Backend Features

- **S3 Integration**: Automatic fetching from NASA TEMPO S3 bucket
- **NetCDF Processing**: Converts scientific data format to web-friendly GeoJSON
- **Background Updates**: Hourly data refresh without user intervention
- **RESTful API**: Clean API endpoints for frontend integration
- **Error Handling**: Comprehensive error handling and logging
- **CORS Support**: Cross-origin requests for frontend integration

### Frontend Features

- **Interactive Maps**: Leaflet-based maps with zoom, pan, and click functionality
- **Dynamic Visualization**: Color-coded markers based on air quality levels
- **Pollutant Selection**: Switch between different air quality parameters
- **Location Pinning**: Click-to-save user locations with custom markers
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Updates**: Automatic data refresh every hour

### Data Visualization

- **Color Coding**: 
  - Green: Good air quality
  - Yellow: Moderate
  - Orange: Unhealthy for sensitive groups
  - Red: Unhealthy
  - Purple: Very unhealthy
- **Marker Sizing**: Proportional to pollution levels
- **Interactive Popups**: Detailed information on click
- **Legend**: Clear visual guide for air quality categories

## Troubleshooting

### Common Issues

1. **AWS Credentials Error:**
   ```bash
   # Ensure AWS credentials are properly set
   aws configure list
   ```

2. **CORS Errors:**
   - Check that Flask-CORS is installed
   - Verify frontend URL is allowed in CORS configuration

3. **Data Not Loading:**
   - Check backend logs for S3 connection issues
   - Verify S3 bucket permissions
   - Ensure NetCDF files are accessible

4. **Map Not Displaying:**
   - Check browser console for JavaScript errors
   - Verify Leaflet CSS and JS are loaded
   - Check network connectivity

### Performance Optimization

- **Data Caching**: Backend caches processed data to reduce S3 requests
- **Lazy Loading**: Frontend loads data on demand
- **Marker Clustering**: For large datasets, consider implementing marker clustering
- **Bounds Filtering**: Use geographical bounds to limit data requests

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **NASA TEMPO**: For providing air quality data
- **Leaflet**: For the mapping library
- **React**: For the frontend framework
- **Flask**: For the backend framework
- **OpenStreetMap**: For map tiles

## Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the API documentation

## Roadmap

- [ ] User authentication and personalized dashboards
- [ ] Historical data analysis and trends
- [ ] Mobile app development
- [ ] Advanced filtering and search capabilities
- [ ] Integration with weather data
- [ ] Alert system for poor air quality
- [ ] Data export functionality
