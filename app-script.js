/**
 * NASA TEMPO Air Quality Monitor - Frontend JavaScript
 * Interactive map with real-time air quality data visualization
 */

class AirQualityMonitor {
    constructor() {
        this.map = null;
        this.markers = [];
        this.currentPollutant = 'NO2';
        this.airQualityData = {};
        this.savedLocations = [];
        this.apiBaseUrl = 'http://localhost:5001/api';
        
        this.init();
    }
    
    init() {
        this.initMap();
        this.bindEvents();
        this.loadInitialData();
        this.startAutoRefresh();
    }
    
    initMap() {
        // Initialize Leaflet map centered on North America
        this.map = L.map('map').setView([39.8283, -98.5795], 4);
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);
        
        // Add click handler for map clicks
        this.map.on('click', (e) => {
            this.handleMapClick(e);
        });
        
        console.log('Map initialized');
    }
    
    bindEvents() {
        // Pollutant selection change
        document.getElementById('pollutant-select').addEventListener('change', (e) => {
            this.currentPollutant = e.target.value;
            this.updateMapDisplay();
        });
        
        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.refreshData();
        });
        
        // Clear locations button
        document.getElementById('clear-locations-btn').addEventListener('click', () => {
            this.clearSavedLocations();
        });
    }
    
    async loadInitialData() {
        try {
            this.updateStatus('Loading data...', 'loading');
            
            // Load air quality data
            await this.fetchAirQualityData();
            
            // Load saved locations
            await this.fetchSavedLocations();
            
            // Update map display
            this.updateMapDisplay();
            
            this.updateStatus('Connected', 'connected');
            
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.updateStatus('Connection failed', 'error');
        }
    }
    
    async fetchAirQualityData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/air-quality`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.airQualityData = await response.json();
            console.log('Air quality data loaded:', this.airQualityData);
            
        } catch (error) {
            console.error('Error fetching air quality data:', error);
            throw error;
        }
    }
    
    async fetchSavedLocations() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/locations`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.savedLocations = await response.json();
            console.log('Saved locations loaded:', this.savedLocations);
            
        } catch (error) {
            console.error('Error fetching saved locations:', error);
            // Don't throw error for locations as it's not critical
        }
    }
    
    updateMapDisplay() {
        // Clear existing markers
        this.clearMarkers();
        
        // Add air quality markers
        this.addAirQualityMarkers();
        
        // Add saved location markers
        this.addSavedLocationMarkers();
        
        // Update statistics
        this.updateStatistics();
    }
    
    clearMarkers() {
        this.markers.forEach(marker => {
            this.map.removeLayer(marker);
        });
        this.markers = [];
    }
    
    addAirQualityMarkers() {
        const pollutantData = this.airQualityData[this.currentPollutant];
        
        if (!pollutantData || !pollutantData.features) {
            console.log(`No data available for ${this.currentPollutant}`);
            return;
        }
        
        pollutantData.features.forEach(feature => {
            const coords = feature.geometry.coordinates;
            const value = feature.properties.value;
            
            // Create marker with color based on air quality
            const marker = L.circleMarker([coords[1], coords[0]], {
                radius: this.getMarkerRadius(value),
                fillColor: this.getAirQualityColor(value),
                color: '#fff',
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.7
            });
            
            // Add popup with air quality information
            const popupContent = this.createPopupContent(feature);
            marker.bindPopup(popupContent);
            
            // Add marker to map
            marker.addTo(this.map);
            this.markers.push(marker);
        });
        
        console.log(`Added ${pollutantData.features.length} markers for ${this.currentPollutant}`);
    }
    
    addSavedLocationMarkers() {
        this.savedLocations.forEach(location => {
            const marker = L.marker([location.latitude, location.longitude], {
                icon: L.divIcon({
                    className: 'custom-marker',
                    html: '<i class="fas fa-bookmark"></i>',
                    iconSize: [30, 30],
                    iconAnchor: [15, 15]
                })
            });
            
            const popupContent = `
                <div class="popup-content">
                    <div class="popup-title">${location.name}</div>
                    <div class="popup-details">
                        <strong>Coordinates:</strong><br>
                        Lat: ${location.latitude.toFixed(4)}<br>
                        Lon: ${location.longitude.toFixed(4)}<br>
                        <strong>Saved:</strong> ${new Date(location.timestamp).toLocaleString()}
                    </div>
                </div>
            `;
            
            marker.bindPopup(popupContent);
            marker.addTo(this.map);
            this.markers.push(marker);
        });
    }
    
    createPopupContent(feature) {
        const value = feature.properties.value;
        const pollutant = feature.properties.pollutant;
        const timestamp = new Date(feature.properties.timestamp).toLocaleString();
        
        const aqiCategory = this.getAQICategory(value);
        
        return `
            <div class="popup-content">
                <div class="popup-title">${pollutant} Air Quality</div>
                <div class="popup-value" style="color: ${this.getAirQualityColor(value)}">
                    ${value.toFixed(2)} ppb
                </div>
                <div class="popup-details">
                    <strong>Category:</strong> ${aqiCategory}<br>
                    <strong>Timestamp:</strong> ${timestamp}<br>
                    <strong>Coordinates:</strong><br>
                    Lat: ${feature.geometry.coordinates[1].toFixed(4)}<br>
                    Lon: ${feature.geometry.coordinates[0].toFixed(4)}
                </div>
            </div>
        `;
    }
    
    getMarkerRadius(value) {
        // Scale marker size based on value (min 3, max 15)
        const minRadius = 3;
        const maxRadius = 15;
        const maxValue = 500; // Adjust based on typical pollutant ranges
        
        return Math.max(minRadius, Math.min(maxRadius, (value / maxValue) * maxRadius));
    }
    
    getAirQualityColor(value) {
        // Color coding based on air quality index
        if (value <= 50) return '#00e400';      // Good - Green
        if (value <= 100) return '#ffff00';     // Moderate - Yellow
        if (value <= 150) return '#ff8c00';     // Unhealthy for Sensitive Groups - Orange
        if (value <= 200) return '#ff0000';     // Unhealthy - Red
        if (value <= 300) return '#8f3f97';     // Very Unhealthy - Purple
        return '#7e0023';                       // Hazardous - Maroon
    }
    
    getAQICategory(value) {
        if (value <= 50) return 'Good';
        if (value <= 100) return 'Moderate';
        if (value <= 150) return 'Unhealthy for Sensitive Groups';
        if (value <= 200) return 'Unhealthy';
        if (value <= 300) return 'Very Unhealthy';
        return 'Hazardous';
    }
    
    async handleMapClick(e) {
        const lat = e.latlng.lat;
        const lng = e.latlng.lng;
        
        // Prompt user for location name
        const name = prompt('Enter a name for this location:', `Location ${this.savedLocations.length + 1}`);
        
        if (name) {
            try {
                await this.saveLocation(name, lat, lng);
            } catch (error) {
                console.error('Error saving location:', error);
                alert('Failed to save location. Please try again.');
            }
        }
    }
    
    async saveLocation(name, lat, lng) {
        const response = await fetch(`${this.apiBaseUrl}/locations`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                latitude: lat,
                longitude: lng
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const newLocation = await response.json();
        this.savedLocations.push(newLocation);
        
        // Update map display
        this.updateMapDisplay();
        
        console.log('Location saved:', newLocation);
    }
    
    async clearSavedLocations() {
        if (confirm('Are you sure you want to clear all saved locations?')) {
            // Note: This would require a DELETE endpoint on the backend
            // For now, we'll just clear from the frontend
            this.savedLocations = [];
            this.updateMapDisplay();
            console.log('Saved locations cleared');
        }
    }
    
    updateStatistics() {
        // Update data points count
        const pollutantData = this.airQualityData[this.currentPollutant];
        const dataPointsCount = pollutantData ? pollutantData.features.length : 0;
        document.getElementById('data-points-count').textContent = dataPointsCount;
        
        // Update saved locations count
        document.getElementById('saved-locations-count').textContent = this.savedLocations.length;
        
        // Update last update time
        const now = new Date();
        document.getElementById('last-update').textContent = now.toLocaleTimeString();
    }
    
    updateStatus(text, status) {
        document.getElementById('status-text').textContent = text;
        const statusDot = document.getElementById('status-dot');
        
        // Remove existing status classes
        statusDot.classList.remove('connected', 'loading', 'error');
        
        // Add new status class
        if (status) {
            statusDot.classList.add(status);
        }
    }
    
    async refreshData() {
        try {
            this.updateStatus('Refreshing...', 'loading');
            
            await this.fetchAirQualityData();
            this.updateMapDisplay();
            
            this.updateStatus('Connected', 'connected');
            
        } catch (error) {
            console.error('Error refreshing data:', error);
            this.updateStatus('Refresh failed', 'error');
        }
    }
    
    startAutoRefresh() {
        // Refresh data every 5 minutes
        setInterval(() => {
            this.refreshData();
        }, 5 * 60 * 1000);
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing NASA TEMPO Air Quality Monitor...');
    
    // Check if we're running on localhost
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        new AirQualityMonitor();
    } else {
        // For production, you might want to use a different API URL
        console.log('Running in production mode');
        new AirQualityMonitor();
    }
});

// Handle errors globally
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});