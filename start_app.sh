#!/bin/bash
# NASA TEMPO Air Quality Monitor Startup Script

echo "🚀 Starting NASA TEMPO Air Quality Monitor..."

# Check if backend is already running
if pgrep -f "flask-backend.py" > /dev/null; then
    echo "⚠️  Backend is already running"
else
    echo "🔄 Starting Flask backend..."
    python3 flask-backend.py &
    sleep 3
fi

# Check if backend is responding
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✅ Backend is running on http://localhost:5000"
else
    echo "❌ Backend failed to start"
    exit 1
fi

echo "🌐 Frontend is ready!"
echo "📱 Open your browser and go to: http://localhost:8000"
echo "🔗 Or simply open index.html in your browser"
echo ""
echo "📊 API Endpoints:"
echo "   - Health: http://localhost:5000/health"
echo "   - Air Quality: http://localhost:5000/api/air-quality"
echo "   - Status: http://localhost:5000/api/status"
echo ""
echo "🛑 To stop the backend, press Ctrl+C or run: pkill -f flask-backend.py"

# Start frontend server
python3 -m http.server 8000