#!/bin/bash
set -e

echo "🚀 Starting Agent Mesh Node"
echo "Node ID: ${NODE_ID:-auto-generated}"
echo "Listen Port: ${LISTEN_PORT}"
echo "Node Role: ${NODE_ROLE}"
echo "Region: ${REGION}"
echo "Bootstrap Peers: ${BOOTSTRAP_PEERS}"

# Wait for database if using PostgreSQL
if [[ $DATABASE_URL == postgresql* ]]; then
    echo "⏳ Waiting for PostgreSQL..."
    
    # Extract host and port from DATABASE_URL
    DB_HOST=$(echo $DATABASE_URL | sed -E 's|postgresql://[^@]*@([^:/]*):?([0-9]*)/.*|\1|')
    DB_PORT=$(echo $DATABASE_URL | sed -E 's|postgresql://[^@]*@([^:/]*):?([0-9]*)/.*|\2|')
    DB_PORT=${DB_PORT:-5432}
    
    until timeout 1 bash -c "cat < /dev/null > /dev/tcp/${DB_HOST}/${DB_PORT}"; do
        echo "⏳ PostgreSQL is unavailable - sleeping"
        sleep 2
    done
    
    echo "✅ PostgreSQL is up - executing command"
fi

# Run database migrations if needed
if [[ "$NODE_ROLE" == "bootstrap" ]]; then
    echo "📊 Running database migrations..."
    python3 scripts/run_migrations.py || echo "⚠️  Migration script not found, continuing..."
fi

# Generate node ID if not provided
if [[ -z "$NODE_ID" ]]; then
    export NODE_ID=$(python3 -c "from uuid import uuid4; print(uuid4())")
    echo "Generated Node ID: $NODE_ID"
fi

# Set up logging
export PYTHONUNBUFFERED=1

# Health check endpoint
echo "🏥 Starting health check endpoint on port ${HEALTH_CHECK_PORT}..."
python3 -c "
import http.server
import socketserver
import threading
import time

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{\"status\":\"healthy\",\"timestamp\":' + str(time.time()).encode() + b'}')
        elif self.path == '/ready':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{\"status\":\"ready\"}')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress health check logs

def start_health_server():
    port = int('${HEALTH_CHECK_PORT}')
    with socketserver.TCPServer(('', port), HealthHandler) as httpd:
        httpd.serve_forever()

health_thread = threading.Thread(target=start_health_server, daemon=True)
health_thread.start()
print('Health check server started')
" &

# Performance tuning based on role
if [[ "$NODE_ROLE" == "bootstrap" ]]; then
    echo "🌟 Configuring for bootstrap node (high performance)"
    export CACHE_SIZE=10000
    export MAX_CONNECTIONS=100
elif [[ "$NODE_ROLE" == "worker" ]]; then
    echo "⚙️  Configuring for worker node (balanced)"
    export CACHE_SIZE=5000
    export MAX_CONNECTIONS=50
fi

echo "✅ Environment configured, starting application..."
echo "================================================"

# Execute the main command
exec "$@"