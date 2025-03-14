#!/bin/bash
# MINDSET EC2 Setup Script

# Set up logging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Starting MINDSET setup..."

# Update system
apt-get update
apt-get upgrade -y

# Install dependencies
apt-get install -y build-essential python3-pip python3-dev python3-venv \
    nginx nodejs npm git libssl-dev pkg-config awscli

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Clone MINDSET repository
cd /home/ubuntu
git clone https://github.com/yourusername/Mindset.git
cd Mindset

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install boto3 dask[complete] pyarrow duckdb

# Build Rust metrics engine
cd src/rust/metrics_engine
python build.py
cd ~/Mindset

# Set up environment variables
cat > .env << ENV_EOF
# AWS Settings
S3_BUCKET=mindset-g1jnilf3-data
AWS_REGION=eu-north-1
# API Keys
NEWSAPI_KEY=e9b0f7a3c3004651a35b5f0b042e1828
# Features
ENABLE_SILICON_LAYER=true
USE_RUST_ENGINE=true
ENV_EOF

# Replace placeholders with actual values
sed -i "s|mindset-g1jnilf3-data|${S3_BUCKET}|g" .env
sed -i "s|eu-north-1|${AWS_REGION}|g" .env
sed -i "s|e9b0f7a3c3004651a35b5f0b042e1828|${NEWSAPI_KEY}|g" .env

# Set up directories
mkdir -p ~/mindset_data/{raw,bronze,silver,silicon,gold}

# Install and configure Nginx
cat > /etc/nginx/sites-available/mindset << 'NGINX_EOF'
server {
    listen 80;
    server_name _;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
NGINX_EOF

ln -s /etc/nginx/sites-available/mindset /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
systemctl restart nginx

# Set up the frontend
cd ~/Mindset/src/frontend
npm install
npm run build

# Set up systemd services
# 1. API Service
cat > /etc/systemd/system/mindset-api.service << 'API_EOF'
[Unit]
Description=MINDSET API Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Mindset
ExecStart=/home/ubuntu/Mindset/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always
Environment="PYTHONPATH=/home/ubuntu/Mindset"
EnvironmentFile=/home/ubuntu/Mindset/.env

[Install]
WantedBy=multi-user.target
API_EOF

# 2. Frontend Service
cat > /etc/systemd/system/mindset-frontend.service << 'FRONTEND_EOF'
[Unit]
Description=MINDSET Frontend Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Mindset/src/frontend
ExecStart=/usr/bin/npm run start
Restart=always
Environment="NEXT_PUBLIC_API_URL=http://localhost/api"

[Install]
WantedBy=multi-user.target
FRONTEND_EOF

# 3. Data Pipeline Service
cat > /etc/systemd/system/mindset-pipeline.service << 'PIPELINE_EOF'
[Unit]
Description=MINDSET Data Pipeline
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Mindset
ExecStart=/home/ubuntu/Mindset/venv/bin/python run_pipeline_aws.py
Environment="PYTHONPATH=/home/ubuntu/Mindset"
EnvironmentFile=/home/ubuntu/Mindset/.env

[Install]
WantedBy=multi-user.target
PIPELINE_EOF

# 4. Data Pipeline Timer
cat > /etc/systemd/system/mindset-pipeline.timer << 'TIMER_EOF'
[Unit]
Description=Run MINDSET Pipeline hourly

[Timer]
OnBootSec=15min
OnUnitActiveSec=1h
Unit=mindset-pipeline.service

[Install]
WantedBy=timers.target
TIMER_EOF

# Enable and start services
systemctl daemon-reload
systemctl enable mindset-api.service
systemctl enable mindset-frontend.service
systemctl enable mindset-pipeline.timer

systemctl start mindset-api.service
systemctl start mindset-frontend.service
systemctl start mindset-pipeline.timer

echo "MINDSET setup complete!"
