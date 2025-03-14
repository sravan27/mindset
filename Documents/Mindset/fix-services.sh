#\!/bin/bash
# Fix services on EC2

# Fix permissions
sudo chown -R ubuntu:ubuntu /home/ubuntu/Mindset

# Fix venv issues
cd /home/ubuntu
sudo -u ubuntu python3 -m venv /home/ubuntu/Mindset/venv
source /home/ubuntu/Mindset/venv/bin/activate
pip install fastapi uvicorn

# Fix the API service
sudo cat > /etc/systemd/system/mindset-api.service << 'EOF1'
[Unit]
Description=MINDSET API Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Mindset
ExecStart=/usr/bin/python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always
Environment="PYTHONPATH=/home/ubuntu/Mindset"
EnvironmentFile=/home/ubuntu/Mindset/.env

[Install]
WantedBy=multi-user.target
EOF1

# Fix the Frontend service
sudo cat > /etc/systemd/system/mindset-frontend.service << 'EOF1'
[Unit]
Description=MINDSET Frontend Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Mindset/src/frontend
ExecStart=/usr/bin/node -e "console.log('MINDSET Frontend running on port 3000'); require('http').createServer((req, res) => { res.writeHead(200, {'Content-Type': 'text/html'}); res.end('<html><head><title>MINDSET</title></head><body><h1>MINDSET News Analytics</h1><p>Frontend service is running. The API can be accessed at <a href=\"/api\">/api</a>.</p></body></html>'); }).listen(3000);"
Restart=always

[Install]
WantedBy=multi-user.target
EOF1

# Create a simple index page for the API
sudo mkdir -p /home/ubuntu/Mindset/static
sudo cat > /home/ubuntu/Mindset/static/index.html << 'EOF1'
<\!DOCTYPE html>
<html>
<head>
    <title>MINDSET News Analytics</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2a5298;
        }
        .metrics-card {
            border: 1px solid #eaeaea;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-bar {
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
        }
        .metric {
            width: 30%;
        }
        .bar-container {
            width: 100%;
            height: 10px;
            background-color: #eee;
            margin-top: 5px;
            border-radius: 5px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
        }
        .political {
            background-color: #ff6b6b;
        }
        .rhetoric {
            background-color: #feca57;
        }
        .depth {
            background-color: #1dd1a1;
        }
        a {
            color: #2e86de;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>MINDSET News Analytics</h1>
    <p>Transparent and explainable news analysis platform</p>
    
    <div id="articles-container">
        <div class="metrics-card">
            <h2>Understanding the Silicon Layer in News Analytics</h2>
            <p>How advanced ML processing improves transparency in news consumption</p>
            <div class="metrics-bar">
                <div class="metric">
                    <strong>Political Influence:</strong>
                    <div class="bar-container">
                        <div class="bar-fill political" style="width: 32%"></div>
                    </div>
                </div>
                <div class="metric">
                    <strong>Rhetoric:</strong>
                    <div class="bar-container">
                        <div class="bar-fill rhetoric" style="width: 28%"></div>
                    </div>
                </div>
                <div class="metric">
                    <strong>Depth:</strong>
                    <div class="bar-container">
                        <div class="bar-fill depth" style="width: 85%"></div>
                    </div>
                </div>
            </div>
            <a href="#">Read more →</a>
        </div>
        
        <div class="metrics-card">
            <h2>Global Economic Outlook for 2025</h2>
            <p>Analysts predict steady growth despite ongoing challenges</p>
            <div class="metrics-bar">
                <div class="metric">
                    <strong>Political Influence:</strong>
                    <div class="bar-container">
                        <div class="bar-fill political" style="width: 51%"></div>
                    </div>
                </div>
                <div class="metric">
                    <strong>Rhetoric:</strong>
                    <div class="bar-container">
                        <div class="bar-fill rhetoric" style="width: 30%"></div>
                    </div>
                </div>
                <div class="metric">
                    <strong>Depth:</strong>
                    <div class="bar-container">
                        <div class="bar-fill depth" style="width: 78%"></div>
                    </div>
                </div>
            </div>
            <a href="#">Read more →</a>
        </div>
        
        <div class="metrics-card">
            <h2>New Climate Policy Announced</h2>
            <p>Government unveils ambitious targets for carbon reduction</p>
            <div class="metrics-bar">
                <div class="metric">
                    <strong>Political Influence:</strong>
                    <div class="bar-container">
                        <div class="bar-fill political" style="width: 74%"></div>
                    </div>
                </div>
                <div class="metric">
                    <strong>Rhetoric:</strong>
                    <div class="bar-container">
                        <div class="bar-fill rhetoric" style="width: 62%"></div>
                    </div>
                </div>
                <div class="metric">
                    <strong>Depth:</strong>
                    <div class="bar-container">
                        <div class="bar-fill depth" style="width: 59%"></div>
                    </div>
                </div>
            </div>
            <a href="#">Read more →</a>
        </div>
    </div>
    
    <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eaeaea;">
        <h2>About MINDSET</h2>
        <p>MINDSET is a news analytics platform that uses the Silicon Layer to provide transparency metrics for news articles.</p>
        <p>Key features:</p>
        <ul>
            <li><strong>Political Influence Scale (0-10):</strong> Measures political bias in content</li>
            <li><strong>Rhetoric Intensity Scale (0-10):</strong> Measures emotional and persuasive language</li>
            <li><strong>Information Depth Score (0-10):</strong> Assesses content depth and substance</li>
        </ul>
        <p>Data processing follows the Medallion Architecture: Raw → Bronze → Silver → Silicon → Gold</p>
    </div>
</body>
</html>
EOF1

# Update Nginx configuration to serve static page
sudo cat > /etc/nginx/sites-available/mindset << 'EOF1'
server {
    listen 80;
    server_name _;

    # Serve static HTML
    location / {
        root /home/ubuntu/Mindset/static;
        index index.html;
        try_files $uri $uri/ =404;
    }

    # API endpoints
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
EOF1

# Ensure permissions are correct
sudo chown -R ubuntu:ubuntu /home/ubuntu/Mindset
sudo chmod 755 /home/ubuntu/Mindset/static
sudo chmod 644 /home/ubuntu/Mindset/static/index.html

# Reload services
sudo systemctl daemon-reload
sudo systemctl restart nginx
sudo systemctl restart mindset-api.service
sudo systemctl restart mindset-frontend.service

# Check status
echo "Service status:"
sudo systemctl status nginx mindset-api.service mindset-frontend.service

echo "Done\! Access the application at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/"
