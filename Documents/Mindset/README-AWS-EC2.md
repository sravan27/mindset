# MINDSET on AWS EC2

This document explains the EC2-based deployment of MINDSET, which provides a simple, containerless deployment option.

## Architecture Overview

The EC2 deployment creates a single instance that hosts:

1. **Backend API**: FastAPI server running on port 8000
2. **Frontend Application**: Next.js frontend served via Nginx
3. **Data Pipeline**: Processing raw data through the medallion architecture
4. **Silicon Layer**: Advanced ML processing for transparency metrics
5. **S3 Storage**: For data persistence across the medallion layers

![Architecture](docs/architecture-ec2.png)

## Deployment Process

The `deploy-mindset-aws-ec2.sh` script automates:

1. Creating S3 buckets for data storage
2. Uploading datasets to S3
3. Creating a security group for EC2
4. Creating an IAM role with S3 access
5. Launching an EC2 instance with Ubuntu
6. Installing dependencies (Python, Rust, Node.js, etc.)
7. Setting up systemd services for all components

## Components on EC2

### Frontend (Next.js)

- Runs as a systemd service
- Served via Nginx on port 80
- Displays news articles with transparency metrics

### Backend API (FastAPI)

- Runs as a systemd service
- Provides API endpoints for articles and metrics
- Connected to S3 for data access

### Data Pipeline

- Runs as a scheduled service via systemd timer
- Processes data through the medallion architecture
- Full pipeline with Raw → Bronze → Silver → Silicon → Gold layers

### Silicon Layer

- Advanced ML processing between Silver and Gold layers
- Provides transparency metrics calculation
- Integrates with Rust metrics engine for performance

## S3 Data Structure

The S3 bucket contains these directories:

- `raw/`: Original source data
- `bronze/`: Cleaned and validated data
- `silver/`: Feature-engineered data
- `silicon/`: Advanced ML processed data
- `gold/`: API-ready data with metrics
- `datasets/`: MINDLarge datasets

## Monitoring and Management

### SSH Access

Connect to your instance:

```bash
ssh -i <key-name>.pem ubuntu@<public-ip>
```

### Logs

View logs for each component:

```bash
# API logs
sudo journalctl -u mindset-api.service

# Frontend logs
sudo journalctl -u mindset-frontend.service

# Pipeline logs
sudo journalctl -u mindset-pipeline.service

# Setup logs
sudo cat /var/log/user-data.log
```

### Managing Services

Control services:

```bash
# Restart API
sudo systemctl restart mindset-api.service

# Restart frontend
sudo systemctl restart mindset-frontend.service

# Run pipeline manually
sudo systemctl start mindset-pipeline.service
```

## Customization

Edit the configuration:

```bash
# API and pipeline config
nano /home/ubuntu/Mindset/.env

# Nginx config
sudo nano /etc/nginx/sites-available/mindset

# Frontend API URL
nano /home/ubuntu/Mindset/src/frontend/.env.local
```

## Troubleshooting

Common issues and solutions:

1. **API not responding**:
   - Check API service: `sudo systemctl status mindset-api.service`
   - View logs: `sudo journalctl -u mindset-api.service`

2. **Frontend not loading**:
   - Check frontend service: `sudo systemctl status mindset-frontend.service`
   - Check Nginx: `sudo systemctl status nginx`

3. **Pipeline failing**:
   - View pipeline logs: `sudo journalctl -u mindset-pipeline.service`
   - Run manually: `cd /home/ubuntu/Mindset && python run_pipeline_aws.py`

4. **S3 access issues**:
   - Check instance role: `curl http://169.254.169.254/latest/meta-data/iam/info`
   - Verify bucket permissions