# NASA TEMPO Data Downloader

A simple Python script to download NASA TEMPO data and upload it to AWS S3.

## Features

- ✅ Simple, robust NASA TEMPO data downloading
- ✅ AWS S3 integration with proper credential handling
- ✅ Comprehensive error handling and logging
- ✅ Configurable date ranges and download settings
- ✅ Support for both config file and environment variables
- ✅ Automatic retry and timeout handling

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run setup (optional - creates config file)
python setup.py
```

### 2. Configuration

Edit `config.json` with your credentials:

```json
{
  "nasa_api_key": "YOUR_NASA_API_KEY_HERE",
  "aws_access_key_id": "YOUR_AWS_ACCESS_KEY_ID_HERE", 
  "aws_secret_access_key": "YOUR_AWS_SECRET_ACCESS_KEY_HERE",
  "aws_region": "us-east-1",
  "s3_bucket": "your-tempo-data-bucket",
  "download_dir": "./downloads",
  "date_range_days": 7
}
```

**Or use environment variables:**
```bash
export NASA_API_KEY="your_nasa_api_key"
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export S3_BUCKET="your-bucket-name"
```

### 3. Run

```bash
# Download data for default date range (last 7 days)
python nasa_tempo_downloader.py

# Download data for specific date range
python nasa_tempo_downloader.py --start-date 2024-01-01 --end-date 2024-01-07

# Use custom config file
python nasa_tempo_downloader.py --config my_config.json
```

## Getting API Keys

### NASA API Key
1. Visit: https://api.nasa.gov/
2. Sign up for a free API key
3. Add the key to your configuration

### AWS Credentials
1. **Option 1**: Use AWS IAM roles (recommended for EC2/ECS)
2. **Option 2**: Create IAM user with S3 permissions
3. **Option 3**: Use AWS CLI configured credentials

Required S3 permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket-name/*"
        }
    ]
}
```

## File Structure

```
├── nasa_tempo_downloader.py  # Main script
├── config.json               # Configuration file
├── requirements.txt          # Python dependencies
├── setup.py                  # Setup script
├── downloads/                # Downloaded data (created automatically)
└── tempo_download.log        # Log file (created automatically)
```

## Error Handling

The script includes comprehensive error handling for:

- ✅ Invalid API keys
- ✅ Network connectivity issues
- ✅ AWS credential problems
- ✅ S3 bucket access issues
- ✅ File system errors
- ✅ Invalid date formats

## Logging

All operations are logged to both console and `tempo_download.log` file.

## Troubleshooting

### Common Issues

1. **NASA API Key Invalid**
   - Verify your API key at https://api.nasa.gov/
   - Check rate limits (1000 requests per hour)

2. **AWS Credentials Error**
   - Verify AWS credentials are correct
   - Check S3 bucket permissions
   - Ensure bucket exists and is accessible

3. **Network Issues**
   - Check internet connectivity
   - Verify firewall settings
   - Try with different date ranges

### Debug Mode

Run with verbose logging:
```bash
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
exec(open('nasa_tempo_downloader.py').read())
"
```

## License

This project is open source and available under the MIT License.