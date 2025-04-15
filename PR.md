# Docker Deployment and Infrastructure Updates

## Overview
This PR introduces Docker containerization and updates the deployment infrastructure for the recommender system, moving from Streamlit Cloud to Google Cloud Platform (GCP). The changes include Dockerfile creation, requirements separation, and infrastructure updates.

## Key Changes

### 1. Docker Containerization
- Added `Dockerfile` for containerized deployment
- Uses Python 3.12 slim base image
- Separates Streamlit-specific requirements into `requirements-st.txt`
- Configures proper port exposure and server settings

### 2. Requirements Management
- Split requirements into two files:
  - `requirements-st.txt`: Streamlit-specific dependencies
  - `requirements.txt`: Core project dependencies
- Added PyTorch and PyTorch Geometric for deep learning capabilities
- Updated dependency versions for better compatibility

### 3. ETL Process Improvements
- Enhanced data extraction process in `scripts/etl/etl.py`
- Added support for handling zip files from Kaggle downloads
- Improved temporary file management and cleanup

### 4. Deployment Updates
- Migrated from Streamlit Cloud to GCP
- Updated deployment URL in README
- Added performance note regarding GCP hardware limitations

## Impact
- Improved deployment flexibility with Docker containerization
- Better dependency management with separated requirements
- More robust data extraction process
- More scalable deployment infrastructure on GCP

## Testing
- Verify Docker build and run locally
- Test data extraction process with new zip handling
- Validate Streamlit app functionality in containerized environment

## Notes
- Current GCP configuration (2GB memory, 1 CPU) may result in slower app loading times
- Consider monitoring performance and scaling resources as needed 