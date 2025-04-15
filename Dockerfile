# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-st.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-st.txt

# Copy the application code
COPY ./models ./models
COPY ./scripts ./scripts
COPY ./main.py ./main.py
COPY ./data ./data
# Expose the port Streamlit runs on
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
