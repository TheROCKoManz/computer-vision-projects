# CROWD LINE 

# gunicorn_config.py

# IP address and port to bind to
bind = "0.0.0.0:6943"  # Bind to all available network interfaces on port 6943

# Set the maximum request body size to 250 MB
max_request_size = 262144000  # 250 MB in bytes

# Set the maximum request line size to 16KB
limit_request_line = 1638400

# Set the maximum number of request headers (default value)
limit_request_fields = 100

# Set the maximum header field size to 32KB
limit_request_field_size = 32768  # 32KB

# Number of worker processes
workers = 2
