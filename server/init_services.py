#!/usr/bin/env python3
import sys
import time
import json
from pathlib import Path
import requests

def wait_for_backend(url, timeout=60):
    """Wait for backend to be ready"""
    print(f"Waiting for backend at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/v1/info")
            if response.status_code == 200:
                print("✓ Backend is ready!")
                return True
        except requests.exceptions.RequestException as e:
            pass
        time.sleep(2)
    print("✗ Backend timeout!")
    return False

def create_service(backend_url, service_name, config):
    """Create a service on the backend"""
    print(f"Creating service '{service_name}'...")
    try:
        response = requests.put(
            f"{backend_url}/v1/app/{service_name}",
            json=config,
            timeout=30
        )
        response.raise_for_status()
        print(f"✓ Service '{service_name}' created")
        return True
    except Exception as e:
        print(f"✗ Failed to create service '{service_name}': {e}")
        print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
        return False

def start_indexing(backend_url, service_name, config):
    """Start indexing for a service"""
    print(f"Starting indexing for '{service_name}'...")
    try:
        response = requests.put(
            f"{backend_url}/v1/index/{service_name}",
            json=config,
            timeout=30
        )
        response.raise_for_status()
        print(f"✓ Indexing started for '{service_name}'")
        return True
    except Exception as e:
        print(f"✗ Failed to start indexing for '{service_name}': {e}")
        print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
        return False

def main():
    backend_url = "http://localhost:1873"
    service_name = "app_colette"
    
    # Load the service config template
    config_path = Path("/app/src/colette/config/vrag_default.json")
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        service_config = json.load(f)
    
    # Set the repository to a writable path (mounted volume)
    service_config["app"]["repository"] = f"/rag/{service_name}"  # ← Full path!
    
    print(f"Backend URL: {backend_url}")
    print(f"Service name: {service_name}")
    print(f"Repository path: {service_config['app']['repository']}")
    
    # Wait for backend
    if not wait_for_backend(backend_url):
        sys.exit(1)
    
    # Check if service already exists
    response = requests.get(f"{backend_url}/v1/info")
    existing_services = response.json()["info"]["services"]
    print(f"Existing services: {existing_services}")
    
    if service_name in existing_services:
        print(f"✓ Service '{service_name}' already exists")
    else:
        # Create the service
        if not create_service(backend_url, service_name, service_config):
            sys.exit(1)
        
        # Start indexing
        if not start_indexing(backend_url, service_name, service_config):
            sys.exit(1)
    
    print("✓ Initialization complete!")

if __name__ == "__main__":
    main()