#!/usr/bin/env python3
import sys
import time
import json
import argparse
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
        except requests.exceptions.RequestException:
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
        return False

def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Initialize Colette backend services')
    parser.add_argument('--service-name', type=str, default='app_colette',
                        help='Name of the service to create (default: app_colette)')
    parser.add_argument('--backend-url', type=str, default='http://localhost:1873',
                        help='Backend URL (default: http://localhost:1873)')
    parser.add_argument('--config', type=str, 
                        default='/app/src/colette/config/vrag_default.json',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    backend_url = args.backend_url
    service_name = args.service_name
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        service_config = json.load(f)
    
    # Set the repository to a writable path
    service_config["app"]["repository"] = f"/rag/{service_name}"
    
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