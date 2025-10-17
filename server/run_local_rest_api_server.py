from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import os
import re
import base64
from io import BytesIO
from PIL import Image
import uvicorn

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from colette.jsonapi import JSONApi
from colette.apidata import APIData

# Initialize FastAPI app
app = FastAPI(
    title="Colette Vision RAG API",
    description="REST API for Colette Vision RAG Service",
    version="1.0.0"
)

# Global variables
colette_api = JSONApi()
colette_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app_name = 'app_colette'
service_created = False

# Pydantic models for request/response validation
class ServiceCreateRequest(BaseModel):
    documents_dir: Optional[str] = None
    app_dir: Optional[str] = None
    models_dir: Optional[str] = None
    config_file: Optional[str] = None

class ServiceIndexRequest(BaseModel):
    documents_dir: Optional[str] = None
    reindex: Optional[bool] = False
    index_config_file: Optional[str] = None

class QueryRequest(BaseModel):
    message: str
    return_images: Optional[bool] = False

class QueryResponse(BaseModel):
    output: str
    sources: List[Dict[str, Any]]

# Helper function to load default configs
def get_default_paths():
    return {
        'documents_dir': os.path.join(colette_root, 'docs/pdf'),
        'app_dir': os.path.join(colette_root, 'app_colette'),
        'models_dir': os.path.join(colette_root, 'models'),
        'config_file': os.path.join(colette_root, 'src/colette/config/vrag_default.json'),
        'index_file': os.path.join(colette_root, 'src/colette/config/vrag_default_index.json')
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Colette Vision RAG API",
        "version": "1.0.0",
        "status": "running",
        "service_created": service_created,
        "endpoints": {
            "create": "/service/create",
            "index": "/service/index",
            "query": "/service/query",
            "status": "/service/status"
        }
    }

@app.post("/service/create")
async def create_service(request: ServiceCreateRequest):
    """Create and initialize the Colette service"""
    global service_created
    
    try:
        paths = get_default_paths()
        
        # Use provided paths or defaults
        documents_dir = request.documents_dir or paths['documents_dir']
        app_dir = request.app_dir or paths['app_dir']
        models_dir = request.models_dir or paths['models_dir']
        config_file = request.config_file or paths['config_file']
        
        # Load configuration
        with open(config_file, 'r') as f:
            create_config = json.load(f)
        
        # Update config with paths
        create_config['app']['repository'] = app_dir
        create_config['app']['models_repository'] = models_dir
        
        # Create the service
        api_data_create = APIData(**create_config)
        colette_api.service_create(app_name, api_data_create)
        
        service_created = True
        
        return {
            "status": "success",
            "message": "Service created successfully",
            "app_name": app_name,
            "config": {
                "documents_dir": documents_dir,
                "app_dir": app_dir,
                "models_dir": models_dir
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating service: {str(e)}")

@app.post("/service/index")
async def index_documents(request: ServiceIndexRequest):
    """Index documents into the RAG system"""
    if not service_created:
        raise HTTPException(status_code=400, detail="Service not created. Call /service/create first.")
    
    try:
        paths = get_default_paths()
        
        # Use provided paths or defaults
        documents_dir = request.documents_dir or paths['documents_dir']
        index_config_file = request.index_config_file or paths['index_file']
        
        # Load index configuration
        with open(index_config_file, 'r') as f:
            index_config = json.load(f)
        
        # Update config with parameters
        index_config['parameters']['input']['data'] = [documents_dir]
        if request.reindex is not None:
            index_config['parameters']['input']['rag']['reindex'] = request.reindex
        
        # Index the documents
        api_data_index = APIData(**index_config)
        colette_api.service_index(app_name, api_data_index)
        
        return {
            "status": "success",
            "message": "Documents indexed successfully",
            "documents_dir": documents_dir,
            "reindex": request.reindex
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing documents: {str(e)}")

@app.post("/service/query")
async def query_rag(request: QueryRequest):
    """Query the Vision RAG system"""
    if not service_created:
        raise HTTPException(status_code=400, detail="Service not created. Call /service/create first.")
    
    try:
        # Prepare query
        query_api_msg = {
            'parameters': {
                'input': {
                    'message': request.message
                }
            }
        }
        
        # Execute query
        query_data = APIData(**query_api_msg)
        response = colette_api.service_predict(app_name, query_data)
        
        # Process sources
        sources = []
        for item in response.sources.get('context', []):
            source_item = {
                'key': item['key'],
                'distance': item['distance'],
                'source': item['source']
            }
            
            # Include image data if requested
            if request.return_images:
                source_item['image_base64'] = item['content']
            
            sources.append(source_item)
        
        return {
            "status": "success",
            "output": response.output,
            "sources": sources,
            "num_sources": len(sources)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying RAG: {str(e)}")

@app.get("/service/status")
async def service_status():
    """Get the current status of the service"""
    return {
        "service_created": service_created,
        "app_name": app_name if service_created else None,
        "colette_root": colette_root
    }

@app.post("/service/reset")
async def reset_service():
    """Reset the service (for development/testing)"""
    global service_created
    service_created = False
    return {
        "status": "success",
        "message": "Service reset. Call /service/create to reinitialize."
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)