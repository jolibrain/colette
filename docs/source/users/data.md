# Data and files

## RAG indexed repository structure

- `config.json`: the RAG configuration file, automatically copied at index creation time
- `kvstore.db`: locate database that contains images and crops from indexed documents, along with some metadata
- `mm_index`: contains the similarity index
- `pdfs`: pdf files generated from images of other types of documents (.docx, .pptx, ...)