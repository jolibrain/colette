# Training a custom DSE for RAG

## Generating the dataset 
```
python ./generate_dataset.py --crops_path /path/to/already/obtained_crops/
```

This generate a `dataset.json` file of the form :
```
{
"images": ["crop1", "crops2", ...], 
"queries": [  
               {
                   "query": "query on image", 
                   "docids": [ list of index of images in crop list],
                   "answers: [ an answer per image in previous list]
               }
           ]
}
```



## Finetuning your model
```
./launch_custom.sh
```

to be modified to your needs, in particular `output_dir` and  `corpus_path`


## Usage 
In your colette call, you can simply put your `output_dir` as `parameters.input.rag.embedding_model`
