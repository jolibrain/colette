```markdown
# Tools

This directory contains tools that support Colette.



## Evaluation

To assess Colette's performance on your custom benchmark, run the `evaluation.py` script. First, prepare a file containing your reference results in the format shown below:

```json
[
  {
    "id": 1,
    "question": "Quelle est la durée pour libérer les orbites après la fin de vie ?", // the question
    "answer": "La durée pour libérer les orbites après la fin de vie est de 25 ans.", // the reference answer
    "short_answer": [
      "25 ans" // expected key phrase(s) in the answer
    ],
    "references": [
      {
        "file": "grdébris-xx-05", // file expected to assist in generating the answer
        "pages": [
          "27"                    // specific page from the document
        ]
      }
    ],
    "lang": "fr"
  },
...
]
```

Once your reference file is ready, launch the evaluation using the following command:

```python
python tools/evaluation.py --app-dir path/to/app/directory --qa path/to/questions_answers_file.json
```

This command creates a new timestamped directory containing three files:

`app_basename_llm.json`: LLM performance metrics (including Rouge and BERT scores)
`app_basename_retriever.csv`: Retriever performance data
`app_basename_results.txt`: A summary of all performance metrics


## kvstore

The `kvstore` tool assists in investigating the key-value store (kvstore) that holds all images used in the Multimodal RAG.

### Usage

```bash
python kvtool.py [OPTIONS] COMMAND [ARGS]...
```

A command-line tool with multiple options.

**Options:**

- `--help`: Show the help message and exit.

**Commands:**

- `check`: Run a kvstore check.
- `extract`: Extract data with a key.
- `info`: Display information about the application.
- `migrate`: Run the migration process.

#### Command: `migrate`

```bash
python kvtool.py migrate [OPTIONS]
```

Run the migration process.

**Options:**

- `--app-dir TEXT`: Specify the application directory. **[required]**
- `--help`: Show the help message and exit.

#### Command: `check`

```bash
python kvtool.py check [OPTIONS]
```

Run a kvstore check.

**Options:**

- `--app-dir TEXT`: Specify the application directory. **[required]**
- `--help`: Show the help message and exit.

#### Command: `extract`

```bash
python kvtool.py extract [OPTIONS]
```

Extract data with a key.

**Options:**

- `--app-dir TEXT`: Specify the application directory. **[required]**
- `--key TEXT`: Specify the extraction key. **[required]**
- `--help`: Show the help message and exit.

#### Command: `info`

```bash
python kvtool.py info [OPTIONS]
```

Display information about the application.

**Options:**

- `--app-dir TEXT`: Specify the application directory. **[required]**
- `--key TEXT`: Optional key for getting more information about a specific key.
- `--help`: Show the help message and exit.
