# Usage

## Parepare your database

### Unix
```
python ii_map.py sample.txt | sort -k1,1 | python ii_reduce.py > index.tsv
```

### Windows
```
python ii_map.py sample.txt | sort | python ii_reduce.py > index.tsv
```

## Run you search engine

```
python search.py index.tsv
```

![demo](demo.png)