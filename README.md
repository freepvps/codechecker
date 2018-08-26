Model using GPU if can

# Install 
```
pip install --upgrade --user gensim numpy tensorflow
```

# Index building
For data-set
```
./build_index
```

For any repository
```
python src/indexmaker/__init__.py -i path/to/dir -o path/to/index.json
```

# Model training
```
python src/checkconfmaker/__init__.py -d path/to/dataset -o file/to/trained [OPTIONS...]
```

Options
```
-d path - dataset path
-o path - output model file
-s number - split repository by groups with {-s} files
-v float - cross-validation %
-t 'group'/'repository' - cross-validation by groups/repositories
-r number - random-seed
```
