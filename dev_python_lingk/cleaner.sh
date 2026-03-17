#/bin/sh

find . -name .ipynb_checkpoints | xargs rm -rf
find . -name __pycache__ | xargs rm -rf
rm -rf lingk_output/ 
rm -rf .pytest_cache/ 

#jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace examples/*.ipynb
#jupyter nbconvert --to python examples/*.ipynb
