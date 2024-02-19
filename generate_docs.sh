# Add to conf.py
# import os
# import sys
# sys.path.insert(0, os.path.abspath('..'))

# extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']
# todo_include_todos = True

# html_theme = 'sphinx_rtd_theme'

# Add modules to index.rst
# .. toctree::
#    :maxdepth: 2
#    :caption: Contents:
#    modules

sphinx-apidoc -o sphx samecode/
cd ./sphx/
make html
mv ./_build/html/* ../docs/