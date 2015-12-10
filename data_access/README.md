## What is needed for running the `query_DLS.py` script:  
* [sql.connector](https://pypi.python.org/pypi/mysql-connector-python-rf/2.1.3)
* numpy 

Optional dependencies:   
* Python Pandas and PyTables 


## Installing dependencies using `pip`:
```
$ pip install mysql-connector-python-rf
```

## How to run:
```
$ python query_DLS.py 0 
```
Argument `0` means don't read in SQL query from file. 
This is for interactive debugging 

# How to use within IPython
```
> %run query_DLS.py 0  # this creates a `dls_db` object 
> dls_db.print_db_tables() 
Printing tables from the database named  RC1c_public
(0, u'Bpz')
(1, u'Coords')
(2, u'Dlsqc')
(3, u'PhotoObj')
(4, u'Probs')
(5, u'Sextractor')
(6, u'SpecZ')
(7, u'StarDb')
```
# For example, if you are Michael Schneider and you 
# like to query all the p(z) 
> dls_db.print_table_schema('Probs')
Querying database with SQL command:
DESCRIBE Probs
[u'subfield', u'char(5)', u'NO', u'', u'', u'']
[u'NUMBERR', u'bigint(20)', u'NO', u'', u'0', u'']
[u'c0', u'float', u'YES', u'', None, u'']
[u'c1', u'float', u'YES', u'', None, u'']
[u'c2', u'float', u'YES', u'', None, u'']
[u'c3', u'float', u'YES', u'', None, u'']
[u'c4', u'float', u'YES', u'', None, u'']
[u'c5', u'float', u'YES', u'', None, u'']
[u'c6', u'float', u'YES', u'', None, u'']
[u'c7', u'float', u'YES', u'', None, u'']
< OUTPUT OMITTED>
[u'c499', u'float', u'YES', u'', None, u'']
[u'objid', u'bigint(20)', u'YES', u'', None, u'']
```


