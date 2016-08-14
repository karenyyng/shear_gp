#!/usr/bin/env python
"""Helper functions for querying the DLS_server database.
The `DLS_server` class of this script can
* send SQL queries with a certain username and password
* query tables
* query the schema of each table
* save the results into either a CSV or a compressed HDF5 table file using Pandas
* if you do decide to use wild card selection of a table, you need to put the
wild card as the LAST entry in the select

For details of the Deep Lens Survey databases, read
http://matilda.physics.ucdavis.edu/working/website/catalogaccess.html

Many thanks to Debbie Bard for an example SQL script.
"""
from __future__ import (print_function)
import argparser
import mysql.connector
import sys


class Database:
    def __init__(self, name, dls_instance, tables=None):
        self.name = name
        if tables is None:
            dls_instance.cursor.execute("USE %s" % name)
            dls_instance.cursor.execute("SHOW TABLES")
            tables = [t[0] for t in dls_instance.cursor.fetchall()]

        self.tables = {t: None for t in tables}

    def __construct_schema_for__(self, table, dls_instance):
        if table not in self.tables.keys():
            raise ValueError("Supplied `table` = {} ".format(table) +
                             "not in list of tables"
                             )
        self.tables[table] = \
            dls_instance.get_query_results("DESCRIBE {}".format(table))

class DLS_server:
    def __init__(self, database_name="RC1c_public", password="", user="guest"):
        self.all_databases = ["RC1c_public", "DLS_serverref", "RC1Stage"]

        self.user = user
        self.password = password
        self.connector = None
        self.cursor = None
        self.config = {
            "user": self.user,
            "password": self.password,
            "host": "matilda.physics.ucdavis.edu",
            "database": database_name
        }
        self.__connect_to_database__()
        self.database = Database(database_name, self)

    def __del__(self):
        self.cursor.close()
        self.connector.close()

    def __connect_to_database__(self):
        # Make connection using the specified config.
        self.connector = mysql.connector.connect(**self.config)

        # While you can do self.cursor.execute(SQL_COMMAND)
        # You will run into error if you don't retrieve results after
        # running `self.cursor.execute(SQL_COMMAND)`
        # Use DLS_server.get_query_results(SQL_COMMAND) instead
        self.cursor = self.connector.cursor()
        return

    def __grab_column_name_from_sql_query__(self, sql_query):
        col_name_lines = \
            [n for n, q in enumerate(sql_query) if 'SELECT' in q][0], \
            [n for n, q in enumerate(sql_query) if 'FROM' in q][0]

        col_names = sql_query[col_name_lines[0] + 1: col_name_lines[1]]
        return [c.replace(',', '') for c in col_names]


    def print_db_tables(self, database=None, verbose=False):
        if database is None:
            database = self.database
            if verbose:
                print ("Using database = ", database)

        print ("Printing tables from the database named ", database)
        tables = self.database.tables.keys()
        map(print, zip(range(len(tables)), tables))
        print ("")

        return

    def print_table_schema(self, table):
        """Lazily (delayed) construct and print table schema for users to view

        :param table: string, name of the table to check
        """
        if self.database.tables[table] is None:
            self.database.__construct_schema_for__(table, self)
        map(print, self.database.tables[table])

        return

    def process_sql_file(self, sql_file, verbose=True):
        """
        sql_file: string, full path to file containing sql query
        """
        sql_fstream = open(sql_file, 'r')
        # Only include lines that are not comments
        sql_lines = [l.strip() for l in sql_fstream.readlines()
                     if l[0] != '#']

        self.sql_column_name = \
            self.__grab_column_name_from_sql_query__(sql_lines)

        sql_query = ' '.join(sql_lines).strip()

        if verbose:
            print ("Extracted col names is ", self.sql_column_name)
            print ("The read-in sql query is:\n", sql_query)

        self.sql_query = sql_query.replace('\r', '').replace('\n', ' ')
        return

    def get_query_results(self, sql_query, verbose=True):
        if verbose:
            print ("Querying database with SQL command:")
            print (sql_query)
        self.cursor.execute(sql_query)
        return [list(i) for i in self.cursor]

    def fix_wild_card_col_names(self, table):
        if self.database.tables[table] is None:
            self.database.__construct_schema_for__(table, self)
        cols = [c[0] for c in self.database.tables[table]]
        self.sql_column_name = self.sql_column_name[:-1] + cols
        return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: query_DLS.py integer\n"
                         + "Integer = 0 means don't read SQL query. "
                         + "Integer > 0 means read SQL query."
                         )
    read_sql_file = int(sys.argv[1])
    fix_wild_card_col_name = True
    fix_table = 'Probs'
    sql_file = "get_p_z.sql"

    try:
        import pandas as pd
        import tables as pytables
        pandas_exists = True

    except ImportError:
        print ("Either pandas or tables (PyTables) was NOT imported successfully. " +
               "Outputting CSV with NumPy instead.")
        pandas_exists = False

    try:
        password_file = "RC1Stage_password.txt"
        with open(password_file, 'r') as f:
            password = f.read()
            password = password.strip()
            print ("The password is read from " +
                   " {0} and is `{1}`.\n".format(password_file, password))
            f.close()
            user = "DLS"
            database = "RC1Stage"

    except IOError:
        print ("No password file was found. Using the default password.")
        user = "guest"
        password = ""
        database = "RC1c_public"

    dls_db = DLS_server(user=user, password=password)
    # dls_db.print_db_tables(database=database)

    if read_sql_file:
        # Read SQL query from file.
        # sql_file = "test.sql"
        dls_db.process_sql_file(sql_file, verbose=False)

        query_results = dls_db.get_query_results(dls_db.sql_query)

        output_file_prefix = "F5_gold_sample"
        if fix_wild_card_col_name:
            dls_db.fix_wild_card_col_names(fix_table)
            # print ("colnames after fixing = ", dls_db.sql_column_name)

        if pandas_exists:
            df = pd.DataFrame(query_results,
                              columns=dls_db.sql_column_name
                              )
            # convert the subfield string to int
            if 'subfield' in df.keys():
                df['subfield'] = df["subfield"].map(lambda l: int(l[1]))
            complevel = 9
            complib = 'zlib'
            pandas_df_key = 'df'
            print ("Outputting content to {}".format(output_file_prefix + ".h5"))
            df.to_hdf(output_file_prefix + ".h5", pandas_df_key,
                      complevel=complevel,
                      complib=complib)
            # To read the file, use
            # > df = pd.read_hdf(output_file_prefix + ".h5", "df")

        else:
            import numpy as np
            results = np.array(query_results)
            header = ','.join(dls_db.sql_column_name)
            print ("Outputting content to {}".format(output_file_prefix + ".csv"))
            np.savetxt(output_file_prefix + ".csv", results, fmt="%3.10f",
                       delimiter=",", header=header)
