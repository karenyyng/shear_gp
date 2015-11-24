"""Helper functions for querying the DLS_server database.

For details of the Deep Lens Survey databases, read
http://matilda.physics.ucdavis.edu/working/website/catalogaccess.html

Many thanks to Debbie Bard for an example SQL script.

"""
from __future__ import (print_function)
import mysql.connector


class Database:
    def __init__(self, name, tables):
        self.name = name
        self.tables = {t: {} for t in tables}

    def __construct_schema_for_(self, table):
        if table not in self.tables:
            raise ValueError("Suppled `table` = {} ".format(table) +
                             "not in list of tables"
                             )


class DLS_server:
    def __init__(self, database_name="RC1c_public", password="", user="guest"):
        self.all_databases = ["RC1c_public", "DLS_serverref", "RC1Stage"]

        self.user = user
        self.password = password
        self.database = database_name
        self.connector = None
        self.cursor = None
        self.tables = {}

        self.config = {
            "user": self.user,
            "password": self.password,
            "host": "matilda.physics.ucdavis.edu",
            "database": self.database
        }

        self.__connect_to_database__()

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

        return sql_query[col_name_lines[0] + 1: col_name_lines[1]]

    def print_db_tables(self, database=None, verbose=False):
        if database is None:
            database = self.database
            if verbose:
                print ("Using database = ", database)

        self.cursor.execute("USE %s" % database)
        self.cursor.execute("SHOW TABLES")
        tables = [t[0] for t in self.cursor.fetchall()]

        print ("Printing tables from the database named ", database)
        map(print, zip(range(len(tables)), tables))
        self.db = Database(database, tables)
        print ("")

        return

    def print_table_schema(self, table):
        """Lazily (delayed) construct and print table schema for users to view

        :param table: string, name of the table to check
        """
        table_schema = self.get_query_results("DESCRIBE {}".format(table))
        map(print, table_schema)
        return

    def process_sql_file(self, sql_file, verbose=True):
        """
        sql_file: string, full path to file containing sql query
        """
        sql_fstream = open(sql_file, 'r')
        # Only include lines that are not comments
        sql_lines = [l.strip() for l in sql_fstream.readlines()
                     if l[0] != '#']

        col_names_in_query = self.__grab_column_name_from_sql_query__(sql_lines)
        print ("col_names = ", col_names_in_query)

        sql_query = ' '.join(sql_lines).strip()

        if verbose:
            print ("The read-in sql query is:\n", sql_query)

        sql_query = sql_query.replace('\r', '').replace('\n', ' ')
        return sql_query

    def get_query_results(self, sql_query):
        self.cursor.execute(sql_query)
        return [list(i) for i in self.cursor]


def save_results_to_pandas_DataFrame(results, column_headers):
    return


if __name__ == "__main__":
    try:
        import h5py
        h5py_exist = True

    except ImportError:
        print ("No h5py was found, outputting CSV instead.")
        h5py_exist = False

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

    try:
        import pandas as pd
    except ImportError:
        print ("Pandas ")

    dls_db = DLS_server(user=user, password=password)

    dls_db.print_db_tables(database=database)

    # Read SQL query from file.
    sql_file = "DBard_shear_peak.sql"
    sql_query = dls_db.process_sql_file(sql_file)
    # sql_file = "test.sql"

    query_results = dls_db.get_query_results(sql_query)
    # print ("Printing queried results:\n")
    # for (alpha, delta) in dls_db.cursor:
    #     print ("{0}, {1}".format(alpha, delta))
