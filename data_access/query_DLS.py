"""Helper functions for querying the DLS database.

For details of the Deep Lens Survey databases, read 
http://matilda.physics.ucdavis.edu/working/website/catalogaccess.html

Many thanks to Debbie Bard for an example SQL script.

"""
from __future__ import (print_function)
import sys
import argparse
import mysql.connector 

class DLS:
    def __init__(self, database_name="RC1c_public", password="", user="guest"):
        self.all_databases = ["RC1c_public", "DLSref", "RC1Stage"]

        self.user = user 
        self.password = password
        self.database = database_name
        self.connector = None
        self.cursor = None

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
        self.cursor = self.connector.cursor()
        return 

    def print_db_tables(self, database=None): 
        if database is None:
            database = self.database

        self.cursor.execute("USE %s" % database)
        self.cursor.execute("SHOW TABLES")
        tables = [t[0] for t in self.cursor.fetchall()]

        print ("Printing tables from the database named ", database)
        map(print, tables)
        print ("")

        return 



if __name__ == "__main__":
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

    dls_db = DLS(user=user, password=password) 

    dls_db.print_db_tables(database=database)

    ### Read SQL query from file.
    # sql_file = "DBard_shear_peak.sql"
    sql_file = "test.sql"
    sql_fstream = open(sql_file, 'r')
    sql_query = ' '.join(sql_fstream.readlines())
    print ("The read-in sql query is:\n", sql_query)
    sql_query = sql_query.replace('\r', '').replace('\n', ' ')

    dls_db.cursor.execute(sql_query)
    print ("Printing queried results:\n")
    for (alpha, delta) in dls_db.cursor:
        print ("{0}, {1}".format(alpha, delta))

