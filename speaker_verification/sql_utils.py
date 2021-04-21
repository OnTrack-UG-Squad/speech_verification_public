import io
import numpy as np
import sqlite3
from os.path import abspath, dirname, join, exists

DATABASE_PATH = join(abspath(dirname(__file__)), "SQL", "sqlite.db")


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)


def establish_sqlite_db(table_name):
    if not exists(DATABASE_PATH):
        sqlite3.connect(DATABASE_PATH.split('/')[-1]).close()
        create_db_table(table_name)


def read_sqlite_table(table):
    """read_sqlite_table.

    print all records within users table.

    Parameters
    ----------
    table : str
        Name of table to remove record from.
    """
    if isinstance(table, str):
        try:
            sqliteConnection = sqlite3.connect(DATABASE_PATH)
            cur = sqliteConnection.cursor()

            sqlite_select_query = f"select * from {table}"
            cur.execute(sqlite_select_query)
            records = cur.fetchall()
            for row in records:
                print("Id: ", row[0])
                print("mfcc: ", type(row[1]))
                print("\n")

        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)
    else:
        raise TypeError("Table name must be string")


def create_db_table(table):
    """create_db_table.

    Creates a table within sqlite database to store user records.

    Parameters
    ----------
    table : str
        Name of table to create.
    """
    if isinstance(table, str):
        try:
            with sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES) as con:
                cur = con.cursor()
                cur.execute(f"create table {table}(id integer primary key, arr array)")
        except Exception as err:
            print(f"Cannot create table for {table}: ", err)
    else:
        raise TypeError("Only strings are allowed")


def remove_db_row(table, id):
    """remove_db_row.

    Removes row within sqlite table according to "id" and "table" parameters.

    Parameters
    ----------
    table : str
        Name of table to remove record from.
    id : str
        Id key for required record within table for removal.
    """
    if isinstance(table, str) and isinstance(id, int):
        try:
            with sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES) as con:
                cur = con.cursor()
                cur.execute(f"delete from {table} where id={id}")
        except Exception as err:
            print(f"Database row doesn't exist for id ({id}) in table ({table}): ", err)
    else:
        raise TypeError("Ensure Table and ID are of vaild data type")


def select_db_row(table, id):
    """select_db_row.

    Selects and prints out a row within a registered sqlite database table.

    Parameters
    ----------
    table : str
        Name of table to select record from.
    id : str
        Id key for required record within table for selection.
    """
    id, id_is_int = try_parse_to_int(id)

    if isinstance(table, str) and id_is_int:
        try:
            with sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES) as con:
                cur = con.cursor()
                rows = cur.execute(f"select * from {table} where id={id}")
                for row in rows:
                    return row
        except Exception as err:
            print("Database doesn't exist: ", err)
    else:
        raise TypeError("Table name must be string and ID must be integer")


def insert_db_row(table, id, mfcc):
    """insert_db_row.

    Takes required parameters and inserts a record of given id and mfcc dataset into the sqlite database table specified.

    Parameters
    ----------
    table : str
        Name of table to insert record within.
    id : int
        Id key for required record within table for insertion.
    mfcc : numpy.array
        MFCC dataset to be inserted within database records.
    """
    length = len(str(id))
    if length == 9:
        id, id_is_int = try_parse_to_int(id)

        if isinstance(table, str) and id_is_int:
            try:
                with sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES) as con:
                    cur = con.cursor()
                    cur.execute(f"insert into {table}(id, arr) values (?, ?)", (id, mfcc,))
            except Exception as err:
                print("Database doesn't exist: ", err)
        else:
            raise TypeError("Table name must be string and ID must be integer")
    else:
        raise Exception("Invalid Input")


def try_parse_to_int(value):
    try:
        return int(value), True
    except ValueError:
        return value, False
