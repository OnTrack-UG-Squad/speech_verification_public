import io
import logging
import sqlite3
from os.path import abspath, dirname, join, exists

import numpy as np

from speaker_verification.utils.logger import SpeakerVerificationLogger

DATABASE_PATH = join(abspath(dirname(__file__)), "SQL", "sqlite.db")
logger = SpeakerVerificationLogger(name=__file__)
logger.setLevel(logging.INFO)


class DatabaseError(Exception):
    pass


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


def get_db_connection(database=DATABASE_PATH):
    sqliteConnection = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = sqliteConnection.cursor()
    return sqliteConnection, cur

def establish_sqlite_db(table_name, database=DATABASE_PATH):
    if not exists(database):
        sqlite3.connect(database.split('/')[-1]).close()
        create_db_table(table_name)


def read_sqlite_table(table, database=DATABASE_PATH):
    """read_sqlite_table.

    print all records within users table.

    Parameters
    ----------
    table : str
        Name of table to read record from.
    """
 
    try:
        _, cur = get_db_connection(database)
        sqlite_select_query = f"select * from {table}"
        cur.execute(sqlite_select_query)
        records = cur.fetchall()
        for row in records:
            logger.info("Id: ", row[0])
            logger.info("mfcc: ", type(row[1]))

    except sqlite3.Error as error:
        logger.error("Failed to read data from sqlite table", error)
        raise DatabaseError()


def create_db_table(table: str, database=DATABASE_PATH):
    """create_db_table.

    Creates a table within sqlite database to store user records.

    Parameters
    ----------
    table : str
        Name of table to create.
    """
    try:
        _, cur = get_db_connection(database)
        cur.execute(f"create table {table}(id integer primary key, arr array)")
    except Exception as err:
        logger.error(f"Cannot create table for {table}: ", err)
        raise DatabaseError()


def remove_db_row(table: str, id: int, database=DATABASE_PATH):
    """remove_db_row.

    Removes row within sqlite table according to "id" and "table" parameters.

    Parameters
    ----------
    table : str
        Name of table to remove record from.
    id : str
        Id key for required record within table for removal.
    """
    try:
        _, cur = get_db_connection(database)
        cur.execute(f"delete from {table} where id={id}")
    except Exception as err:
        logger.error(f"Database row doesn't exist for id ({id}) in table ({table}): ", err)
        raise DatabaseError()


def select_db_row(table: str, id: int, database=DATABASE_PATH):
    """select_db_row.

    Selects and prints out a row within a registered sqlite database table.

    Parameters
    ----------
    table : str
        Name of table to select record from.
    id : str
        Id key for required record within table for selection.
    """
    try:
        _, cur = get_db_connection(database)
        rows = cur.execute(f"select * from {table} where id={id}")
        for row in rows:
            return row
    except Exception as err:
        logger.error("Database Error: ", err)


def insert_db_row(table: str, id: int, mfcc: np.array, database=DATABASE_PATH):
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
    try:
        con, cur = get_db_connection(database)
        cur.execute(f"insert into {table}(id, arr) values (?, ?)", (id, mfcc,))
        con.commit()
    except Exception as err:
        logger.error("Database Error: ", err)
        raise DatabaseError()
