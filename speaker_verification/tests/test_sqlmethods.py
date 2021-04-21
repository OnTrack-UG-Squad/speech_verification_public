import unittest

import pytest
import numpy as np
from os.path import join, abspath, dirname
import pathlib
import sqlite3

from speaker_verification.sql_utils import(
    create_db_table,
    insert_db_row,
    select_db_row,
    remove_db_row
)

table_id = 219944922
mfcc=np.array([1,2,3])

def test_database():
    """Method test for valid datbase"""
    try:
        assert select_db_row("asbjasn",111111111)
    except AssertionError:
        return "Datbase does not exist"

def test_create_table():
    """Method test to check if table can be created"""
    try:
        assert create_db_table("mock_table")
    except AssertionError:
        return "Could not establish connection"

def test_create_invalid_table():
    """ Method test to check if table will be created by sending invaid name
        by passing through the parameter as an integer"""
    with pytest.raises(TypeError):
        create_db_table(112223444)

def test_insert_table(): 
    '''Method test to check by trying to unsert values in the table 
        by passing in valid values for each parameter'''
    try:
        assert insert_db_row("mock_table",table_id,mfcc)
    except AssertionError:
        return "Insertion error."

def test_insert_invalid_table():
    '''Method test to check by trying to insert the row passing in 
        table with valid values for each of the columns or with invalid table name
    '''
    with pytest.raises(TypeError):
        insert_db_row(1223,219249449,2331212)
    with pytest.raises(Exception):
        insert_db_row("mock_table",211221,1123)

def test_select_row():
    '''Method test to check by trying to select the row passing in 
            valid values for each of the columns
    '''
    try:
        assert select_db_row("hello",123344433)
    except AssertionError:
        return "Error selecting the row."
        
def test_select_invalid_row():
    '''Method test to check by trying to select the row passing in 
    invaid values for each of the columns
    '''
    with pytest.raises(TypeError):
       select_db_row(11223355,123456789)
       select_db_row("mock_table", "124565")

def test_remove_db_row():
    '''Method test to remove the data from table with valid 
    values for each of the column names
    '''
    try:
        assert remove_db_row("mock_table",table_id)
    except AssertionError:
        return "Unable to remove row in db table."
   
def test_remove_invaild_db_row():
    '''Method test to remove the data from table with invalid 
    values for each of the column names
    '''
    with pytest.raises(TypeError):
        remove_db_row(1223,219249449)
    with pytest.raises(TypeError):
        remove_db_row("mock_table","211221")
