import numpy as np
import pytest
import os

from speaker_verification.sql_utils import (
    create_db_table,
    insert_db_row,
    remove_db_row,
    select_db_row,
    get_db_connection,
    DatabaseError,
)


@pytest.fixture()
def test_db():
    """
    Pytest fixture to create a new "test" database and insert the nessesary data.
    Path of the database is passed to each test and then removed after each test.
    """
    database_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "input", "test.db"
    )
    con, _ = get_db_connection(database_path)
    con.close()
    create_db_table("unit_test", database_path)
    insert_db_row("unit_test", 219944922, np.array([1, 2, 3]), database_path)
    insert_db_row("unit_test", 219941122, np.array([1, 2, 4, 5, 3]), database_path)
    yield database_path
    # teardown
    os.remove(database_path)


def test_select_db_row_returns_expected_row(test_db):
    """Method test to check by trying to select the row passing in
       valid values for each of the columns
    """
    expected = (219944922, np.array([1, 2, 3]))
    rows = select_db_row("unit_test", 219944922, test_db)

    assert rows[0] == expected[0]
    assert all([a == b for a, b in zip(rows[1], expected[1])])


def test_select_db_row_returns_no_value(test_db):
    """Method test to check by trying to select the row passing in
      invalid values for student id that does not exist in the table
    """
    expected = None
    rows = select_db_row("unit_test", 219942322, test_db)

    assert expected == rows


def test_insert_db_row_valid(test_db):
    """Method test to check by trying to insert the row passing in
    valid values for student id and select that row using relevant method
    """
    expected = (218008432, np.array([2, 3, 4]))

    insert_db_row("unit_test", 218008432, np.array([2, 3, 4]), test_db)
    rows = select_db_row("unit_test", 218008432, test_db)

    assert rows[0] == expected[0]
    assert all([a == b for a, b in zip(rows[1], expected[1])])


def test_insert_db_row_invalid(test_db):
    """Method test to insert data to the table with invalid
    value for the column
    """
    with pytest.raises(DatabaseError):
        insert_db_row("unit_test", "string", np.array([2, 3, 4]), test_db)


def test_removes_db_row():
    """ Method test to remove the data from table with valid
    values for each of the column names
    """
    with pytest.raises(DatabaseError):
        remove_db_row("unit_test", 219941122, test_db)
