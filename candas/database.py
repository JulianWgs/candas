from sqlalchemy import MetaData, Table, Column
from sqlalchemy import (Float, BigInteger, Integer, Boolean,
                        String, DateTime, Date, Time)


def initialize_database(engine, dbc_db):
    """
    Initialize SQL database with logging and datatables for dbc database.

    Only initialize database when message and signal names and the version
    of the dbc database are final.

    Parameters
    ----------
    engine : sqlalchemy.engine.
        engine of the database to which the data should be pushed. More
        information here: docs.sqlalchemy.org/en/13/core/engines.html.
    dbc_db : cantools.db.Database
        The dbc database which was used to convert the data from a binary
        format. This databse contains all the data for the table and column
        names.

    Raises
    ------
    RuntimeError
        When the database is initialized partially.

    Returns
    -------
    None.

    """
    connection = engine.connect()
    metadata = MetaData(bind=connection)
    metadata.reflect()
    print("Tables in database:")
    for table in metadata.tables:
        print(table)
    connection.close()

    # Check if all logging tables exist
    logging_tables = ["sessions", "uploads", "errors"]
    tables_exist = check_if_tables_exist(metadata, logging_tables)
    if not all(tables_exist) and any(tables_exist):
        raise RuntimeError("Logging tables not complete nor missing! "
                           "Please fix manually.")
    if all(tables_exist):
        print("Logging tables already exist.")
    else:
        create_logging_tables(engine)

    # Get all message names from dbc
    # table name are dbc version message name, separated by an underscore
    # signals are the columns
    message_names = []
    for message in dbc_db.messages:
        message_names.append(dbc_db.version + "_" + message.name)
    tables_exist = check_if_tables_exist(metadata, message_names)
    if not all(tables_exist) and any(tables_exist):
        raise RuntimeError("Data tables not complete nor missing! "
                           "Please fix manually.")
    if all(tables_exist):
        print("Data tables already exist.")
    else:
        create_data_tables(engine, dbc_db)


def check_if_tables_exist(metadata, table_names):
    """
    Return array with booleans if corresponding table exists.

    Parameters
    ----------
    metadata : sqlalchemy.sql.schema.MetaData
        Collection of metadata entities like Tables and Columns is stored in
        MetaData. This is used to access the datbase. More information here
        docs.sqlalchemy.org/en/13/core/metadata.html.
    table_names : :obj:`list` of :obj:`str`
        List of the table names to check.

    Returns
    -------
    tables_exist : :obj:`list` of :obj:`bool`
        List of boolean. "True" means the table exists.

    """
    tables_exist = []
    for table_name in table_names:
        if table_name in metadata.tables:
            tables_exist.append(True)
        else:
            tables_exist.append(False)
    return tables_exist


def create_logging_tables(engine):
    """
    Create the logging tables, where uploads and session are being logged.

    Logging tables are universally used by every dbc database.

    Parameters
    ----------
    engine : sqlalchemy.engine.
        engine of the database to which the data should be pushed. More
        information here: docs.sqlalchemy.org/en/13/core/engines.html.

    Returns
    -------
    None.

    """
    print("Creating logging tables...")
    connection = engine.connect()
    metadata = MetaData(bind=connection)
    # Reflection not needed
    Table('sessions', metadata,
          Column("session_id", Integer, primary_key=True),
          Column("name", String(length=30), nullable=False),
          Column("dbc_version", String(length=4), nullable=False),
          Column("dbc_commit_hash", String(length=40)),
          Column("file_hash_blf", String(length=64)),
          Column("file_hash_mat", String(length=64)),
          Column('date', Date),
          Column("start_time", Time),
          Column("description", String(140)),
          Column("upload_date_time", DateTime),
          Column("event", String(length=30)),
          Column("location", String(length=30)),
          )
    Table('uploads', metadata,
          Column("id", BigInteger, primary_key=True),
          Column("session_id", Integer),
          Column("message_name", String(length=40), nullable=False),
          Column("signal_name", String(length=40), nullable=False),
          Column("date_time", DateTime, nullable=False),
          )
    Table('errors', metadata,
          Column("id", BigInteger, primary_key=True),
          Column("session_id", Integer),
          Column("message_name", String(length=40), nullable=False),
          Column("signal_name", String(length=40), nullable=False),
          Column("date_time", DateTime, nullable=False),
          )
    metadata.create_all(engine)
    connection.close()
    print("Done.")


def create_data_tables(engine, dbc_db, ignored_messages=None, verbose=False):
    """


    Parameters
    ----------
    engine : sqlalchemy.engine.
        engine of the database to which the data should be pushed. More
        information here: docs.sqlalchemy.org/en/13/core/engines.html.
    dbc_db : cantools.db.Database
        The dbc database which was used to convert the data from a binary
        format. This databse contains all the data for the table and column
        names.
    ignored_messages : :obj:`list` of :obj:`str`, optional
        A list of messages in the dbc database which should be ignored.
        The default is None.
    verbose : bool, optional
        Set to False to have no readout. The default is False.

    Returns
    -------
    None.

    """
    if ignored_messages is None:
        ignored_messages = []
    connection = engine.connect()
    metadata = MetaData(bind=connection)
    metadata.reflect()
    print("Creating data tables..")
    for message in dbc_db.messages:
        if verbose:
            print(message.name)
        if message.name in ignored_messages:
            continue
        signals = [
            Column("id", BigInteger, primary_key=True),
            Column("session_id", Integer, nullable=False),
            Column("time", Float, nullable=False),
        ]
        for signal in message.signals:
            # TODO: Investigate why choices is defined twice
            if signal.choices:
                for choice in signal.choices:
                    max_len = 0
                    for choice in signal.choices.values():
                        if max_len < len(choice):
                            max_len = len(choice)
                data_type = String(length=max_len)
            elif signal.length == 1:
                data_type = Boolean
            else:
                data_type = Float

            signals.append(Column(signal.name, data_type))
            if verbose:
                print("\t" + signal.name, data_type)
        Table(dbc_db.version + "_" + message.name, metadata, *signals)
        metadata.create_all(engine)
    connection.close()
    print("Done")
