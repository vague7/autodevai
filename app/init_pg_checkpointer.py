# init_pg_checkpointer.py
import os   
from langgraph.checkpoint.postgres import PostgresSaver, ConnectionPool
from dotenv import load_dotenv
from psycopg import Connection

#https://github.com/langchain-ai/langgraph/issues/2887

load_dotenv()

db_uri = os.getenv("DB_URI")

if db_uri is None:
    print("DB_URI is not set, we'll use InMemoryCheckpointer instead!")
else: 
    print("DB_URI is set, we'll use PostgresSaver instead!")
    print("DB_URI: ", db_uri)

    # Create connection pool
    # pool = ConnectionPool(db_uri)
    conn = Connection.connect(db_uri, autocommit=True)

    # Create the saver
    checkpointer = PostgresSaver(conn)

    # This runs DDL like CREATE TABLE and CREATE INDEX
    # including CREATE INDEX CONCURRENTLY, which must be run outside a transaction
    checkpointer.setup()

    print("✅ Checkpointer tables & indexes initialized.")