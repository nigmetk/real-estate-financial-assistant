# ----------------------------------------
# Database Connection
# ----------------------------------------

import psycopg2


def get_connection():
    """
    Create connection to PostgreSQL database
    """

    conn = psycopg2.connect(
        host="localhost",
        database="real_estate_db",
        user="postgres",
        password="postgres123",
        port="5432"
    )

    return conn