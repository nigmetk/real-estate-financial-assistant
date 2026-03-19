# ----------------------------------------
# Property Database Queries
# ----------------------------------------

from database.db_connection import get_connection


def get_all_properties():
    """
    Get all properties from database
    """

    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM properties"

    cursor.execute(query)

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows


def get_properties_by_city(city):
    """
    Get properties for a specific city
    """

    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT p.address, p.metro_area, p.property_type, f.revenue
    FROM properties p
    JOIN financials f
    ON p.property_id = f.property_id
    WHERE p.metro_area = %s
    """

    cursor.execute(query, (city,))

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows



    if __name__ == "__main__":

    data = get_properties_by_city("Chicago")

    for row in data:
        print(row)