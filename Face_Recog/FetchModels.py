import mysql.connector
import pandas as pd
from Database_Connect.utils import DB_Disconnect, DB_Connect

def Fetch_Models():

    connection = DB_Connect()
    cursor = connection.cursor()
    try:

        fetch_query = """
            select * from Model_Registry;
        """
        cursor.execute(fetch_query)
        results = pd.DataFrame(cursor.fetchall(), columns=['Model_ID', 'Targets', 'Created_At', 'Accuracy', 'Loss'])
        print(results)

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    cursor.close()
    DB_Disconnect(connection)
    print('done')

if __name__ == "__main__":
    Fetch_Models()