import mysql.connector

from Database_Connect.utils import DB_Disconnect, DB_Connect

def Register_Model(modelID, targets, timestamp, accuracy, loss):

    connection = DB_Connect()
    cursor = connection.cursor()
    try:

        insert_query = """
            INSERT INTO Model_Registry (ModelID, TargetIDs, Train_timestamp, Accuracy, Loss)
            VALUES (%s, %s, %s, %s, %s)
        """


        targetsIDs = ', '.join(list(targets.keys()))

        data = (modelID, targetsIDs, timestamp, accuracy, loss)
        print(data)
        cursor.execute(insert_query, data)
        connection.commit()
        print('\nModel Registered!')
    except mysql.connector.Error as err:
        print(f"Error: {err}")

    cursor.close()

    DB_Disconnect(connection)