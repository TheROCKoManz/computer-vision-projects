import mysql.connector
import sys
sys.path.append('Facial_Recognition/')
from Database_Connect.utils import DB_Connect, DB_Disconnect
db_config = {
    'host':'176.9.99.185',
    'user':'computervision',
    'password':'Computervision@253#87',
    'database':'user_db_computer_vision',
    'ssl_disabled': True
}

def insert_user_info(user_info):
    print(user_info)

    connection = DB_Connect()
    cursor = connection.cursor()
    try:

        insert_query = """
            # INSERT INTO UserInfo (FirstName, LastName, Age, Ethnicity, Gender)
            # VALUES (%s, %s, %s, %s, %s, %s)
            # """
        cursor.execute(insert_query, (user_info['first_name'], user_info['last_name'], user_info['age'], user_info['ethnicity'], user_info['gender']))
        connection.commit()
        print(f'''...User Recorded...
        FirstName: {user_info["first_name"]}, 
        LastName: {user_info['last_name']}, 
        Age: {user_info['age']}, 
        Race: {user_info['ethnicity']}, 
        Gender: {user_info['gender']})''')

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    cursor.close()
    DB_Disconnect(connection)
    print('done')







