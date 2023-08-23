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


def insert_user_encodings(userID,face_encodings):
    connection = DB_Connect()
    cursor = connection.cursor()
    try:
        update_query = "UPDATE Users SET face_encodings = %s WHERE UserID = %s"
        update_data = (face_encodings, userID)

        cursor.execute(update_query, update_data)
        connection.commit()
        print(f"...Facial_Encoding_Recorded...")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    cursor.close()
    DB_Disconnect(connection)

def insert_user_info(user_info):
    connection = DB_Connect()
    cursor = connection.cursor()
    scenario = 1
    try:

        first_name = user_info['FirstName']
        last_name = user_info['LastName']
        age = int(user_info['Age'])
        gender = user_info['Ethnicity']
        ethnicity = user_info['Gender']

        # Check if the user already exists in the database based on First Name, Last Name, and Age
        check_query = '''SELECT * FROM Users WHERE FirstName = %s AND LastName = %s AND Age = %s AND Gender = %s AND Ethnicity = %s'''
        cursor.execute(check_query, (first_name, last_name, age, gender, ethnicity))
        existing_user = cursor.fetchall()


        if len(existing_user)>0:
            # User already exists, handle accordingly (e.g., display an error message)
            print("User already exists!")
        else:
            insert_query = """
                        INSERT INTO Users (FirstName, LastName, Age, Ethnicity, Gender)
                        VALUES (%s, %s, %s, %s, %s)
                    """
            data = (first_name, last_name, age, ethnicity, gender)

            cursor.execute(insert_query, data)
            connection.commit()
            print(f'''...User Recorded...
            FirstName: {user_info["FirstName"]}, 
            LastName: {user_info['LastName']}, 
            Age: {user_info['Age']}, 
            Race: {user_info['Ethnicity']}, 
            Gender: {user_info['Gender']})''')

            scenario = 2

        ID_query = '''SELECT UserID FROM Users WHERE FirstName = %s AND LastName = %s AND Age = %s AND Gender = %s AND Ethnicity = %s'''
        cursor.execute(ID_query, (first_name, last_name, age, gender, ethnicity))
        userID = cursor.fetchall()[0]

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    cursor.close()
    DB_Disconnect(connection)
    print('done')
    return scenario, userID
