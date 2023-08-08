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



-------------------------------------------------------------------------------------------------------------

def load_user_data_toSQL(XL_file):
    table = pd.DataFrame(pd.read_excel(XL_file))
    table = table.where(pd.notna(table), None)
    connection = DB_Connect()
    cursor =connection.cursor()
    try:
        truncate_query = f"TRUNCATE TABLE Users"
        cursor.execute(truncate_query)

        # Reset the auto-increment value to 1
        # reset_query = f"ALTER TABLE UserDetails AUTO_INCREMENT = 00001"
        # cursor.execute(reset_query)

        insert_query = """
            INSERT INTO Users (UserID, FirstName, LastName, Age, Ethnicity, Gender)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        for index, row in table.iterrows():

            UserID = 'User_'+"{:03d}".format(index+1)
            data = (UserID, row['FirstName'], row['LastName'], int(row['Age']), row['Ethnicity'], row['Gender'])
            print(data)
            cursor.execute(insert_query,data)
        connection.commit()
        print('\nData Loaded!')
    except mysql.connector.Error as err:
        print(f"Error: {err}")

    cursor.close()

    DB_Disconnect(connection)



load_user_data_toSQL(r'M:\Private\Softwares\Python Envs\HyperSpace\ComputerVision\Data\Data_Load_to_sqlDB\UserDetails.xlsx')






