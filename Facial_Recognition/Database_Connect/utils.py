import mysql.connector

host = 'x.x.x.x'
user = 'computervision'
password = 'xxxxxxxx'
database = 'user_db_computer_vision'

def DB_Connect():
    try:
        db_connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            ssl_disabled = True
        )
        if db_connection.is_connected():
            print("\nConnected to the MySQL database!\n")
            return db_connection

    except mysql.connector.Error as error:
        print("Error connecting to the database: {}".format(error))

def DB_Disconnect(connection):
    connection.close()
    print("\nConnection closed.")

