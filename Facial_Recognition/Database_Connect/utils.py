import mysql.connector

host = '176.9.99.185'
user = 'computervision'
password = 'Computervision@253#87'
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

