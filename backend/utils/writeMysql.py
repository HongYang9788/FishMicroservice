from mysql.connector import connect, Error
from getpass import getpass

species = [
    {'ids': 1,  'sciName': 'Thunnus alalonga', 'commonName': "Albacore", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=382500"},
    {'ids': 2,  'sciName': "Thunnus obesus", 'commonName': "Bigeye tuna", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=382502"},
    {'ids': 3,  'sciName': "Makaira nigricans", 'commonName': "Blue marlin", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=382689"},
    {'ids': 4,  'sciName': "Coryphaena hippurus", 'commonName': "Common dolphinfish", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=381638"},
    {'ids': 5,  'sciName': "Lepidocybium flavobrunneum", 'commonName': "Escolar", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=381672"},
    {'ids': 6,  'sciName': "Kajikia audax", 'commonName': "Red Marlin", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=382691"},
    {'ids': 7,  'sciName': "Ruvettus pretiosus", 'commonName': "Rough skin oilfish", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=381676"},
    {'ids': 8,  'sciName': "Thunnus maccoyii", 'commonName': "Southern bluefin tuna", 'link': "https://zh.wikipedia.org/wiki/%E8%93%9D%E9%B3%8D%E9%87%91%E6%9E%AA%E9%B1%BC"},
    {'ids': 9,  'sciName': "Istiophorus platypterus", 'commonName': "Indo-pacific sailfish", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=382687"},
    {'ids': 10, 'sciName': "Katsuwonus pelamis", 'commonName': "Skipjack tuna", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=382489"},
    {'ids': 11, 'sciName': "Tetrapturus angustirostris", 'commonName': "Shortbill spearfish", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=382690"},
    {'ids': 12, 'sciName': "Xiphias gladius", 'commonName': "Swordfish", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=382692"},
    {'ids': 13, 'sciName': "None", 'commonName': "Shark", 'link': "https://zh.wikipedia.org/wiki/%E9%B2%A8%E9%B1%BC"},
    {'ids': 14, 'sciName': "Acanthocybium solandri", 'commonName': "Wahoo", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=382484"},
    {'ids': 15, 'sciName': "Thunnus albacares", 'commonName': "Yellowfin tuna", 'link': "https://fishdb.sinica.edu.tw/chi/species.php?id=382501"}]

if __name__ == '__main__':
    try:
        connection = connect(
            host="10.233.106.105",
            user=input("Enter username: "),
            password=getpass("Enter password: "),
            database="FishSpecies"
        )
        if connection.is_connected():
            print('Successful')
        
        with connection.cursor() as cursor:
            '''
            for data in species:
                print(data.values())
                create_db_query = "INSERT INTO FishMetaData (Id, SciName, CommonName, Link) VALUES (%s, %s, %s, %s);"
                cursor.execute(create_db_query, list(data.values()))

            '''
            db_query = "SELECT * FROM FishMetaData;"
            cursor.execute(db_query)
            sql_result = cursor.fetchall()
            print(sql_result)
            connection.commit()
                
    except Error as e:
        print(e)
    finally:
        connection.close()