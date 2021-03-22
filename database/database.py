import base64
import mysql.connector

def connection():
    try:
        con = mysql.connector.connect(
                          host="localhost",
                          user="root",
                          password="clevervision",
                          database="jewel"
                        )
        cur = con.cursor()

        cur.execute("CREATE TABLE Query(Id VARCHAR(255) PRIMARY KEY, Query_img VARCHAR(255), Total_result BIGINT(255),Total_images BIGINT(255))")
        print("Tabels CREATED")
    except Exception as e:
        # print ("Error %s:" % e)
        pass
    return con

def convert2binary(img_path):
    with open(img_path, "rb") as f:
        im_b64_str = base64.b64encode(f.read())
        img = "data:image/png;base64," + im_b64_str.decode("utf-8")
    return img

def db_insert_query(Query_img,total_result,total_images):
    conn = connection()
    cur_q = conn.cursor()
    try:
        cur_q.execute("CREATE TABLE Query(Id VARCHAR(255) PRIMARY KEY, Query_img VARCHAR(255), Total_result BIGINT(255),Total_images BIGINT(255))")
    except:
        pass
    finally:
        ID = "id_" + Query_img
        Q_query = "INSERT INTO Query VALUES('{}','{}','{}','{}')".format(ID, Query_img, total_result, total_images)
        cur_q.execute(Q_query)
        print("Q_Inserted_Successfully")
        conn.commit()
        conn.close()

def db_insert_result(Query_img,Result_img):
    conn = connection()
    cur_r = conn.cursor()
    try:
        cur_r.execute("CREATE TABLE Result(Id VARCHAR(255), num_image BIGINT(255), Result_img VARCHAR(255), similariries DOUBLE(50,10), img_path VARCHAR(255))")
    except:
        pass
    finally:
        ID = "id_" + Query_img
        for num,(sim, rimg_path) in enumerate(Result_img):
            R_query = "INSERT INTO Result VALUES('{}','{}','{}','{}','{}')".format(ID, num+1, rimg_path.split('/')[-1], sim, rimg_path)
            cur_r.execute(R_query)
        print("R_Inserted_Successfully")   
        conn.commit()
        conn.close() 

def db_delete(q_id, r_name):
    conn = connection()
    cur = conn.cursor()
    D_query = "DELETE FROM Result WHERE Id = 'id_{}' and Result_img = '{}'".format(q_id, r_name)
    U_in_Query_T = "SELECT * FROM Query WHERE id = 'id_{}'".format(q_id)
    cur.execute(D_query)
    cur.execute(U_in_Query_T)
    rows = cur.fetchall()
    update_total_Result = rows[0][2] - 1
    U_query = "UPDATE Query SET Total_result = '{}' WHERE Id = 'id_{}'".format(update_total_Result, q_id)
    cur.execute(U_query)
    conn.commit()
    conn.close()
    print("Deleted_Successfully")

def db_update(q_id, r_name, sim):
    conn = connection()
    cur = conn.cursor()
    U_query = "UPDATE Result SET similariries='{}' WHERE Id = 'id_{}' and Result_img = '{}'".format(sim, q_id, r_name)
    cur.execute(U_query)
    conn.commit()
    conn.close()
    print("Updated_Successfully")

def fet_img(q_name,page):
    conn = connection()
    cur = conn.cursor()
    F_query = "SELECT * FROM Result WHERE id = 'id_{}'".format(q_name)
    cur.execute(F_query)
    rows = cur.fetchall()
    final_=[]
    end = int(page)*40
    for data in rows[end - 40:end]:
        # image = convert2binary(data[4])
        final_.append((data[2],data[3],data[1]))  # data[2] is img_name data[3] is sim data[1] is 1,2,3 and image is img in bytes format
    final_.sort(key = lambda x: x[1], reverse = True)
    # final_=[]
    # for i,data in enumerate(temp_):
    #     final_.append((data[0],data[1],data[2],i+1))
    print("Fetched_for_Pages")
    return final_

def fetch_data(q_name):
    conn = connection()
    cur = conn.cursor()
    F_query = "SELECT * FROM Result WHERE id = 'id_{}'".format(q_name)
    cur.execute(F_query)
    rows = cur.fetchall()
    final_=[]
    for data in rows:
        final_.append((data[2],data[3]))  # data[2] is img_name data[3] is sim
    final_.sort(key = lambda x: x[1], reverse = True)
    final_numbered=[]
    for i,data in enumerate(final_):
        final_numbered.append((i+1,data[0],data[1]))
    print(final_numbered)
    print("Fetched_Successfully")
    return final_numbered,len(rows)

def check_img(q_name):
    conn = connection()
    cur = conn.cursor()
    F_query = "SELECT Query_img FROM Query"
    cur.execute(F_query)
    rows = cur.fetchall()
    q_ids = [data[0] for data in rows]
    if q_name in q_ids:
        cur.execute("SELECT Total_result,Total_images from Query WHERE id='id_{}'".format(q_name))
        rows = cur.fetchall()
        return True,rows[0]
    else:
        return False,[0,0]
