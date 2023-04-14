import pymysql
from flask import Flask, request
import json
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
CORS(app=app)
app.debug = True
db = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='python')
cursor = db.cursor()  # 创建游标


@app.route('/')
def index():
    return 'Index Page'


@app.route('/add', methods=['post'])
def add():
    req_data = request.get_data()
    data = json.loads(req_data)
    print(data)
    try:
        sql_data = (data['name'], int(data['age']), data['major'], data['hobby'])
        sql = "insert into userinfo(name,age,major,hobby) values (%s,%s,%s,%s)"
        cursor.execute(sql, sql_data)
        db.commit()
        return {'code': 200, 'msg': '数据新增成功'}
    except:
        db.rollback()  # 发生错误时回滚
        return {'code': 1000, 'msg': "添加失败"}


@app.route('/del', methods=['delete'])
def delete():
    deleteName = request.args.get('name')
    sql = f'delete from `userinfo` where name="{deleteName}";'
    try:
        cursor.execute(sql)
        db.commit()
        return {'code': 200, 'msg': '删除成功'}
    except:
        db.rollback()  # 发生错误时回滚
        return {'code': 1000, 'msg': "删除失败"}


@app.route('/edit', methods=['put'])
def edit():
    req_data = request.get_data()
    data = json.loads(req_data)
    print('修改消息：', data)
    try:
        sql = f"update userinfo set name='{data['afterName']}' where name='{data['beforeName']}'"
        cursor.execute(sql)
        db.commit()
        return {'code': 200, 'msg': '修改成功'}
    except:
        db.rollback()
        return {'code': 1000, 'msg': "修改失败"}


@app.route('/select', methods=['get'])
def select():
    try:
        cursor.execute("SELECT * FROM userinfo")
        array = []
        data = ()
        while isinstance(data, tuple):  # 循环遍历出data的数据
            data = cursor.fetchone()  # fetchone方法用于取出数据库中查询的单个数据
            if (data == None): break
            obj = {}
            obj['id'] = data[0]
            obj['name'] = data[1]
            obj['age'] = data[2]
            obj['major'] = data[3]
            obj['hobby'] = data[4]
            array.append(obj)
        return {'code': 200, 'msg': '查询成功', 'data': array}
    except:
        db.rollback()
        return {'code': 1000, 'msg': "查询失败"}


if __name__ == '__main__':
    http_server = WSGIServer(('127.0.0.1', 3389), app)
    http_server.serve_forever()
    # app.run(host="localhost", port=3389, debug=True)

    cursor.close()
    db.close()
