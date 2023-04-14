import pymysql
db = pymysql.connect(
    host='localhost',
    user='root',
    database='business',
    password='123456',
    port=3306,
    charset='utf8mb4'
)
cursor = db.cursor()
# #sql 语句
# sql = 'CREATE TABLE students (id VARCHAR(255) PRIMARY KEY, name VARCHAR(255) NOT NULL, age INT NOT NULL, grade INT)'
#
# # 通过游标执行 sql 语句
# cursor.execute(sql)
#
# #关闭数据库连接
# db.close()

# #插入语句
# sql = 'INSERT INTO students(id, name, age, grade) values ("%(id)s", "%(name)s", %(age)d, %(grade)d)'
#
# try:
#     #执行语句
#     cursor.execute(sql % {'id': '1001', 'name': '张三', 'age': 25, 'grade': 92})
#     #提交
#     db.commit()
# except:
#     #插入异常则回滚数据
#     print('插入异常')
#     db.rollback()
#
# db.close()

# 查询年龄大于20岁的记录
sql = """
SELECT * FROM students
"""

cursor.execute(sql)
result = cursor.fetchall()
print(result)
