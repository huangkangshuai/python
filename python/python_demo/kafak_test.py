from kafka import KafkaProducer
import random
import json

# 实例化一个KafkaProducer示例，用于向Kafka投递消息
producer = KafkaProducer(
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    bootstrap_servers=['10.7.2.20:9092']
)
#读取样例数据
files = open("样例数据文件的目录", "r", encoding='UTF-8')
content = files.readlines()

while True :
    index = random.randint(0, len(content) - 1)
    producer.send("test", content[index])

producer.close()
