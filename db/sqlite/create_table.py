import sqlite3

# 连接到 SQLite 数据库
conn = sqlite3.connect('bilibili.db')
cursor = conn.cursor()

# 创建 dynamic 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS dynamic (
    id TEXT PRIMARY KEY,
    audio BLOB,
    content TEXT,
    summary TEXT,
    is_sent INTEGER DEFAULT 0
)
''')

# 插入数据
# id = 'BV1xx411c7mD1'
# audio_data = b'\x00\x01\x02...'
# content = '这是一个示例文本'
# summary = '这是一个示例摘要'
#
# cursor.execute('INSERT INTO dynamic (id, audio, content, summary) VALUES (?, ?, ?, ?)', (id, audio_data, content, summary))
#
# # 提交事务
# conn.commit()
#
# # 关闭连接
# conn.close()