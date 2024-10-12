import uuid

def generate_unique_value():
    return str(uuid.uuid4())[:8]

# 示例调用
unique_value = generate_unique_value()
print(unique_value)
