from ssl import SSL
import pickle
import os
import rsa
import base64


def __get_info():
    user_list = os.listdir('./server_info')
    user_list = [user_path.replace('.txt', '') for user_path in user_list]

    return user_list


def __save(username, password):
    user_ssl = SSL()
    prikey = user_ssl.prikey
    data = user_ssl.base_encode(password)
    if os.path.exists(f'./client_info/{username}.pkl'):
        os.remove(f'./client_info/{username}.pkl')
    if os.path.exists(f'./server_info/{username}.txt'):
        os.remove(f'./server_info/{username}.txt')
    # 将私钥保存到用户文件中
    with open(f'./client_info/{username}.pkl', 'wb') as f:
        pickle.dump(prikey, f)

    # 将密文和用户名保存在服务端文件中
    with open(f'./server_info/{username}.txt', 'wb') as f:
        pickle.dump(data, f)

    return prikey


def __retrieve(username):
    # 在服务端中找到该用户对应的ssl文件
    with open(f'./server_info/{username}.txt', 'rb') as f:
        data = pickle.load(f)

    # 找到该用户对应的私钥文件
    with open(f'./client_info/{username}.pkl', 'rb') as f:
        prikey = pickle.load(f)

    return data, prikey


def validate_pwd(username, password):
    data, prikey = __retrieve(username)
    # 使用私钥解码
    de_password = rsa.decrypt(base64.b64decode(data), prikey)
    if de_password == password.encode('utf-8'):
        return True
    return False


def register(*args, **kwargs):
    username, password = args[0], args[1]
    user_list = __get_info()
    if username in user_list:
        return False, '该用户已存在，请重新输入'
    # 生成公钥和私钥对密码进行加密
    __save(username, password)

    return True, '注册成功'


def login(*args, **kwargs):
    username = args[0]
    password = args[1]
    user_list = __get_info()
    if username not in user_list:
        return False, '该用户不存在，请先注册'

    flag = validate_pwd(username, password)
    if flag:
        return True, '登录成功'
    return False, '密码错误或秘钥被篡改'


def modify_password(*args, **kwargs):
    # 已经登录的情况下修改密码
    username, new_password = args[0], args[1]
    print(new_password, username)
    __save(username, new_password)
    return True


menu_list = {
    '1、注册': register,
    '2、登录': login,
    '3、修改密码': modify_password
}
