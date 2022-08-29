import socket

phone = socket.socket()
phone.connect(('127.0.0.1', 8080))


def register():
    username = input('输入用户名：').strip()
    phone.send(username.encode('utf-8'))
    phone.recv(1024)
    password = input('输入密码：').strip()
    phone.send(password.encode('utf-8'))
    status = phone.recv(1024).decode('utf-8')
    print(status)


def login():
    username = input('输入用户名：').strip()
    phone.send(username.encode('utf-8'))
    phone.recv(1024)
    password = input('输入密码：').strip()
    phone.send(password.encode('utf-8'))
    msg = phone.recv(1024)

    print(msg.decode('utf-8'))


def modify_pwd():
    # 先校验登录状态
    is_login = phone.recv(1024).decode('utf-8')
    if is_login == 'false':
        print('你还没有登录，请先登录')
        return False
    # 校验当前密码
    password = input('输入当前密码：').strip()
    phone.send(password.encode('utf-8'))
    validate = phone.recv(1024).decode('utf-8')
    if validate == 'false':
        print('密码输入错误，请重新输入')
        return False

    # 输入新密码，并修改
    new_password = input('输入新密码：').strip()
    phone.send(new_password.encode('utf-8'))
    msg = phone.recv(1024).decode('utf-8')
    print(msg)

    return True


while True:
    recv = phone.recv(1024)
    print(recv.decode('utf-8'))  # 打印菜单
    msg = input('输入要选择的功能：').strip()
    if msg == '0':
        break
    # 向服务端发送请求，并接收校验后的响应
    phone.send(msg.encode('utf-8'))
    recv = phone.recv(1024).decode('utf-8')
    if recv == '输入有误，请重新输入':
        print(recv)
        continue
    if int(msg) == 1:
        register()
    elif int(msg) == 2:
        login()
    elif int(msg) == 3:
        status = modify_pwd()
        if not status:
            continue
phone.close()
