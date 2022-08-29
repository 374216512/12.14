import socketserver
from user import menu_list
from user import validate_pwd


class RequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print(self.client_address)
        self.is_login = False  # 记录当前用户是否登录
        while True:  # 通信循环
            try:
                # 先发送服务
                recv = self.send_menu()
                if len(recv) == 0:
                    break
                # 判断用户响应是否正确
                flag = self.validate_input(recv)
                if not flag:  # 非法输入
                    continue
                if self.choose == 0:  # 退出
                    break

                # 响应正确，开始享受系统美好的服务
                if self.choose == 1:
                    flag, msg = self.register()
                    self.request.send(msg.encode('utf-8'))
                elif self.choose == 2:
                    flag, msg = self.login()
                    self.request.send(msg.encode('utf-8'))
                elif self.choose == 3:
                    self.modify_pwd()
            except Exception:
                break
        self.request.close()

    def show_menu(self):
        msg = ''
        for key in menu_list:
            msg += key
            msg += '\n'
        msg += '0、退出\n'

        return msg

    def send_menu(self):
        msg = self.show_menu()
        self.request.send(msg.encode('utf-8'))
        # 获取用户响应
        recv = self.request.recv(1024)
        return recv

    def validate_input(self, recv):
        try:
            choose = int(recv.decode('utf-8'))
        except ValueError:
            self.request.send('输入有误，请重新输入'.encode('utf-8'))
            return False
        if choose not in range(len(menu_list) + 1):
            self.request.send('输入有误，请重新输入'.encode('utf-8'))
            return False
        self.choose = choose
        self.request.send(' '.encode('utf-8'))
        return True

    def register(self):
        func = list(menu_list.values())[self.choose - 1]
        username = self.request.recv(1024).decode('utf-8')
        self.request.send('输入密码：'.encode('utf-8'))
        password = self.request.recv(1024).decode('utf-8')
        flag, msg = func(username, password)

        return flag, msg

    def login(self):
        username = self.request.recv(1024).decode('utf-8')
        self.username = username
        self.request.send('收到用户名了'.encode('utf-8'))
        password = self.request.recv(1024).decode('utf-8')
        func = list(menu_list.values())[self.choose - 1]
        flag, msg = func(username, password)
        if flag:
            self.is_login = True

        return flag, msg

    def modify_pwd(self):
        # 查看登录状态
        if not self.is_login:
            self.request.send('false'.encode('utf-8'))
            return False, 'false'
        self.request.send('true'.encode('utf-8'))

        # 校验当前密码
        password = self.request.recv(1024).decode('utf-8')
        flag = validate_pwd(self.username, password)
        if not flag:
            self.request.send('false'.encode('utf-8'))
            return False, 'false'
        self.request.send('true'.encode('utf-8'))

        # 修改密码
        new_password = self.request.recv(1024).decode('utf-8')
        func = list(menu_list.values())[self.choose - 1]
        func(self.username, new_password)
        self.request.send('密码修改成功'.encode('utf-8'))

        return True, 'true'


server = socketserver.ThreadingTCPServer(('127.0.0.1', 8080),
                                         RequestHandler,
                                         bind_and_activate=True)
# 链接循环

server.serve_forever()
