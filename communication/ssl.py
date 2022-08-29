import rsa, base64
from OpenSSL.crypto import PKey
from OpenSSL.crypto import TYPE_RSA, FILETYPE_PEM, FILETYPE_ASN1
from OpenSSL.crypto import dump_privatekey, dump_publickey


class SSL:
    def __init__(self):
        self.generate_ssl()

    def generate_ssl(self):
        self.pk = PKey()
        self.pk.generate_key(TYPE_RSA, 512)  # 生成非对称加密的密钥对
        self.public_key = dump_publickey(FILETYPE_PEM, self.pk)  # 生成公钥
        self.primary_key = dump_privatekey(FILETYPE_ASN1, self.pk)  # 生成私钥

        self.pubkey = rsa.PublicKey.load_pkcs1_openssl_pem(self.public_key)  # 生成公钥对象
        self.prikey = rsa.PrivateKey.load_pkcs1(self.primary_key, 'DER')  # 生成私钥对象

    def base_encode(self, password):
        self.data = rsa.encrypt(password.encode('utf-8'), self.pubkey)  # 用公钥加密
        self.data = base64.b64encode(self.data)  # base64转码

        return self.data

    def base_decode(self, password, prikey):
        data = rsa.decrypt(base64.b64decode(self.data), prikey)  # 用私钥解密，base64解码
        print(type(prikey))
        if data == password.encode('utf-8'):
            return True
        return False
