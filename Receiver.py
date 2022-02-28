import sys
# from apc_log import APC_Log
from socket import *
import json 
import cv2
import os
import time
import shutil
import datetime
import numpy as np
from Segmentation import Detectron_inference

detectron = Detectron_inference()

class RECEIVER():
    def __init__(self):
        self.HOST = 'Localhost'
        self.PORT = 5555
        
    def Connect(self):        
        self.Receiver = socket(AF_INET, SOCK_STREAM)
        self.Receiver.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.Receiver.bind((self.HOST, self.PORT))
        print('Receiver Socket binding {}'.format(self.Receiver))
        self.Receiver.listen(5)
        print('Receiver Listening')
    
    def Get_Tcp_mode(self):
        self.conn, self.addr = self.Receiver.accept()
        self.conn.settimeout(3)
        print('Connection accepted {}'.format(self.addr))
        self.mode_check = self.conn.recv(1024).decode()
        if self.mode_check == 'connect':
            self.conn.sendall('connected'.encode())        
        if self.mode_check == 'processing':
            self.conn.sendall('Working'.encode())        
        if self.mode_check == 'Health_check':
            self.conn.sendall('Alive'.encode())
        if self.mode_check == 'Make Idle':
            detectron.empty_gpu()
            self.conn.sendall('Idling'.encode())
            
    def Get_File_Info(self):        
        try:
            self.file_info = json.loads(self.conn.recv(4096).decode())    
            print('Json dumping done')
        except:
            print('Json dumping failed')
            self.Receiver.close()
            self.Connect()
            pass

    def Get_File(self):
        data_transferred = 0
        data = self.file_info['file_length'] 

        self.conn.sendall('READY'.encode())
        print('Receiver ready to get a image')
        try:
            with open("./tmp/"+self.file_info['file_name'], 'wb') as f: 
                while True: 
                    data = self.conn.recv(1024)  
                    f.write(data) 
                    data_transferred += len(data)
                    if not data:
                        break
        except:
            print('Receiver got a image successfully')     
            obj_list, area_list, area_per_list = detectron.Main("./tmp/"+self.file_info['file_name'])
            print('detectron2 processing done!!!')
            self.conn.sendall('READY'.encode())
            
            return obj_list, area_list, area_per_list
            pass

    def Dump_Json(self, obj_list, area_list, area_per_list, base_path, img_name):
        file_info = {
                    'file_name' : img_name,
                    'file_length' : self.GetFileSize(base_path),
                    'file_reverse_name' : 'test2.jpg',
                    'file_reverse_length' : self.GetFileSize('./test2.jpg'),
                    'create_time' : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'obj_list' : obj_list, 
                    'area_list' : area_list, 
                    'area_per_list' : area_per_list
                    }

        send_data = json.dumps(file_info)
        print(file_info)
        self.conn.send(send_data.encode())
        time.sleep(0.01)
            
    def GetFileSize(self, base_path):        
        filesize = os.path.getsize(base_path)
        return str(filesize)

    def GetFileData(self, base_path):
        data_transferred = 0
        self.conn.settimeout(1)
        with open(base_path, 'rb') as f:
            data = f.read(1024)
            while data:
                data_transferred += self.conn.send(data)
                data = f.read(1024)
        # os.remove(self.base_path)
        print('image returned successfully')
        time.sleep(0.01)
        
        
class WB_Process():
    def __init__(self, HOST, PORT):
        self.HOST = HOST
        self.PORT = int(PORT)
        self.connect()

    def connect(self):
        self.receiver = RECEIVER()
        self.receiver.HOST = self.HOST
        self.receiver.PORT = self.PORT        
        self.receiver.Connect()
        while True:            
            self.receiver.Get_Tcp_mode()
            print(self.receiver.mode_check)
            if self.receiver.mode_check == 'processing':
                self.receiver.Get_File_Info() #get image infomation
                obj_list, area_list, area_per_list = self.receiver.Get_File() #get image file
                self.receiver.Dump_Json(obj_list, area_list, area_per_list, base_path = './test.jpg', img_name = 'test.jpg') #return result info
                status = self.receiver.conn.recv(1024) #check if ready to return result image
                if status.decode() == 'READY':         
                    self.receiver.GetFileData(base_path = './test.jpg') #return result image
                    self.receiver.conn.settimeout(3)
                    print(self.receiver.conn.recv(1024).decode())
                    self.receiver.GetFileData(base_path = './test2.jpg')
            else:
                pass
        self.receiver.conn.close()


        
if __name__ == '__main__':
    wb = WB_Process(HOST = '172.23.69.100', PORT = '5555')

