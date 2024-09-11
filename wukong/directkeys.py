

import ctypes
import time

SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
L = 0x26
M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
R = 0x13#用R代替识破
V = 0x2F
T = 0x14

Q = 0x10
I = 0x17
O = 0x18
P = 0x19
C = 0x2E
F = 0x21

up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

esc = 0x01

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    

def attack():
    PressKey(M)
    time.sleep(0.01)
    ReleaseKey(M)


def attack2():
    PressKey(M)
    time.sleep(0.01)
    ReleaseKey(M)





def attack3():
    PressKey(J)
    time.sleep(0.01)
    ReleaseKey(J)


def go_forward():
    PressKey(W)
    time.sleep(0.01)
    ReleaseKey(W)
    
def go_back():
    PressKey(S)
    time.sleep(0.8)
    ReleaseKey(S)
    
def go_left():
    PressKey(A)
    time.sleep(2)
    ReleaseKey(A)
    
def go_right():
    PressKey(D)
    time.sleep(2)
    ReleaseKey(D)
    
def jump():
    PressKey(K)
    time.sleep(0.01)
    ReleaseKey(K)

    
def dodge2():#闪避
    PressKey(L)
    time.sleep(0.01)
    ReleaseKey(L)
    time.sleep(0.01)
    PressKey(L)
    time.sleep(0.01)
    ReleaseKey(L)


def dodge1():#闪避
    PressKey(S)
    time.sleep(0.01)
    PressKey(L)
    time.sleep(0.01)
    ReleaseKey(L)
    time.sleep(0.01)
    ReleaseKey(S)



    
def lock_vision():
    PressKey(P)
    time.sleep(0.01)
    ReleaseKey(P)

    
def go_forward_QL(t):
    PressKey(W)
    time.sleep(t)
    ReleaseKey(W)
    
def turn_left(t):
    PressKey(left)
    time.sleep(t)
    ReleaseKey(left)
    
def turn_up():
    PressKey(T)
    time.sleep(0.01)
    ReleaseKey(T)
    
def turn_right(t):
    PressKey(right)
    time.sleep(t)
    ReleaseKey(right)
    
def F_go():
    PressKey(F)
    time.sleep(0.5)
    ReleaseKey(F)
    
def forward_jump(t):
    PressKey(W)
    time.sleep(t)
    PressKey(K)
    ReleaseKey(W)
    ReleaseKey(K)
    
def drink():

    PressKey(R)
    time.sleep(0.01)
    ReleaseKey(R)
    
def dead():
    PressKey(J)
    time.sleep(0.5)
    ReleaseKey(J)

