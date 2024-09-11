import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
from ultralytics import YOLO
import directkeys
import time
import keyboard 



def pause_game(paused):
    if keyboard.is_pressed('t'):  # 检测 'T' 键是否被按下
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    
    if paused:
        print('paused')
        while True:
            if keyboard.is_pressed('t'):  # 检测 'T' 键是否被按下以恢复
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    
    return paused

def ability_count2(ability_color2):
    # 将图像从BGR转换为灰度图像
    gray_image = cv2.cvtColor(ability_color2, cv2.COLOR_BGR2GRAY)
    
    # 定义灰度阈值，低于此阈值的像素被视为“深灰色”
    threshold = 250
    
    # 统计灰度图像中灰度值高于阈值的像素数量
    ability_blood2 = np.sum(gray_image > threshold)
    
    return ability_blood2

def detect_color_and_attack(blood2_in_range_count, current_cls_id):
    ability_window2 = (1205, 705, 1215, 720)

    # 排除 0, 1, 2, 3, 5 类别，只有当 current_cls_id 不在这些类别时才允许执行 attack3
    print(f"Checking if attack3 can be triggered for class ID: {current_cls_id}")
    if current_cls_id not in [6, 7, 8]:
        print(f"Current class ID {current_cls_id} in [0, 1, 2, 3, 5], skipping attack3.")
        return blood2_in_range_count  # 如果是这些状态则跳过颜色检测

    # 检测特定区域颜色
    screen_color3 = grab_screen(ability_window2)
    self_blood2 = ability_count2(screen_color3)

    # Debugging 输出，显示当前 self_blood2 的值
    print(f"Current self_blood2 value: {self_blood2}")

    # 检测 self_blood2 是否在指定范围内
    if 10 <= self_blood2 <= 40:
        blood2_in_range_count += 1
        #print(f"self_blood2 in range. Count: {blood2_in_range_count}")
    else:
        blood2_in_range_count = 0  # 如果不在范围内则重置计数器
        #print("self_blood2 out of range, resetting count.")

    # 如果 self_blood2 在范围内持续 3 帧，则执行 attack3
    if blood2_in_range_count >= 2:
        #print("Conditions met for attack3. Attempting to execute attack3.")
        directkeys.attack3()
        #print("Successfully executed attack3 due to self_blood2 staying in range.")
        blood2_in_range_count = 0  # 执行后重置计数器
    else:
        print("Conditions for attack3 not met.")

    return blood2_in_range_count

def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')  
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


# Load the YOLOv8 model
model = YOLO(r"D:\AI\yolo8\training_output\wukong2\weights\best.pt")
screen_region = (26, 50, 1200, 750)

def get_box_area(box):
    # YOLO box contains (x1, y1, x2l, y2)
    x1, y1, x2, y2 = box.xyxy[0]
    width = x2 - x1
    height = y2 - y1
    return width * height, width, height


paused = False
blood2_in_range_count = 0


while True:
    start_time = time.time()
    paused = pause_game(paused)
    if paused:
        continue  # 如果暂停，跳过后续操作    
        # 检测玩家的血量


    
    action_performed = False

    screen_image = grab_screen(screen_region)
    
    # Convert BGRA to BGR
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2BGR)
    results = model(screen_image, iou=0.5, conf=0.4)
    current_cls_id = None  # 初始化当前类别IDr
    if len(results[0].boxes) == 0:
        # 如果没有检测到任何结果，执行 dodge2

        #directkeys.dodge2()
        pass
        action_performed = True

    else:
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[cls_id] if hasattr(model, "names") else str(cls_id)

            if conf >= 0.4:
                #print(f"Detected: {class_name}, Confidence: {conf:.2f}")
                current_cls_id = cls_id  # 获取当前类别ID            
                if not action_performed:
                    if cls_id in [6, 7, 8]:  # boss_cry, boss_down, boss_stand
                        directkeys.attack()
                        action = "attack"
                    elif cls_id in [1, 4]:  # Other defined actions
                        directkeys.dodge1()
                        action = "dodge1"
                    elif cls_id in [0, 2, 3, 5]:  # Other defined actions
                        directkeys.dodge2()
                        action = "dodge2"                        
                    else:  # All other undefined actions
                        directkeys.dodge2()
                        action = "dodge2"

                    action_performed = True  # 标记本帧已经执行动作


    # 检测颜色并触发 attack3
    if current_cls_id is not None:
        #print(f"Passing current class ID: {current_cls_id} to detect_color_and_attack function.")
        blood2_in_range_count = detect_color_and_attack(blood2_in_range_count, current_cls_id)

    # 显示带注释的图像
    annotated_image = results[0].plot()

    cv2.imshow("YOLOv8 Inference - Screen Capture", annotated_image)


    # 按 'q' 键退出循环
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()