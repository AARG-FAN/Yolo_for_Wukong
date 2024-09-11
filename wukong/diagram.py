import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.font_manager import FontProperties

# Initialize variables to store the points
ref_point = []
cropping = False


# Mouse callback function to record clicks
def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]  # Record the first point (top-left)
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))  # Record the second point (bottom-right)
        cropping = False

        # Draw a rectangle around the selected region
        cv2.rectangle(param, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Select Region", param)

def select_blood_region(prompt, screen_image):
    global ref_point
    ref_point = []  # Reset the reference points before every new selection

    # Display the screen image for selection
    clone = screen_image.copy()
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", click_and_crop, clone)

    print(prompt)
    while True:
        cv2.imshow("Select Region", clone)
        key = cv2.waitKey(1) & 0xFF

        # Break the loop after the user has selected the region
        if key == ord("q") or len(ref_point) == 2:
            break

    cv2.destroyWindow("Select Region")

    # Return the selected region (left, top, right, bottom)
    return (ref_point[0][0], ref_point[0][1], ref_point[1][0], ref_point[1][1])

# Screen capture function
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


# Initialize YOLO model
model = YOLO(r"D:\AI\yolo8\training_output\wukong2\weights\best.pt")
screen_region = (0, 0, 1300, 750)

# Data for plotting
time_data = []
boss_health_data = []
self_health_data = []
player_action_history = []  # Store history of player actions (for each time step)
boss_action_history = []  # Store history of boss actions (for each time step)
action_start_times = []
action_labels = {
    0: "boss_jump_attack", 1: "boss_feet_attack", 2: "boss_clap_attack", 3: "boss_far_attack",
    4: "boss_boom_attack", 5: "boss_sweep_attack", 6: "boss_cry", 7: "boss_down", 8: "boss_stand",
    9: "self_run", 10: "self_stand", 11: "self_light_attack", 12: "self_pigun",
    13: "self_ligun", 14: "self_chuogun", 15: "self_down", 16: "self_dodge",
    99: "no_action_detected"  # 添加表示没有检测到动作的标签
}
# Action colors for player and boss with bright tones to contrast black background
action_colors = {
    0: '#56B4E9', 1: '#009E73', 2: '#F0E442', 3: '#E69F00', 4: '#D55E00', 5: '#CC79A7',
    6: '#0072B2', 7: '#999999', 8: '#ADFF2F', 9: '#FF6347', 10: '#FFD700', 11: '#1E90FF',
    12: '#32CD32', 13: '#FF4500', 14: '#DAA520', 15: '#00CED1', 16: '#FF1493',
    99: '#000000'  # 为表示未检测到动作的99提供默认黑色
}
# Initialize plot with 2 subplots: one for main plot and one for bars below
plt.ion()  # Interactive mode on
fig, (ax1, ax_bar) = plt.subplots(2, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 6))

# Adjust margins to bring everything closer and fit on the screen
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

# Set the background color to black for both plots
fig.patch.set_facecolor('black')
ax1.set_facecolor('black')
ax_bar.set_facecolor('black')

# Set axis labels and axis ranges for boss health and actions with bright labels for contrast
ax1.set_xlabel("时间 / Time", fontsize=12, fontweight='bold', color='white')
ax1.set_ylabel("血量 / Blood", color='white', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 10)  # Adjust as per boss health range

# Set subplot for the action bars below
ax_bar.set_ylim(0, 3)  # Keep space for player and boss actions
ax_bar.set_xlim(0, 120)  # Assuming max time is 120 seconds for this example
ax_bar.axis('off')  # Hide axis lines and ticks for bar plot

# Set axis colors (for ticks, labels, etc.) to white
ax1.spines['bottom'].set_color('white')
ax1.spines['top'].set_color('white') 
ax1.spines['left'].set_color('white')
ax1.spines['right'].set_color('white')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

# Use a flat, sans-serif font style
plt.rcParams.update({'font.size': 10, 'font.sans-serif': ['Arial'], 'axes.edgecolor': 'white'})

# Enable grid with a subtle color for a grid background effect
ax1.grid(True, color='#444444', linestyle='--', linewidth=0.5)

# Define the font properties for flat design
font_properties = FontProperties(family='Microsoft YaHei', weight='bold', size=12)
font_properties1 = FontProperties(family='Microsoft YaHei', weight='light', size=8)

# Fixed text position for "boss行为" and "玩家行为" using flat design
fig.text(0.02, 0.225, "Boss行为", va='center', ha='left', color='white', fontsize=12, fontproperties=font_properties)
fig.text(0.02, 0.118, "玩家行为", va='center', ha='left', color='white', fontsize=12, fontproperties=font_properties)
fig.text(0.2, 0.87, "血量/Blood", va='center', ha='right', color='white', fontsize=12, fontproperties=font_properties)
fig.text(0.94, 0.35, "时间/Time", va='center', ha='right', color='white', fontsize=12, fontproperties=font_properties)

# Function to update the plot
def update_plot(current_time, boss_health, player_action, boss_action):
    # Append current data
    time_data.append(current_time)
    boss_health_data.append(boss_health)
    player_action_history.append(player_action)
    boss_action_history.append(boss_action)

    # Clear the previous plot
    ax1.clear()
    ax_bar.clear()

    # Reapply the black background after clearing the plots
    ax1.set_facecolor('black')
    ax_bar.set_facecolor('black')
    ax1.grid(True, color='#444444', linestyle='--', linewidth=0.5)  # Reapply grid after clearing

    # Set axis colors again after clearing
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['right'].set_color('white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')

    # Interpolate boss health for smoother curve using CubicSpline, but ignore values below 100
    filtered_boss_health = np.array(boss_health_data, dtype=float)  # Convert to numpy array for easy manipulation
    filtered_time_data = np.array(time_data, dtype=float)  # Time data must match filtered health data

    # Filter out boss health data below 100
    mask = filtered_boss_health >= 10
    filtered_boss_health = filtered_boss_health[mask]
    filtered_time_data = filtered_time_data[mask]

    # Only interpolate if we have more than one valid point
    if len(filtered_time_data) > 1:
        cs_boss = CubicSpline(filtered_time_data, filtered_boss_health, extrapolate=False)
        smooth_time = np.linspace(filtered_time_data[0], filtered_time_data[-1], 300)
        smooth_boss_health = cs_boss(smooth_time)
        ax1.plot(smooth_time, smooth_boss_health, '-', color='#56B4E9', linewidth=2, label='Boss Health')
    else:
        ax1.plot(filtered_time_data, filtered_boss_health, '-', color='#56B4E9', linewidth=2, label='Boss Health')

    # Add critical points as markers
    ax1.scatter(filtered_time_data, filtered_boss_health, color='#F0E442', zorder=5, label="Critical Points (Boss)", s=40)

    # Iterate over action history and plot each segment with corresponding color
    for i in range(len(time_data) - 1):
        # Plot player action history
        ax_bar.barh(1, time_data[i+1] - time_data[i], height=0.25,
                    color=action_colors.get(player_action_history[i], '#000000'), left=time_data[i], alpha=0.8)

        # Plot boss action history
        ax_bar.barh(2, time_data[i+1] - time_data[i], height=0.25,
                    color=action_colors.get(boss_action_history[i], '#000000'), left=time_data[i], alpha=0.8)

    # Add action labels to the current step
    ax_bar.text(current_time, 1, action_labels.get(player_action, "No Action Detected"), va='center', ha='left', color='white', fontsize=10)
    ax_bar.text(current_time, 2, action_labels.get(boss_action, "No Action Detected"), va='center', ha='left', color='white', fontsize=10)

    # Set axis limits dynamically based on time
    ax1.set_xlim(0, current_time + 5)  # Keep expanding x-axis with time
    ax_bar.set_xlim(0, current_time + 5)

    # Redraw the updated plot
    plt.draw()
    plt.pause(0.2)  # Increased pause to reduce update frequency

previous_boss_blood = 0

def boss_blood_count(boss_color):
    global previous_boss_blood  # 声明使用全局变量
    
    gray_image = cv2.cvtColor(boss_color, cv2.COLOR_BGR2GRAY)
    height = gray_image.shape[0]
    middle_row = gray_image[height // 2, :]
    threshold = 150
    
    # 计算大于阈值的像素数量，即血量
    boss_blood = np.sum(middle_row > threshold)
    
    # 如果当前 boss_blood 为 0，则使用上一个保存的血量
    if boss_blood < 10:
        boss_blood = previous_boss_blood
    else:
        previous_boss_blood = boss_blood  # 更新上一个保存的血量
    
    return boss_blood

# Initialize counters for action consistency check
boss_action_count = 0
player_action_count = 0
previous_boss_action = 99  # No action detected initially
previous_player_action = 99  # No action detected initially
action_threshold = 2  # Define threshold for number of consecutive frames

def simulate_game_frame(blood_window2, model, screen_region, action_labels):
    global boss_action_count, player_action_count, previous_boss_action, previous_player_action

    # 获取屏幕图像并转换颜色格式
    screen_image = grab_screen(screen_region)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2BGR)
    
    # 使用 YOLO 模型进行检测
    results = model(screen_image, iou=0.5, conf=0.4)
    
    # 初始化动作和血量
    boss_action = None
    player_action = None
    boss_health = 0


    # Iterate over the detected bounding boxes
    for box in results[0].boxes:
        cls_id = int(box.cls)  # Get the class index
        class_name = action_labels.get(cls_id, "Unknown")  # Get class name based on index
        
        # Detect if it's a Boss action or Player action
        if "boss" in class_name:
            boss_action = cls_id  # Update the boss action
            boss_health = boss_blood_count(grab_screen(blood_window2))  # Immediately update boss health

        elif "self" in class_name:
            player_action = cls_id  # Update the player action
    
    if boss_action is None:
        boss_action = 99  # Special value indicating no action detected
        screen_image[blood_window2[1]:blood_window2[3], blood_window2[0]:blood_window2[2]] = 0
    
    if player_action is None:
        player_action = 99  # Special value indicating no action detected
    
    return boss_health, player_action, boss_action

# Main loop
start_time = time.time()
action_start_times.append(start_time)  # Track action start times

# Define custom blood window for boss (user must specify)
blood_window2  = (509, 632, 754, 638)


frame_count = 0  # Initialize frame counter to lower the update frequency
while True:
    current_time = time.time() - start_time
    boss_screen_image = grab_screen(blood_window2)
    cv2.imshow("Boss Blood Region", boss_screen_image)

    # Simulate a game frame with actual blood detection
    # Pass the necessary arguments: model, screen_region, and action_labels
    boss_health, player_action, boss_action = simulate_game_frame(
        blood_window2, model, screen_region, action_labels)
    print(f"Boss Health before plot update: {boss_health}")    
    # Only update plot every few frames to reduce frequency
    if frame_count % 5 == 0:
        update_plot(current_time, boss_health, player_action, boss_action)
    
    frame_count += 1
    
    # Break the loop after 600 seconds (for demo purposes)
    if current_time > 600:
        break

# Close plot after completion
plt.ioff()
plt.show()
