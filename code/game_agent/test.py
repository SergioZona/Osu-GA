import pyautogui
import keyboard
import time
import sched


from multiprocessing import Process

scheduler = sched.scheduler(time.time, time.sleep)

def move_mouse_path_with_repetitions(points, repetitions, time):
    # Compute total distance of the path
    reversed_points = list(reversed(points))

    total_distance = 0
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        total_distance += ((x2-x1)**2 + (y2-y1)**2)**0.5
    
    total_distance = total_distance if repetitions == 0 else total_distance*repetitions

    # Compute time per unit distance - x=vt -> v=x/t
    velocity = total_distance/time
    
    for i in range(0, repetitions + 1):
        if i % 2 == 0:
            # iterate through the list forwards
            for j in range(len(points)):
                x1, y1 = points[j]

                if j < len(points) - 1:
                    # Move to the next point
                    x2, y2 = points[j+1]
                    distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
                    duration = distance / velocity
                    pyautogui.mouseDown(x=x1, y=y1, button='left')
                    pyautogui.moveTo(x2, y2, duration=duration)

        elif i % 2 == 1:
            # iterate through the list forwards
            for j in range(len(reversed_points)):
                x1, y1 = reversed_points[j]

                if j < len(reversed_points) - 1:
                    # Move to the next point
                    x2, y2 = reversed_points[j+1]
                    distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
                    duration = distance / velocity
                    pyautogui.mouseDown(x=x1, y=y1, button='left')
                    pyautogui.moveTo(x2, y2, duration=duration)
        
    pyautogui.mouseUp()
    
if __name__ == '__main__':
    # The failsafe allows you to move the cursor on the upper left corner of the screen to terminate the program. 
    # It is STRONGLY RECOMMENDED to keep it True
    pyautogui.FAILSAFE = True
    
    points = [(960, 540), (800, 400), (1120, 200)]
    repetitions = 1
    total_time = 0.3

    # Spawn and start the process
    #execute_program_process = Process(target=move_mouse_path_with_repetitions(points, repetitions, total_time))
    #execute_program_process.start()

    # Call the click_action function at the specified time
    scheduler.enter(2, 1, move_mouse_path_with_repetitions, (points, repetitions, total_time))

    # Call the click_action function at the specified time
    scheduler.enter(3, 2, move_mouse_path_with_repetitions, (points, repetitions, total_time))

        # Call the click_action function at the specified time
    scheduler.enter(3.5, 1, move_mouse_path_with_repetitions, (points, repetitions, total_time))

    # Run the scheduled tasks
    scheduler.run()

    while True:
        if keyboard.is_pressed('ctrl+c'):
            #execute_program_process.terminate()
            break

    print('\nDone.')