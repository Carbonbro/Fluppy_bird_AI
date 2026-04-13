import pyautogui
import time
import func
 
def jump():
    pyautogui.press("space")

def posMonitor(): # get your position of your mouse
    while 1:
        print(pyautogui.position())

def clickTest():
    time.sleep(3)
    for i in range(5):
        pyautogui.click(500,850)
        pyautogui.press('space')
        print("s\n")
        time.sleep(1)

def main():
    #time.sleep(3) 
    #posMonitor() 
    func.process()

main()