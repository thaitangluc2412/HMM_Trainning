from pynput.mouse import Button, Controller
from hmm_train import predict, record, count_THESHOLD
import sys
import keyboard
import time 
mouse = Controller()
while True:
    record(count_THESHOLD())
    command = predict()
    if (command=="trai"):
        mouse.press(Button.left)
        mouse.release(Button.left)
    elif (command=="phai"):
        mouse.press(Button.right)
        mouse.release(Button.right)
    elif (command=="nha"):
        mouse.release(Button.left)
    elif (command=="giu"):
        mouse.press(Button.left)
    elif(command=="dup") :
        mouse.click(Button.left, 2)
    if keyboard.is_pressed('Esc'):
            print("\nyou pressed Esc, so exiting...")
            sys.exit(0):
