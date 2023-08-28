from pyfirmata import Arduino, SERVO
from time import sleep
PORT='COM3'
pin=11

board=Arduino(PORT)

board.digital[pin].mode=SERVO

def rotateServo(pin, angle):
    board.digital[pin].write(angle)

def wasteOpening(val):
     if val==1:
        for i in range(0,180):
           sleep(0.0005)
           rotateServo(pin,i)

def wasClosing(val):
      if val==0:        
        #speech("The waste box door is closing now")
        for i in range(180,1,-1):
                sleep(0.005)
                rotateServo(pin,i)
                #pisleep(0.01)

