import mouse
from screeninfo import get_monitors


screens = get_monitors()

print(screens[0])

# mouse.move(100, 100, absolute=True, duration=0.1)
